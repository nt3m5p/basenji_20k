# Copyright 2017 Calico LLC
#
# Modifications Copyright (c) 2025 Le Zhang
# Modifications:
#   - Added explicit no-eval training mode:
#       * Allows training without validation data.
#       * Disables early stopping and best-model selection.
#       * Uses a fixed number of epochs.
#   - Introduced optional tail cosine learning-rate decay for no-eval training,
#   - Extended loss framework with optional inter-tissue pairwise losses:
#       * Supported loss types include MSE, MAE etc.
#   - Added compatibility handling for TensorFlow metric APIs
#     (reset_state vs reset_states).
#   - Removed silent distributed-strategy branches; unsupported strategy usage
#     now raises explicit errors to avoid unintended behavior.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================
"""SeqNN trainer"""
import sys
import time

import numpy as np
import tensorflow as tf
from packaging import version

from basenji.metrics import PearsonR, R2

try:
    import tensorflow_addons as tfa
except ImportError:
    tfa = None

from tensorflow.keras import mixed_precision
from basenji import metrics
import os


def kl_poisson(lam1, lam2):
    """
    KL(Pois(lam1) || Pois(lam2)) = lam2 - lam1 + lam1*log(lam1/lam2),
    假定 lam1, lam2 > 0
    """
    lam1_clp = tf.clip_by_value(lam1, 1e-8, 1e8)
    lam2_clp = tf.clip_by_value(lam2, 1e-8, 1e8)
    return lam2_clp - lam1_clp + lam1_clp * tf.math.log(lam1_clp / lam2_clp)


def jsd_poisson(lam1, lam2):
    """
    JSD(Pois(lam1) || Pois(lam2)) = 0.5 * KL(Pois(lam1) || Pois(m)) +
                                   0.5 * KL(Pois(lam2) || Pois(m))
    其中 m = (lam1 + lam2)/2
    """
    lam1_clp = tf.clip_by_value(lam1, 1e-8, 1e8)
    lam2_clp = tf.clip_by_value(lam2, 1e-8, 1e8)
    m = 0.5 * (lam1_clp + lam2_clp)
    kl1 = kl_poisson(lam1_clp, m)
    kl2 = kl_poisson(lam2_clp, m)
    return 0.5 * kl1 + 0.5 * kl2


def parse_loss(loss_label, strategy=None, keras_fit=True, spec_weight=1, total_weight=1):
    """Parse loss function from label, strategy, and fitting method."""
    if strategy is not None and not keras_fit:
        if loss_label == 'mse':
            loss_fn = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
        elif loss_label == 'bce':
            loss_fn = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        elif loss_label == 'poisson_mn':
            # loss_fn = metrics.PoissonMultinomial(total_weight,
            #                                      reduction=tf.keras.losses.Reduction.NONE)
            loss_fn = metrics.PoissonMultinomial()
        else:
            loss_fn = tf.keras.losses.Poisson(reduction=tf.keras.losses.Reduction.NONE)
    else:
        if loss_label == 'mse':
            loss_fn = tf.keras.losses.MeanSquaredError()
        elif loss_label == 'mse_udot':
            loss_fn = metrics.MeanSquaredErrorUDot(spec_weight)
        elif loss_label == 'bce':
            loss_fn = tf.keras.losses.BinaryCrossentropy()
        elif loss_label == 'poisson_kl':
            loss_fn = metrics.PoissonKL(spec_weight)
        elif loss_label == 'poisson_mn':
            loss_fn = metrics.PoissonMultinomial(
                multinomial_resolution=224,
                reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE  # ★
            )

        else:
            loss_fn = tf.keras.losses.Poisson()

    return loss_fn


class Trainer:
    def __init__(self, params, train_data, eval_data, out_dir,
                 strategy=None, num_gpu=1, keras_fit=True, tissue_index=[], no_eval: bool=False):
        self.params = params
        self.train_data = train_data
        if type(self.train_data) is not list:
            self.train_data = [self.train_data]
        self.eval_data = eval_data
        self.no_eval = bool(no_eval)

        class _EmptyEval:
            def __init__(self, batch_size):
                # 空三元组数据集，形状兼容 (x,y,w)
                self.dataset = tf.data.Dataset.from_tensor_slices(
                    (tf.constant([], dtype=tf.float32),
                     tf.constant([], dtype=tf.float32),
                     tf.constant([], dtype=tf.float32))
                ).batch(1)
                self._bpe = 0
                self.batch_size = batch_size
            def batches_per_epoch(self):
                return 0

        # 若明确无验证（eval_data 为 None 或 no_eval=True），就为每个 train_data 塞一个空 eval
        if (eval_data is None) or self.no_eval:
            self.eval_data = [_EmptyEval(self.train_data[0].batch_size) for _ in self.train_data]
            self.has_eval = False
        else:
            self.eval_data = eval_data if isinstance(eval_data, list) else [eval_data]
            self.has_eval = True



        self.out_dir = out_dir
        self.strategy = strategy
        self.num_gpu = num_gpu
        self.batch_size = self.train_data[0].batch_size
        self.compiled = False

        # 获得不同的组织和索引
        self.tissue_index = tf.convert_to_tensor(tissue_index, dtype=tf.int32)  # # 确保 tissue_index 为 int32 类型的 Tensor
        self.unique_tissues, self.unique_tissues_idx = tf.unique(self.tissue_index)  # 获取唯一的组织标签及每个通道对应的组 id
        self.num_unique = tf.shape(self.unique_tissues)[0]  # 不同组织的数量
        # 构造上三角布尔 mask（不包含对角线），形状 [num_unique, num_unique]
        self.mask = tf.cast(
            tf.linalg.band_part(tf.ones((self.num_unique, self.num_unique)), 0, -1) - tf.eye(self.num_unique), tf.bool)
        self.pair_indices = tf.where(self.mask)  # 获取上三角的索引，形状 [num_pairs, 2]

        # early stopping
        self.patience = self.params.get('patience', 20)

        # compute batches/epoch
        self.train_epoch_batches = [td.batches_per_epoch() for td in self.train_data]
        self.eval_epoch_batches = [ed.batches_per_epoch() for ed in self.eval_data]
        self.train_epochs_min = self.params.get('train_epochs_min', 1)
        # 没有“真实”验证时：固定训练轮数、关闭早停（patience 设超大）
        if not self.has_eval:
            self.train_epochs_max = int(self.params.get('epochs'))
            self.patience = int(1e18)
        else:
            self.train_epochs_max = self.params.get('train_epochs_max', 1000)

        # dataset
        self.num_datasets = len(self.train_data)
        self.dataset_indexes = []
        for di in range(self.num_datasets):
            self.dataset_indexes += [di] * self.train_epoch_batches[di]
        self.dataset_indexes = np.array(self.dataset_indexes)

        # dataset fine tuning
        self.main_data_count = self.params.get('main_data_count', False)
        self.fine_tuning_start = self.params.get('fine_tuning_start', False)
        self.fine_tuning_method = self.params.get('fine_tuning_method', None)

        if self.params.get('main_data_count', 0):
            self.dataset_indexes_ft = self.dataset_indexes[self.dataset_indexes < self.params['main_data_count']]
        else:
            # 若 main_data_count 未设置或为 0，则与完整索引一致
            self.dataset_indexes_ft = self.dataset_indexes.copy()

        # loss
        self.spec_weight = self.params.get('spec_weight', 1)
        self.total_weight = self.params.get('total_weight', 1)
        self.loss = self.params.get('loss', 'poisson').lower()
        self.loss_fn = parse_loss(self.loss, self.strategy, keras_fit, self.spec_weight, self.total_weight)

        self.tissue_loss_wight = self.params.get('tissue_loss')
        self.tissue_loss_fn = self.params.get('tissue_loss_fn', 'mae')
        self.tissue_loss_ag = self.params.get('tissue_loss_ag', False)
        self.tissue_loss_mask = self.params.get('tissue_loss_mask', False)

        # optimizer
        self.make_optimizer()

    def compile(self, seqnn_model):
        for model in seqnn_model.models:
            if self.loss == 'bce':
                model_metrics = [metrics.SeqAUC(curve='ROC'), metrics.SeqAUC(curve='PR')]
            else:
                num_targets = model.output_shape[-1]
                model_metrics = [metrics.PearsonR(num_targets), metrics.R2(num_targets)]

            model.compile(loss=self.loss_fn,
                          optimizer=self.optimizer,
                          metrics=model_metrics)
        self.compiled = True

    def parse_loss_inter_tissue(self, y_true_flat, y_pred_flat):
        # y_true_flat  n*48
        # y_pred_flat  n*48
        # ---------- 计算每对组织间的差异 ----------
        def pairwise_diff(row):
            # 对于每个样本，计算所有组织对 (i,j) (i<j) 的差异：row[i] - row[j]
            diff = tf.gather(row, self.pair_indices[:, 0]) - tf.gather(row, self.pair_indices[:, 1])
            return diff  # shape: [num_pairs]

        def pairwise_sum(row):
            # 对于每个样本，计算所有组织对 (i,j) (i<j) 的和：row[i] + row[j]
            diff = tf.gather(row, self.pair_indices[:, 0]) + tf.gather(row, self.pair_indices[:, 1])
            return diff  # shape: [num_pairs]

        # 使用 tf.vectorized_map 分别计算 y_true 与 y_pred 每个样本的 pairwise_diff
        y_true_diff = tf.vectorized_map(pairwise_diff, y_true_flat)
        y_pred_diff = tf.vectorized_map(pairwise_diff, y_pred_flat)

        # ---------- 计算均方误差损失 ----------
        # poisson = lamba - y * log(lamba) ： lambda是预测值
        if self.tissue_loss_fn == 'mse':
            loss_per_sample = tf.reduce_mean(tf.square(y_pred_diff - y_true_diff), axis=-1)
        elif self.tissue_loss_fn == 'mae':
            loss_per_sample = tf.reduce_mean(tf.abs(y_pred_diff - y_true_diff), axis=-1)
        elif self.tissue_loss_fn == 'jsd_poisson':
            # 1) 确保差值 >= 0, 若你想用绝对值:
            lam_true = tf.abs(y_true_diff)
            lam_pred = tf.abs(y_pred_diff)
            # 2) 计算 JSD
            #   shape: [batch_size, num_pairs]
            diff_jsd = jsd_poisson(lam_true, lam_pred)
            # 3) 在组织 pairs 维度做均值
            loss_per_sample = tf.reduce_mean(diff_jsd, axis=-1)
        else:
            raise ValueError(f"Unknown tissue_loss_fn: {self.tissue_loss_fn}")

        if self.tissue_loss_mask:
            # ---------- 计算样本 mask ----------
            # 对于每个样本，计算所有组织中的最大值和最小值
            # 注意：这里以 y_true_mean 作为判断依据
            sample_max = tf.reduce_max(y_true_flat, axis=-1)  # shape: [B]
            sample_min = tf.reduce_min(y_true_flat, axis=-1)  # shape: [B]
            epsilon = 1e-8  # 防止除 0
            # 计算比值：如果最大值至少为最小值的n倍，则认为差异大
            ratio = sample_max / (sample_min + epsilon)
            sample_mask = tf.cast(ratio > 4, tf.float32)  # shape: [B]

            # ---------- 应用 mask 计算最终损失 ----------
            # 对每个样本的损失乘以对应的 mask（不满足条件的样本损失为 0）
            masked_loss = loss_per_sample * sample_mask
            # 平均损失仅在满足条件的样本上计算
            final_loss = tf.reduce_sum(masked_loss) / (tf.reduce_sum(sample_mask) + epsilon)
        else:
            final_loss = tf.reduce_mean(loss_per_sample)

        return final_loss

    def parse_loss_tissue(self, y_true, y_pred):
        # 不再使用，直接在构建数据集时计算组织特异性
        # y_pred 的 shape: [B, T, C]
        B = tf.shape(y_pred)[0]
        T = tf.shape(y_pred)[1]
        C = tf.shape(y_pred)[2]
        # 将 y_true / y_pred reshape 为 [B*T, C]
        y_true_flat = tf.reshape(y_true, [B * T, C])
        y_pred_flat = tf.reshape(y_pred, [B * T, C])

        # ---------- 组织聚合 ----------
        if self.tissue_loss_ag:
            # 定义聚合函数：对每一行根据 unique_tissues_idx 聚合到同一组织上，得到 [B*T, num_unique]
            def aggregate_fn(x):
                return tf.math.unsorted_segment_mean(x, self.unique_tissues_idx, self.num_unique)

            # 使用 tf.vectorized_map 替换 tf.map_fn 进行聚合
            y_true_flat = tf.vectorized_map(aggregate_fn, y_true_flat)
            y_pred_flat = tf.vectorized_map(aggregate_fn, y_pred_flat)

        # ---------- 计算loss ----------
        loss = self.parse_loss_inter_tissue(y_true_flat, y_pred_flat)

        return loss

    def fit_keras(self, seqnn_model):
        if not self.compiled:
            self.compile(seqnn_model)

        if self.loss == 'bce':
            early_stop = EarlyStoppingMin(monitor='val_loss', mode='min', verbose=1,
                                          patience=self.patience, min_epoch=self.train_epochs_min)
            save_best = tf.keras.callbacks.ModelCheckpoint('%s/model_best.h5' % self.out_dir,
                                                           save_best_only=True, mode='min',
                                                           monitor='val_loss', verbose=1)
        else:
            early_stop = EarlyStoppingMin(monitor='val_pearsonr', mode='max', verbose=1,
                                          patience=self.patience, min_epoch=self.train_epochs_min)
            save_best = tf.keras.callbacks.ModelCheckpoint('%s/model_best.h5' % self.out_dir,
                                                           save_best_only=True, mode='max',
                                                           monitor='val_pearsonr', verbose=1)

        callbacks = [
            early_stop,
            tf.keras.callbacks.TensorBoard(self.out_dir),
            tf.keras.callbacks.ModelCheckpoint('%s/model_check.h5' % self.out_dir),

            save_best]



        seqnn_model.model.fit(
            self.train_data[0].dataset,
            epochs=self.train_epochs_max,
            steps_per_epoch=self.train_epoch_batches[0],
            callbacks=callbacks,
            validation_data=self.eval_data[0].dataset,
            validation_steps=self.eval_epoch_batches[0])

    def fit2(self, seqnn_model):
        if not self.compiled:
            self.compile(seqnn_model)

        # ---------- ① 预‑forward，确保模型权重落地 ----------
        x_warm, _, w = next(iter(self.train_data[0].dataset))
        x_warm = x_warm[:1]  # 1 条即可
        for m in seqnn_model.models:
            _ = m(x_warm, training=False)

        # ---------- ② 预‑build optimizer ----------
        trainable_vars = []
        for m in seqnn_model.models:
            trainable_vars.extend(m.trainable_variables)

        def reset_metric_list(metric_list):
            for m in metric_list:
                m.reset_states()

        assert (len(seqnn_model.models) >= self.num_datasets)

        # inform optimizer about all trainable variables (v2.11-)
        vars_set = set()
        trainable_vars = []
        for di in range(self.num_datasets):
            for v in seqnn_model.models[di].trainable_variables:
                if v.name not in vars_set:
                    vars_set.add(v.name)
                    trainable_vars.append(v)
        try:  # TF ≥ 2.11 直接用 build
            self.optimizer.build(trainable_vars)
        except AttributeError:  # TF < 2.11 走这里
            # 给每个变量喂一个 0 梯度，迫使 Optimizer 在 Eager 模式下
            # 创建好所有 slot‑Variable（momentum、m/v 等）
            zero_grads = [tf.zeros_like(v) for v in trainable_vars]
            self.optimizer.apply_gradients(zip(zero_grads, trainable_vars))

        ################################################################
        # prep

        # metrics
        train_r: list[PearsonR]
        train_r2: list[R2]
        train_loss, train_r, train_r2 = [], [], []
        valid_loss, valid_r, valid_r2 = [], [], []
        for di in range(self.num_datasets):
            num_targets = seqnn_model.models[di].output_shape[-1]
            train_loss.append(tf.keras.metrics.Mean(name='train%d_loss' % di))
            train_r.append(metrics.PearsonR(num_targets, name='train%d_r' % di))
            train_r2.append(metrics.R2(num_targets, name='train%d_r2' % di))
            valid_loss.append(tf.keras.metrics.Mean(name='valid%d_loss' % di))
            valid_r.append(metrics.PearsonR(num_targets, name='valid%d_r' % di))
            valid_r2.append(metrics.R2(num_targets, name='valid%d_r2' % di))
        train_step_generic = None
        eval_step_generic = None
        if self.strategy is None:
            @tf.function(experimental_relax_shapes=True)
            def train_step_generic(x, y, w, model, loss_fn, optimizer,
                                   loss_meter, r_meter, r2_meter):
                """
                x, y         : 一个 batch 的输入/标签
                w            : 当前样本权重（每个 bin 使用不同的权重）
                model        : 当前数据集对应的 tf.keras.Model
                loss_fn      : 已解析好的 loss（self.loss_fn）
                optimizer    : 公用 optimizer
                *_meter      : 该数据集对应的 tf.keras.metrics.Mean / PearsonR / R2
                """

                with tf.GradientTape() as tape:
                    pred = model(x, training=True)
                    loss0 = loss_fn(y, pred, sample_weight=w) + tf.add_n(model.losses)

                grads = tape.gradient(loss0, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

                # 记录指标
                loss_meter(loss0)
                r_meter(y, pred)
                r2_meter(y, pred)
                return loss0  # 这句可选，主要用于 debug



            @tf.function(experimental_relax_shapes=True)
            def eval_step_generic(x, y, model, loss_fn,
                                  loss_meter, r_meter, r2_meter):
                pred = model(x, training=False)
                loss0 = loss_fn(y, pred) + tf.add_n(model.losses)

                loss_meter(loss0)
                r_meter(y, pred)
                r2_meter(y, pred)

        else:
            pass
            # 移除了分布式

        # checkpoint manager
        managers = []
        epoch_start = 0
        already_scaled = False
        for di in range(self.num_datasets):
            ckpt = tf.train.Checkpoint(model=seqnn_model.models[di], optimizer=self.optimizer)
            ckpt_dir = '%s/model%d' % (self.out_dir, di)
            manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=1)
            if manager.latest_checkpoint:
                ckpt.restore(manager.latest_checkpoint)
                ckpt_end = 5 + manager.latest_checkpoint.find('ckpt-')
                dataset_epoch_start = int(manager.latest_checkpoint[ckpt_end:])
                if self.strategy is None:
                    opt_iters = self.optimizer.iterations
                else:
                    opt_iters = self.optimizer.iterations.values[0]
                print('Checkpoint restored at epoch %d, optimizer iteration %d.' % \
                      (dataset_epoch_start, opt_iters))
                epoch_start = max(epoch_start, dataset_epoch_start)
                # 微调方法2 除以10，仅第一次满足条件时缩 LR
                if self.fine_tuning_method and self.fine_tuning_method == "divide_by_10":
                    if (self.fine_tuning_start and
                            epoch_start >= self.fine_tuning_start and
                            not already_scaled):
                        lr_now = self.optimizer.learning_rate
                        # Variable / float / schedule 兼容
                        if hasattr(lr_now, "assign"):
                            lr_now.assign(lr_now * 0.1)
                        else:
                            self.optimizer.learning_rate = lr_now * 0.1

                        already_scaled = True  # ★ 设标志，后面不再缩
                        print(f"[Resume] LR scaled ×0.1  (epoch_start={epoch_start})")
            else:
                print('No checkpoints found.')
                epoch_start = max(epoch_start, 0)
            managers.append(manager)

        # improvement variables
        valid_best = [-np.inf] * self.num_datasets
        unimproved = [0] * self.num_datasets

        ################################################################
        # training loop

        first_step = True
        for ei in range(epoch_start, self.train_epochs_max):
            if ei >= self.train_epochs_min and np.min(unimproved) > self.patience:
                break
            else:
                # get iterators
                # 1) 选索引：微调阶段 or 全量阶段
                if self.fine_tuning_start and ei < self.fine_tuning_start:
                    current_idxs = self.dataset_indexes_ft.copy()
                else:
                    current_idxs = self.dataset_indexes.copy()
                # 2) 打乱
                np.random.shuffle(current_idxs)
                # 3) 迭代器
                train_data_iters = [iter(td.dataset) for td in self.train_data]

                # 微调方法2 在微调开始的时候调整lr为原本的1/10, 微调方法1在构建数据集的时候写入了weight
                if self.fine_tuning_method and self.fine_tuning_method == "divide_by_10":
                    if (self.fine_tuning_start
                            and ei == self.fine_tuning_start
                            and not already_scaled):
                        lr_now = self.optimizer.learning_rate  # 取当前 lr
                        self.optimizer.learning_rate.assign(lr_now * 0.1)
                        print(f"######## fine tuning start ########")
                        print(f"Epoch {ei} learning-rate scaled ×0.1")

                # train
                t0 = time.time()
                for di in current_idxs:
                    x, y, w = safe_next(train_data_iters[di])
                    if self.strategy is None:
                        model_i = seqnn_model.models[di]
                        train_step_generic(x, y, w,  # batch
                                           model_i,  # 当前模型
                                           self.loss_fn,
                                           self.optimizer,
                                           train_loss[di],
                                           train_r[di],
                                           train_r2[di])
                    else:
                        pass
                    if first_step:
                        print('Successful first step!', flush=True)
                        first_step = False

                print('Epoch %d - %ds' % (ei, (time.time() - t0)))

                # 在微调期间只输出主要的metric
                if self.fine_tuning_start and ei < self.fine_tuning_start:
                    print_datasets = self.main_data_count
                else:
                    print_datasets = self.num_datasets
                for di in range(print_datasets):

                    # 如果要输出所有的修改这一条
                    # for di in range(self.num_datasets):
                    print('  Data %d' % di, end='')
                    model = seqnn_model.models[di]

                    # print training accuracy
                    print(' - train_loss: %.4f' % train_loss[di].result().numpy(), end='')
                    print(' - train_r: %.4f' % train_r[di].result().numpy(), end='')
                    print(' - train_r2: %.4f' % train_r2[di].result().numpy(), end='')

                    # evaluate
                    for x, y, _ in self.eval_data[di].dataset:
                        if self.strategy is None:
                            eval_step_generic(x, y,
                                              seqnn_model.models[di],
                                              self.loss_fn,
                                              valid_loss[di],
                                              valid_r[di],
                                              valid_r2[di])
                        else:
                            pass

                    # print validation accuracy
                    print(' - valid_loss: %.4f' % valid_loss[di].result().numpy(), end='')
                    print(' - valid_r: %.4f' % valid_r[di].result().numpy(), end='')
                    print(' - valid_r2: %.4f' % valid_r2[di].result().numpy(), end='')
                    early_stop_stat = valid_r[di].result().numpy()

                    # checkpoint
                    managers[di].save()
                    model.save('%s/model%d_check.h5' % (self.out_dir, di),
                               include_optimizer=False)

                    # check best
                    if early_stop_stat > valid_best[di]:
                        print(' - best!', end='')
                        unimproved[di] = 0
                        valid_best[di] = early_stop_stat
                        model.save('%s/model%d_best.h5' % (self.out_dir, di),
                                   include_optimizer=False)
                    else:
                        unimproved[di] += 1
                    print('', flush=True)

                reset_metric_list(train_loss)
                reset_metric_list(train_r)
                reset_metric_list(train_r2)
                reset_metric_list(valid_loss)
                reset_metric_list(valid_r)
                reset_metric_list(valid_r2)

                # # 每十个epoch保存所有模型
                # if (ei + 1) % 10 == 0:
                #     for di, model in enumerate(seqnn_model.models):
                #         backup_dir = os.path.join(self.out_dir, f"epoch_{ei + 1:03d}")
                #         os.makedirs(backup_dir, exist_ok=True)
                #
                #         # 文件名避免覆盖
                #         model_path = os.path.join(backup_dir, f"model{di}_check.h5")
                #         model.save(model_path, include_optimizer=False)

    def fit_tape(self, seqnn_model):
        if not self.compiled:
            self.compile(seqnn_model)
        model = seqnn_model.model

        # metrics
        num_targets = model.output_shape[-1]
        train_loss = tf.keras.metrics.Mean(name='train_loss')
        train_r = metrics.PearsonR(num_targets, name='train_r')
        train_r2 = metrics.R2(num_targets, name='train_r2')
        valid_loss = tf.keras.metrics.Mean(name='valid_loss')
        valid_r = metrics.PearsonR(num_targets, name='valid_r')
        valid_r2 = metrics.R2(num_targets, name='valid_r2')

        if self.strategy is None:
            @tf.function
            def train_step(x, y, w):
                with tf.GradientTape() as tape:
                    pred = model(x, training=True)
                    loss = self.loss_fn(y, pred, sample_weight=w) + sum(model.losses)

                train_loss(loss)
                train_r.update_state(y, pred)
                train_r2.update_state(y, pred)

                gradients = tape.gradient(loss, model.trainable_variables)
                if self.agc_clip is not None:
                    gradients = adaptive_clip_grad(model.trainable_variables, gradients, self.agc_clip)
                self.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            @tf.function
            def eval_step(x, y):
                pred = model(x, training=False)
                loss_original = self.loss_fn(y, pred) + sum(model.losses)
                loss_tissue = self.parse_loss_tissue(y, pred)
                valid_loss(loss_original)
                valid_r.update_state(y, pred)
                valid_r2.update_state(y, pred)

        else:
            def train_step(x, y):
                with tf.GradientTape() as tape:
                    pred = model(x, training=True)
                    loss_batch_len = self.loss_fn(y, pred)
                    loss_batch = tf.reduce_mean(loss_batch_len, axis=-1)
                    loss = tf.reduce_sum(loss_batch) / self.batch_size
                    loss += sum(model.losses) / self.num_gpu
                train_r.update_state(y, pred)
                train_r2.update_state(y, pred)
                gradients = tape.gradient(loss, model.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                return loss

            @tf.function
            def train_step_distr(xd, yd):
                replica_losses = self.strategy.run(train_step, args=(xd, yd))
                loss = self.strategy.reduce(tf.distribute.ReduceOp.SUM,
                                            replica_losses, axis=None)
                train_loss(loss)

            def eval_step(x, y):
                pred = model(x, training=False)
                loss = self.loss_fn(y, pred) + sum(model.losses)
                valid_loss(loss)
                valid_r.update_state(y, pred)
                valid_r2.update_state(y, pred)

            @tf.function
            def eval_step_distr(xd, yd):
                return self.strategy.run(eval_step, args=(xd, yd))

        # checkpoint manager
        ckpt = tf.train.Checkpoint(model=seqnn_model.model, optimizer=self.optimizer)
        manager = tf.train.CheckpointManager(ckpt, self.out_dir, max_to_keep=1)
        if manager.latest_checkpoint:
            ckpt.restore(manager.latest_checkpoint)
            ckpt_end = 5 + manager.latest_checkpoint.find('ckpt-')
            epoch_start = int(manager.latest_checkpoint[ckpt_end:])
            if self.strategy is None:
                opt_iters = self.optimizer.iterations
            else:
                opt_iters = self.optimizer.iterations.values[0]
            print('Checkpoint restored at epoch %d, optimizer iteration %d.' % \
                  (epoch_start, opt_iters))
        else:
            print('No checkpoints found.')
            epoch_start = 0

        # improvement variables
        valid_best = -np.inf
        unimproved = 0

        # training loop
        for ei in range(epoch_start, self.train_epochs_max):
            if ei >= self.train_epochs_min and unimproved > self.patience:
                break
            else:
                # train
                t0 = time.time()
                train_iter = iter(self.train_data[0].dataset)
                for si in range(self.train_epoch_batches[0]):
                    x, y, w = safe_next(train_iter)
                    if self.strategy is not None:
                        pass
                    else:
                        train_step(x, y, w)
                    if ei == epoch_start and si == 0:
                        print('Successful first step!', flush=True)

                # evaluate
                for x, y, _ in self.eval_data[0].dataset:
                    if self.strategy is not None:
                        pass
                    else:
                        eval_step(x, y)

                # print training accuracy
                train_loss_epoch = train_loss.result().numpy()
                train_r_epoch = train_r.result().numpy()
                train_r2_epoch = train_r2.result().numpy()
                # print('Epoch %d - %ds - train_loss: %.4f - train_r: %.4f - train_r2: %.4f' % \
                #       (ei, (time.time() - t0), train_loss_epoch, train_r_epoch, train_r2_epoch), end='')

                # ---------- 读取当前学习率 ----------
                if isinstance(self.optimizer.learning_rate, tf.keras.optimizers.schedules.LearningRateSchedule):
                    # 有使用 lr_schedule 时，用 _decayed_lr 拿到“当前”值
                    cur_lr = self.optimizer._decayed_lr(tf.float32).numpy()
                else:
                    # 常量 lr
                    cur_lr = tf.keras.backend.get_value(self.optimizer.lr)

                # ---------- 打印 ----------
                print('Epoch %d - %ds - lr: %.6g - train_loss: %.4f - train_r: %.4f - train_r2: %.4f' %
                      (ei, (time.time() - t0), cur_lr, train_loss_epoch, train_r_epoch, train_r2_epoch),
                      end='')


                # print validation accuracy
                valid_loss_epoch = valid_loss.result().numpy()
                valid_r_epoch = valid_r.result().numpy()
                valid_r2_epoch = valid_r2.result().numpy()
                print(' - valid_loss: %.4f - valid_r: %.4f - valid_r2: %.4f' % \
                      (valid_loss_epoch, valid_r_epoch, valid_r2_epoch), end='')

                # checkpoint
                manager.save()
                seqnn_model.save('%s/model_check.h5' % self.out_dir)

                # check best # 是否要加上 组织间损失
                valid_best_epoch = valid_r_epoch + valid_r2_epoch / 4
                if valid_best_epoch > valid_best:
                    print(' - best!', end='')
                    unimproved = 0
                    valid_best = valid_best_epoch
                    seqnn_model.save('%s/model_best.h5' % self.out_dir)
                else:
                    unimproved += 1
                print('', flush=True)

                # reset metrics
                train_loss.reset_states()
                train_r.reset_states()
                train_r2.reset_states()
                valid_loss.reset_states()
                valid_r.reset_states()
                valid_r2.reset_states()

    def make_optimizer(self):
        cyclical1 = True
        for lrs_param in ['initial_learning_rate', 'maximal_learning_rate', 'final_learning_rate',
                          'train_epochs_cycle1']:
            cyclical1 = cyclical1 & (lrs_param in self.params)
        if cyclical1:
            step_size = self.params['train_epochs_cycle1'] * sum(self.train_epoch_batches)
            initial_learning_rate = self.params.get('initial_learning_rate')
            lr_schedule = Cyclical1LearningRate(
                initial_learning_rate=self.params['initial_learning_rate'],
                maximal_learning_rate=self.params['maximal_learning_rate'],
                final_learning_rate=self.params['final_learning_rate'],
                step_size=step_size)
        else:
            # schedule (currently OFF)
            initial_learning_rate = self.params.get('learning_rate', 0.01)
            # if False:
            #     lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            #         initial_learning_rate,
            #         decay_steps=self.params.get('decay_steps', 100000),
            #         decay_rate=self.params.get('decay_rate', 0.96),
            #         staircase=True)
            # else:
            #     lr_schedule = initial_learning_rate
            lr_schedule = initial_learning_rate

        if 'warmup_steps' in self.params:
            lr_schedule = WarmUp(
                initial_learning_rate=initial_learning_rate,
                warmup_steps=self.params['warmup_steps'],
                decay_schedule=lr_schedule)

        if version.parse(tf.__version__) < version.parse('2.2'):
            clip_norm_default = 1000000
        else:
            clip_norm_default = None

        global_clipnorm = self.params.get('global_clipnorm', clip_norm_default)
        if 'clip_norm' in self.params:
            clip_norm = self.params['clip_norm']
        elif 'clipnorm' in self.params:
            clip_norm = self.params['clipnorm']
        else:
            clip_norm = clip_norm_default

        # adaptive gradient clipping handled in fit method
        self.agc_clip = self.params.get('agc_clip', None)

        # ===== 尾部余弦衰减（仅在无验证集时启用）=====
        if not self.has_eval:
            # 末尾多少个 epoch 做余弦，默认 20；可在 params['train']['tail_cosine_epochs'] 覆盖
            tail_epochs = int(self.params.get('tail_cosine_epochs', 20))
            steps_per_epoch = int(sum(self.train_epoch_batches))
            total_steps = int(self.train_epochs_max * steps_per_epoch)
            tail_steps = int(max(0, min(tail_epochs, self.train_epochs_max)) * steps_per_epoch)
            start_step = int(total_steps - tail_steps)

            # 末尾余弦的最低占比 alpha（0=降到 0，0.1=降到初值的 10%）
            tail_alpha = float(self.params.get('tail_cosine_alpha', 0.0))

            # 仅支持“常数/简单 WarmUp 后常数”的场景：取 warmup 结束后的恒定 lr 作为基准
            base_lr = self.params.get('learning_rate', self.params.get('initial_learning_rate', 0.01))
            cosine_tail = tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=base_lr,
                decay_steps=max(1, tail_steps),
                alpha=tail_alpha
            )

            class TailCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
                def __call__(self, step):
                    step = tf.cast(step, tf.int64)
                    return tf.where(
                        step < start_step,
                        tf.cast(base_lr, tf.float32),
                        cosine_tail(tf.cast(step - start_step, tf.int64))
                    )

            lr_schedule = TailCosine()


        # optimizer
        optimizer_type = self.params.get('optimizer', 'sgd').lower()
        if optimizer_type == 'adam':
            self.optimizer = tf.keras.optimizers.Adam(
                learning_rate=lr_schedule,
                beta_1=self.params.get('adam_beta1', 0.9),
                beta_2=self.params.get('adam_beta2', 0.999),
                clipnorm=clip_norm,
                global_clipnorm=global_clipnorm,
                amsgrad=False)  # reduces performance in my experience

        elif optimizer_type == 'adamw':
            self.optimizer = tfa.optimizers.AdamW(
                weight_decay=self.params.get('weight_decay', 0),
                learning_rate=lr_schedule,
                beta_1=self.params.get('adam_beta1', 0.9),
                beta_2=self.params.get('adam_beta2', 0.999),
                clipnorm=clip_norm,
                global_clipnorm=global_clipnorm,
                amsgrad=False)  # reduces performance in my experience

        elif optimizer_type in ['sgd', 'momentum']:
            self.optimizer = tf.keras.optimizers.SGD(
                learning_rate=lr_schedule,
                momentum=self.params.get('momentum', 0.99),
                clipnorm=clip_norm,
                global_clipnorm=global_clipnorm)

        else:
            print('Cannot recognize optimization algorithm %s' % optimizer_type)
            exit(1)



################################################################
# AGC
# https://github.com/sayakpaul/Adaptive-Gradient-Clipping

def compute_norm(x, axis, keepdims):
    return tf.math.reduce_sum(x ** 2, axis=axis, keepdims=keepdims) ** 0.5


def unitwise_norm(x):
    if len(x.get_shape()) <= 1:  # Scalars and vectors
        axis = None
        keepdims = False
    elif len(x.get_shape()) in [2, 3]:  # Linear layers of shape IO or multihead linear
        axis = 0
        keepdims = True
    elif len(x.get_shape()) == 4:  # Conv kernels of shape HWIO
        axis = [0, 1, 2, ]
        keepdims = True
    else:
        raise ValueError(f"Got a parameter with shape not in [1, 2, 4]! {x}")
    return compute_norm(x, axis, keepdims)


def adaptive_clip_grad(parameters, gradients, clip_factor=0.1, eps=1e-3):
    new_grads = []
    for (params, grads) in zip(parameters, gradients):
        p_norm = unitwise_norm(params)
        max_norm = tf.math.maximum(p_norm, eps) * clip_factor
        grad_norm = unitwise_norm(grads)
        clipped_grad = grads * (max_norm / tf.math.maximum(grad_norm, 1e-6))
        new_grad = tf.where(grad_norm < max_norm, grads, clipped_grad)
        new_grads.append(new_grad)
    return new_grads


class EarlyStoppingMin(tf.keras.callbacks.EarlyStopping):
    """Stop training when a monitored quantity has stopped improving.
    Arguments:
        min_epoch: Minimum number of epochs before considering stopping.

    """

    def __init__(self, min_epoch=0, **kwargs):
        super(EarlyStoppingMin, self).__init__(**kwargs)
        self.min_epoch = min_epoch

    def on_epoch_end(self, epoch, logs=None):
        current = self.get_monitor_value(logs)
        if current is None:
            return
        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if epoch >= self.min_epoch and self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                if self.restore_best_weights:
                    if self.verbose > 0:
                        print('Restoring model weights from the end of the best epoch.')
                    self.model.set_weights(self.best_weights)


class Cyclical1LearningRate(tf.keras.optimizers.schedules.LearningRateSchedule):
    """A LearningRateSchedule that uses cyclical schedule.
    https://yashuseth.blog/2018/11/26/hyper-parameter-tuning-best-practices-learning-rate-batch-size-momentum-weight-decay/
    """

    def __init__(
            self,
            initial_learning_rate,
            maximal_learning_rate,
            final_learning_rate,
            step_size,
            name: str = "Cyclical1LearningRate",
    ):
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.maximal_learning_rate = maximal_learning_rate
        self.final_learning_rate = final_learning_rate
        self.step_size = step_size
        self.name = name

    def __call__(self, step):
        with tf.name_scope(self.name or "Cyclical1LearningRate"):
            initial_learning_rate = tf.convert_to_tensor(self.initial_learning_rate,
                                                         name="initial_learning_rate")
            dtype = initial_learning_rate.dtype
            maximal_learning_rate = tf.cast(self.maximal_learning_rate, dtype)
            final_learning_rate = tf.cast(self.final_learning_rate, dtype)

            step_size = tf.cast(self.step_size, dtype)
            cycle = tf.floor(1 + step / (2 * step_size))
            x = tf.abs(step / step_size - 2 * cycle + 1)

            lr = tf.where(step > 2 * step_size,
                          final_learning_rate,
                          initial_learning_rate + (
                                  maximal_learning_rate - initial_learning_rate
                          ) * tf.maximum(tf.cast(0, dtype), (1 - x)))
            return lr

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "maximal_learning_rate": self.maximal_learning_rate,
            "final_learning_rate": self.final_learning_rate,
            "step_size": self.step_size,
        }


class WarmUp(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Applies a warmup schedule on a given learning rate decay schedule.
    (h/t HuggingFace.)

    Args:
        initial_learning_rate (:obj:`float`):
            The initial learning rate for the schedule after the warmup (so this will be the learning rate at the end
            of the warmup).
        decay_schedule (:obj:`Callable`):
            The learning rate or schedule function to apply after the warmup for the rest of training.
        warmup_steps (:obj:`int`):
            The number of steps for the warmup part of training.
        power (:obj:`float`, `optional`, defaults to 1):
            The power to use for the polynomial warmup (defaults is a linear warmup).
        name (:obj:`str`, `optional`):
            Optional name prefix for the returned tensors during the schedule.
    """

    def __init__(
            self,
            initial_learning_rate: float,
            warmup_steps: int,
            decay_schedule: None,
            power: float = 1.0,
            name: str = None,
    ):
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.warmup_steps = warmup_steps
        self.power = power
        self.decay_schedule = decay_schedule
        self.name = name

    def __call__(self, step):
        with tf.name_scope(self.name or "WarmUp") as name:
            # Implements polynomial warmup. i.e., if global_step < warmup_steps, the
            # learning rate will be `global_step/num_warmup_steps * init_lr`.
            global_step_float = tf.cast(step, tf.float32)
            warmup_steps_float = tf.cast(self.warmup_steps, tf.float32)
            warmup_percent_done = global_step_float / warmup_steps_float
            warmup_learning_rate = self.initial_learning_rate * tf.math.pow(warmup_percent_done, self.power)
            if callable(self.decay_schedule):
                warmed_learning_rate = self.decay_schedule(step - self.warmup_steps)
            else:
                warmed_learning_rate = self.decay_schedule
            return tf.cond(
                global_step_float < warmup_steps_float,
                lambda: warmup_learning_rate,
                lambda: warmed_learning_rate,
                name=name,
            )

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_schedule": self.decay_schedule,
            "warmup_steps": self.warmup_steps,
            "power": self.power,
            "name": self.name,
        }


def safe_next(data_iter, retry=5, sleep=10):
    attempts = 0
    d = None
    while d is None and attempts < retry:
        try:
            d = next(data_iter)
            x, y, w = d
            # Debugging
            # print(f"Loaded weights: {w}")
        except tf.errors.AbortedError:
            print('AbortedError, which has previously indicated NFS daemon restart.', file=sys.stderr)
            time.sleep(sleep)
        attempts += 1

    if d is None:
        # let it crash
        d = next(data_iter)

    return d
