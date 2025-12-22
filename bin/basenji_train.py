#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2017 Calico LLC
#
# Modifications Copyright (c) 2025 Le Zhang
# Modifications:
#   - Auto-detect validation TFRecords (valid-*.tfr); disable evaluation when absent,
#     and set a fixed number of epochs for no-validation training.
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

from __future__ import print_function
from optparse import OptionParser

import json
import os
import shutil

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import mixed_precision

if tf.__version__[0] == '1':
    tf.compat.v1.enable_eager_execution()
import glob
from basenji import dataset
from basenji import seqnn
from basenji import trainer

"""
basenji_train.py

Train Basenji model using given parameters and data.
"""


################################################################################
# main
################################################################################
def main():
    usage = 'usage: %prog [options] <params_file> <data1_dir> ...'
    parser = OptionParser(usage)
    parser.add_option('-k', dest='keras_fit',
                      default=False, action='store_true',
                      help='Train with Keras fit method [Default: %default]')
    parser.add_option('-m', dest='mixed_precision',
                      default=False, action='store_true',
                      help='Train with mixed precision [Default: %default]')
    parser.add_option('-o', dest='out_dir',
                      default='train_out',
                      help='Output directory for test statistics [Default: %default]')
    parser.add_option('--restore', dest='restore',
                      help='Restore model and continue training [Default: %default]')
    parser.add_option('--trunk', dest='trunk',
                      default=False, action='store_true',
                      help='Restore only model trunk [Default: %default]')
    parser.add_option('--tfr_train', dest='tfr_train_pattern',
                      default=None,
                      help='Training TFR pattern string appended to data_dir/tfrecords for subsetting [Default: %default]')
    parser.add_option('--tfr_eval', dest='tfr_eval_pattern',
                      default=None,
                      help='Evaluation TFR pattern string appended to data_dir/tfrecords for subsetting [Default: %default]')

    parser.add_option('--tissues_scale', dest='tissues_scale',
                      default=False, action='store_true')
    parser.add_option('--weight_type', dest='weight_type',
                      default=None)

    (options, args) = parser.parse_args()

    if len(args) < 2:
        parser.error('Must provide parameters and data directory.')
    else:
        params_file = args[0]
        data_dirs = args[1:]

    if options.keras_fit and len(data_dirs) > 1:
        print('Cannot use keras fit method with multi-genome training. ')
        exit(1)

    if not os.path.isdir(options.out_dir):
        os.mkdir(options.out_dir)
    if params_file != '%s/params.json' % options.out_dir:
        shutil.copy(params_file, '%s/params.json' % options.out_dir)

    # read model parameters
    with open(params_file) as params_open:
        params = json.load(params_open)
    params_model = params['model']
    params_train = params['train']


    # ---- 新增：若用户没有给 --tfr_eval，则检查是否存在 valid-*.tfr
    def _has_valid(data_dir: str) -> bool:
        pat = os.path.join(data_dir, 'tfrecords', 'valid-*.tfr')
        return len(glob.glob(pat)) > 0
    user_spec_eval = (options.tfr_eval_pattern is not None)
    do_eval = True
    if not user_spec_eval:
        # 所有数据目录都无 valid-* 则关闭 eval
        do_eval = any(_has_valid(d) for d in data_dirs)
    # 如果用户显式指定了 --tfr_eval，则保持 do_eval=True（按用户要求执行）

    # ---- 新增：无验证则固定训练 x 轮 100
    if not do_eval:
        params_train['epochs'] = 180



    if params_train.get('fine_tuning_method') == "by_targets":
        print("by_targets")
        targets_scale = True
    else:
        targets_scale = False


    # read datasets
    train_data = []
    eval_data = []
    strand_pairs = []

    for data_dir in data_dirs:
        # set strand pairs
        targets_df = pd.read_csv('%s/targets.txt' % data_dir, sep='\t', index_col=0)
        if 'strand_pair' in targets_df.columns:
            strand_pairs.append(np.array(targets_df.strand_pair))

        # load train data
        train_data.append(dataset.SeqDataset(data_dir,
                                             split_label='train',
                                             batch_size=params_train['batch_size'],
                                             shuffle_buffer=params_train.get('shuffle_buffer', 128),
                                             mode='train',
                                             tfr_pattern=options.tfr_train_pattern,
                                             tissues_scale=options.tissues_scale,
                                             targets_scale=targets_scale,
                                             weight_type=options.weight_type)
                          )

        # load eval data
        if do_eval:
            eval_data.append(dataset.SeqDataset(data_dir,
                                                split_label='valid',
                                                batch_size=params_train['batch_size'],
                                                mode='eval',
                                                tfr_pattern=options.tfr_eval_pattern)
                         )

    params_model['strand_pair'] = strand_pairs

    if options.mixed_precision:
        mixed_precision.set_global_policy('mixed_float16')

    if params_train.get('num_gpu', 1) == 1:
        ########################################
        # one GPU

        # initialize model
        seqnn_model = seqnn.SeqNN(params_model)

        # restore
        if options.restore:
            seqnn_model.restore(options.restore, trunk=options.trunk)

        # # 添加上关于组织的索引以便于进行组织间的loss计算
        # tissues = targets_df['identifier'].values
        # unique_tissues, index_array = np.unique(tissues, return_inverse=True)

        # initialize trainer
        seqnn_trainer = trainer.Trainer(params_train, train_data,
                                        eval_data if do_eval else None,
                                        options.out_dir,
                                        no_eval=(not do_eval))

        # compile model
        seqnn_trainer.compile(seqnn_model)

    else:
        ########################################
        # two GPU

        strategy = tf.distribute.MirroredStrategy()

        with strategy.scope():

            if not options.keras_fit:
                # distribute data
                for di in range(len(data_dirs)):
                    train_data[di].distribute(strategy)
                    eval_data[di].distribute(strategy)

            # initialize model
            seqnn_model = seqnn.SeqNN(params_model)

            # restore
            if options.restore:
                seqnn_model.restore(options.restore, options.trunk)

            # initialize trainer
            seqnn_trainer = trainer.Trainer(params_train, train_data, eval_data, options.out_dir,
                                            strategy, params_train['num_gpu'], options.keras_fit)

            # compile model
            seqnn_trainer.compile(seqnn_model)

    # train model
    if options.keras_fit:
        seqnn_trainer.fit_keras(seqnn_model)
    else:
        if len(data_dirs) == 1:
            seqnn_trainer.fit_tape(seqnn_model)
        else:
            seqnn_trainer.fit2(seqnn_model)


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
