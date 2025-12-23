# Basenji (Rice Variant Effect Prediction)

This repository is a customized version of **Basenji** for **cross-cultivar training in rice** and **variant effect prediction (SAD)**.

---

## Installation

```bash
conda env create -f prespecified.yml
conda activate basenji
conda install scikit-learn-intelex
python setup.py develop --no-deps
```

---

## Prepare Input Data

Download MH63 reference genome:

```bash
cd input
wget http://rice.hzau.edu.cn/rice_rs3/download_ext/MH63RS3.fasta.gz
gunzip MH63RS3.fasta.gz
cd ..
```

Expected file:

```
input/MH63RS3.fasta
```

---

## Variant Effect Prediction (SAD)

Run SAD prediction on VCF variants:

```bash
CUDA_VISIBLE_DEVICES=3 \
bin/basenji_sad.py \
  -f input/MH63RS3.fasta \
  -o output \
  --rc \
  -t model/targets_mh.txt \
  model/params_rep123.json \
  model/32k_w128_cpm_mu_rep123_1/model0_best.h5 \
  input/test_mh.vcf
```

---

## Output

Results are written to `output/` as an HDF5 file containing per-variant regulatory effect scores.

---

## Project Structure

```bash
.
├── bin/                  # Executable scripts (bam_cov, train)
├── basenji/              # Core model, trainer, loss functions
├── model/                # Trained models and parameter files
├── input/                # Genome FASTA, VCFs
├── output/               # SAD results (HDF5)
├── prespecified.yml      # Conda environment
├── setup.py
└── README.md
```
---

## Notes

- Model trained on **three rice cultivars**
- Input length: **32 kb**
- Output resolution: **128 bp**
- GPU recommended

---

## Maintainer

Le Zhang
