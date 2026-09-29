# harnet-torch

A PyTorch reimplementation of **HARNet**, the dilated-convolution model for realized volatility forecasting by Reisenhofer, Bayer & Hautsch (2022), together with the HAR baselines it builds on.

> **Status:** work in progress.

## Goals

1. **Reproduce** the reference results of the official TensorFlow implementation (HAR with OLS / WLS / log-OLS fits, and HARNet trained with MAE and QLIKE losses).
2. **Reimplement** the full pipeline in pandas + PyTorch and match those reference numbers.
3. **Extend** the evaluation out of sample to 2021–2026 data, with Diebold–Mariano tests for significance.

## Data

Daily 5-minute realized variance (`rv5`) from the Oxford-Man Realized Library, as distributed with the official repository. Data files are not included here.

## Reference

R. Reisenhofer, X. Bayer, and N. Hautsch. *HARNet: A Convolutional Neural Network for Realized Volatility Forecasting.* arXiv:2205.07719, 2022. Official implementation: [mdsunivie/HARNet](https://github.com/mdsunivie/HARNet).
