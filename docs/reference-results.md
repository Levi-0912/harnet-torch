# Reference results from the official implementation

Numbers produced by running the official TensorFlow code ([mdsunivie/HARNet](https://github.com/mdsunivie/HARNet), commit `3b2832f`). The PyTorch implementation in this repository is validated against them.

## Environment

The official repository is cloned locally into `reference/HARNet` (not tracked here) with its own virtual environment.

| Component | Version / setting |
|---|---|
| Python | 3.12.14 |
| TensorFlow | 2.21.0, CPU only (native Windows GPU support ended with TF 2.11) |
| Keras | `tf_keras` 2.21.0 with `TF_USE_LEGACY_KERAS=1` |
| Other | numpy 2.5.3, pandas 3.0.6, pydantic 2.13.5, scikit-learn 1.9.1, plus `tensorboard` |

Changes to the official code (the minimum needed to run):

1. `src/harnet/util.py`: every `'rv5_ss'` replaced by `'rv5'`, because the published data file has no `rv5_ss` column.
2. `src/harnet/model.py:441`: `np.Inf` replaced by `np.inf` (`np.Inf` was removed in NumPy 2; same meaning).

## Data

- Realized variance is **`rv5`** (5-minute RV, no subsampling). The paper uses `rv5_ss`, so absolute values will not match the paper's tables; comparisons with the paper are qualitative.
- In the published data file, `.SPX` covers 2000-01-03 to 2020-01-31.

## Seeding

The official code does not fix a random seed: training batches are drawn with `np.random.randint` (`model.py:114`). To make the reference numbers reproducible, a thin wrapper seeds NumPy and TensorFlow before calling the official entry point, leaving the official code untouched:

```bash
cd reference/HARNet
TF_USE_LEGACY_KERAS=1 .venv/Scripts/python -c "import sys, numpy as np, tensorflow as tf; np.random.seed(0); tf.random.set_seed(0); sys.argv=['harnet','<config>']; from harnet.main import main; main()"
```

HAR fitted by OLS is deterministic and does not depend on the seed.

## Window 1: `.SPX`, train 2012–2015, test 2016

### HAR20_OLS (`configs/RV/RecField_20/HAR20_OLS.in`)

The scaler is `None`, so the `norm_*` metrics equal the raw ones.

| Split | MAE | MSE | QLIKE |
|---|---|---|---|
| Train | 3.244761e-05 | 1.525291e-08 | 0.2947405 |
| Test | 3.269629e-05 | 4.370383e-09 | 0.3463029 |

OLS coefficients (raw RV scale, from `results/HAR20_OLS/lm.joblib`):

| β0 | β1 (daily) | β2 (weekly, 5 days) | β3 (monthly, 20 days) |
|---|---|---|---|
| 2.48258226e-05 | 0.213556753 | 0.181968421 | 0.137705602 |

Forecasts are clipped from below at 8.1050030e-07 = 0.5 × the minimum RV in the training window (`model.py:44-45`, `model.py:179`).

### HARNet20_QLIKE_OLS (`configs/RV/RecField_20/QLIKE/HARNet20_QLIKE_OLS.in`)

MinMax scaler with fixed constants (0, 1e-3), QLIKE loss, seed 0, 10,000 training iterations (about 4 minutes on CPU). Metrics use the weights after the final iteration (`save_best_weights = false`); the test set is not used for model selection.

| Split | MAE | MSE | QLIKE |
|---|---|---|---|
| Train | 3.160732e-05 | 1.651125e-08 | 0.2512888 |
| Test | 2.881615e-05 | 4.108840e-09 | 0.2611258 |

Change relative to HAR20_OLS:

| Split | MAE | MSE | QLIKE |
|---|---|---|---|
| Train | −2.59% | +8.25% | −14.74% |
| Test | **−11.87%** | −5.98% | **−24.60%** |

Note: the paper's headline ~11.74% MAE reduction is the median over all windows and indices for HARNet trained with the **MAE** loss. The figure above is a single window with the QLIKE loss, so the two are not directly comparable.

**Initialization check.** Before training (step 0), HARNet's metrics match HAR20_OLS to float32 precision, e.g. test MAE 3.2696288e-05 vs 3.2696287e-05. A correct HAR-based initialization must reproduce this.

**Learned weights** (13 parameters, on the scaled axis where 1 = 1e-3). Within each convolution kernel, higher indices are closer to day *t*:

| Layer | Initial | After training |
|---|---|---|
| Weekly conv (5 taps, dilation 1) | 0.2 × 5 | 0.0765, 0.1411, 0.2651, 0.3404, 0.3523 |
| Monthly conv (4 taps, dilation 5) | 0.25 × 4 | 0.1392, 0.1309, 0.2086, 0.3766 |
| Output layer [β0, β1, β2, β3] | 0.0248, 0.2136, 0.1820, 0.1377 | 0.0073, 0.3793, 0.3093, 0.1195 |

Training moves the convolutions away from uniform averages toward weights that favour more recent days; β1 and β2 increase and β0 decreases.

Test QLIKE is still decreasing slowly at iteration 10,000; its minimum, 0.25994, occurs at iteration 9,911.

**Reproducibility.** A second run with seed 0 gives bit-identical values for all 12 metrics in `metrics.csv`, so results are deterministic on the same machine and software stack. On other hardware or TF versions, oneDNN's operation ordering may cause float-level differences.
