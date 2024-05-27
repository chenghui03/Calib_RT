#!/usr/bin/env python
# -*- coding:utf-8 -*-

import time
import numpy as np
import matplotlib.pyplot as plt

import calib_rt

def cal_mrd(y, y_pred):
    return np.nanmean(np.abs(y - y_pred) / y)

def fit_by_calib(x_mixed, y_mixed, x_target, y_target, bin_size):
    t0 = time.time()

    model = calib_rt.Calib_RT(bins=bin_size, tol_bins=20)
    model.fit(x_mixed, y_mixed, manual_frac=0.1)

    # pred for x_target
    y_pred = model.predict(x_target)
    mrd = cal_mrd(y_target, y_pred)

    # pred for xplot
    xplot = np.linspace(x_mixed.min(), x_mixed.max(), 100)
    yplot = model.predict(xplot.reshape(-1, 1))

    t = time.time() - t0

    return xplot, yplot, mrd, t

def sample_data(x, y, sr=None, target_num=None):

    np.random.seed(123)
    if sr is not None:
        target_num = int(sr * len(x))
    if target_num > len(x):
        target_num = len(x)

    idx = np.random.choice(len(x), target_num, replace=False)
    idx_max = [np.argmax(x)]
    idx_min = [np.argmin(x)]
    idx = np.concatenate([idx, idx_max, idx_min])

    return x[idx], y[idx]

def add_noise(x, y, fdr):
    noise_num = int(np.ceil(len(x) / (1 - fdr) - len(x)))
    np.random.seed(123)
    x_decoy = np.random.uniform(x.min()+0.1, x.max()-0.1, noise_num)
    y_decoy = np.random.uniform(y.min()+0.1, y.max()-0.1, noise_num)
    xx = np.concatenate([x, x_decoy])
    yy = np.concatenate([y, y_decoy])
    return xx, yy

def main():
    # most points
    data = calib_rt.RTdatasets.get_pandas(
        sample_type='distort_right', index_in_group=1
    )
    x_raw = data["Spectral library RT"].values
    y_raw = data["Measured RT"].values
    sr, fdr = 0.5, 0.75
    x_target1, y_target1 = sample_data(x_raw, y_raw, sr=sr)
    x_mixed1, y_mixed1 = add_noise(x_target1, y_target1, fdr)
    print(f'#points: {len(x_mixed1)}')

    # fewest points
    data = calib_rt.RTdatasets.get_pandas(
        sample_type='exp', index_in_group=1
    )
    x_raw = data["Spectral library RT"].values
    y_raw = data["Measured RT"].values
    sr, fdr = 0.01, 0.01
    x_target2, y_target2 = sample_data(x_raw, y_raw, sr=sr)
    x_mixed2, y_mixed2 = add_noise(x_target2, y_target2, fdr)
    print(f'#points: {len(x_mixed2)}')

    # plot
    bin_size_v = [25, 50, 100, 200]

    plt.rc('font', family='Arial', size=7)
    fig, axes = plt.subplots(2, 4, figsize=(18 / 2.54, 10 / 2.54))

    for i, (x_target, y_target, x_mixed, y_mixed) in enumerate([
        (x_target1, y_target1, x_mixed1, y_mixed1),
        (x_target2, y_target2, x_mixed2, y_mixed2)]
    ):
        for j, bin_size in enumerate(bin_size_v):
            x, y, mrd, t = fit_by_calib(x_mixed, y_mixed, x_target, y_target, bin_size)
            print(f'mrd: {mrd:.3f}')

            axes[i, j].get_xaxis().set_visible(False)
            axes[i, j].get_yaxis().set_visible(False)

            if i == 0:
                axes[i, j].plot(x_mixed, y_mixed, 'o', markersize=0.01, color='grey')
            else:
                axes[i, j].plot(x_mixed, y_mixed, 'o', markersize=1., color='grey')
            axes[i, j].plot(x, y, lw=2., color='r')

            if j == 0:
                axes[i, j].get_yaxis().set_visible(True)
                ylim = axes[i, j].get_ylim()
            else:
                axes[i, j].set_ylim(ylim)
            if i == 1:
                axes[i, j].get_xaxis().set_visible(True)

    plt.tight_layout()
    plt.savefig('bin_size.jpg', dpi=350, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    main()

