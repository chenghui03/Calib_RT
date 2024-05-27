#!/usr/bin/env python
# -*- coding:utf-8 -*-

import time
import numpy as np
import matplotlib.pyplot as plt

import calib_rt

def cal_mrd(y, y_pred):
    return np.nanmean(np.abs(y - y_pred) / y)

def fit_by_calib(x_mixed, y_mixed, x_target, y_target, tol_bins):
    t0 = time.time()

    model = calib_rt.Calib_RT(tol_bins=tol_bins)
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

    np.random.seed(1234)
    if sr is not None:
        target_num = int(sr * len(x))
    if target_num > len(x):
        target_num = len(x)

    idx = np.random.choice(len(x), target_num, replace=False)
    idx_max = [np.argmax(x)]
    idx_min = [np.argmin(x)]
    idx = np.concatenate([idx, idx_max, idx_min])

    return x[idx], y[idx]


def add_noise(x, y, fdr, method):
    noise_num = int(np.ceil(len(x) / (1 - fdr) - len(x)))
    np.random.seed(123)
    x_width = x.max() - x.min()
    y_width = y.max() - y.min()
    if method == 'inner':
        x_decoy = np.random.uniform(x.min()+0.1, x.max()-0.1, noise_num)
        y_decoy = np.random.uniform(y.min()+0.1, y.max()-0.1, noise_num)
    elif method == 'outer':
        x_decoy = np.random.uniform(x.min() - x_width*0.2, x.max() + x_width*0.2, noise_num)
        y_decoy = np.random.uniform(max(y.min() - y_width*0.2, 0), y.max() + y_width*0.2, noise_num)
    xx = np.concatenate([x, x_decoy])
    yy = np.concatenate([y, y_decoy])
    return xx, yy

def plot_one_big(data, fname):
    sr = 0.3
    fdr = 0.35

    x_raw = data["Spectral library RT"].values
    y_raw = data["Measured RT"].values

    plt.rc('font', family='Arial', size=7)
    plt.figure(figsize=(12 / 2.54, 7 / 2.54))

    # 1-sampling + decoy + 2-sample
    idx = y_raw > 0
    x_raw, y_raw = x_raw[idx], y_raw[idx]
    x_target, y_target = sample_data(x_raw, y_raw, sr=sr)
    x_mixed, y_mixed = add_noise(x_target, y_target, fdr, method='outer')

    line1 = x_target.min()
    line2 = x_target.max()
    nl = sum(x_mixed < line1)
    nr = sum(x_mixed > line2)
    nc = len(x_mixed) - nl - nr
    print(nl, nc, nr)

    # fit by calib w/wo polish
    x4, y4, mrd4, t4 = fit_by_calib(x_mixed, y_mixed, x_target, y_target, 10)
    x5, y5, mrd5, t5 = fit_by_calib(x_mixed, y_mixed, x_target, y_target, 99)

    # plot
    plt.plot(x_mixed, y_mixed, 'o', markersize=0.4, color='grey', label='Data points')
    if 'x4' in locals():
        plt.plot(x4, y4, lw=1., label='Calib-RT w/ polishing')
    if 'x5' in locals():
        plt.plot(x5, y5, lw=1., label='Calib-RT wo/ polishing')
    plt.xlim(-150, 280)
    plt.ylim(-50, 450)
    # plt.axvline(line1, color='black', linestyle='--')
    # plt.axvline(line2, linestyle='--')
    # plt.legend()

    plt.tight_layout()
    plt.savefig('polish.jpg', dpi=350, bbox_inches='tight')
    plt.show()


def main():
    types = ['S']
    groups = [0]

    for st in types:
        for g in groups:
            data = calib_rt.RTdatasets.get_pandas(sample_type=st,
                                                  index_in_group=g)
            plot_one_big(data, st+'_'+str(g))


if __name__ == '__main__':
    main()

