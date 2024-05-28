#!/usr/bin/env python
# -*- coding:utf-8 -*-

import time
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.linear_model import RANSACRegressor
from sklearn.base import BaseEstimator, RegressorMixin

import calib_rt
from ALR import Automated_Loess_Regression

import logging
logging.basicConfig(level=logging.INFO)
file_handler = logging.FileHandler(filename='log.txt', mode='w')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(message)s'))

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(message)s'))

logger = logging.getLogger()
for handler in logger.handlers[:]:
    logger.removeHandler(handler)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

def cal_mrd(y, y_pred):
    return np.nanmean(np.abs(y - y_pred) / y)

def add_noise_to_duplicates(arr, noise_scale=0.01):
    unique, counts = np.unique(arr, return_counts=True)
    duplicates = unique[counts > 1]

    for value in duplicates:
        indices = np.where(arr == value)[0]
        for index in indices:
            arr[index] += np.random.uniform(-noise_scale, noise_scale)

    return arr

class ALR(BaseEstimator, RegressorMixin):

    def __init__(self, param):
        self.param = param

    def fit(self, x_input, y_input):

        if x_input.ndim > 1:
            x = x_input[:, 0]
            y = y_input
        else:
            x, y = x_input, y_input
        x = np.concatenate([x, [self.param[0], self.param[2]]])
        y = np.concatenate([y, [self.param[1], self.param[3]]])

        # erry_y should > 0
        self.model = Automated_Loess_Regression(x, y, err_y=0.1, outliers_det=False)

    def predict(self, xnew):
        if xnew.ndim > 1:
            x = xnew[:, 0]
        else:
            x = xnew
        x[x > 0] -= 0.01
        x[x < 0] += 0.01
        ynew = self.model.interp(x)

        return ynew[0]

def fit_by_loess(x, y, xtarget, ytarget):
    t0 = time.time()
    kf = KFold(n_splits=5, shuffle=True, random_state=123)
    frac_v = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    best_score = 1e6
    best_frac = None
    for frac in frac_v:
        scores = []
        for train_index, test_index in kf.split(x):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            if len(x_train) * frac < 1:
                break
            y_pred = sm.nonparametric.lowess(y_train, x_train, frac,
                                             return_sorted=False,
                                             xvals=x_test)
            score = cal_mrd(y_test, y_pred)
            scores.append(score)

        avg_score = np.nanmean(scores)
        if avg_score < best_score:
            best_score = avg_score
            best_frac = frac

    # pred for xtarget
    lowess = sm.nonparametric.lowess
    y_pred = lowess(y, x, best_frac, xvals=xtarget)
    mape = cal_mrd(ytarget, y_pred)

    # pred for plot
    xplot = np.linspace(x.min(), x.max(), 100)
    yplot = lowess(y, x, best_frac, xvals=xplot)

    t = time.time() - t0

    return xplot, yplot, mape, t

def fit_by_alr(x, y, xtarget, ytarget):
    t0 = time.time()

    # erry_y shold big than 0
    ALR = Automated_Loess_Regression(x, y, err_y=0.1, outliers_det=True)

    # pred for xtarget
    y_pred = ALR.interp(xtarget)[0]
    mrd = cal_mrd(ytarget, y_pred)

    # pred for plot
    xplot = np.linspace(x.min(), x.max(), 100)
    yplot = ALR.interp(xplot)[0]

    t = time.time() - t0

    return xplot, yplot, mrd, t

def fit_by_ransac_alr(x, y, x_target, y_target):
    t0 = time.time()

    idx = np.argmin(x_target)
    p1x, p1y = x_target[idx], y_target[idx]
    idx = np.argmax(x_target)
    p2x, p2y = x_target[idx], y_target[idx]
    ransac_loess = RANSACRegressor(estimator=ALR((p1x, p1y, p2x, p2y)),
                                   min_samples=int(len(x)*0.2))
    ransac_loess.fit(x.reshape(-1, 1), y)

    # pred for x_target
    y_pred = ransac_loess.predict(x_target.reshape(-1, 1))
    mrd = cal_mrd(y_target, y_pred)

    # pred for xplot
    xplot = np.linspace(x.min(), x.max(), 100)
    yplot = ransac_loess.predict(xplot.reshape(-1, 1))

    t = time.time() - t0

    return xplot, yplot, mrd, t

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

def  plot_one_big(data, fname):
    sr_v = [0.5, 0.3, 0.1, 0.01]
    fdr_v = [0.01, 0.05, 0.35, 0.75]

    x_raw = data["Spectral library RT"].values
    y_raw = data["Measured RT"].values

    plt.rc('font', family='Arial', size=7)
    fig, axes = plt.subplots(len(sr_v), len(fdr_v),
                             figsize=(18 / 2.54, 15 / 2.54))

    for i, sr in enumerate(sr_v):
        for j, fdr in enumerate(fdr_v):
            # 1-sampling + decoy + 2-sample
            idx = y_raw > 0
            x_raw, y_raw = x_raw[idx], y_raw[idx]
            x_target, y_target = sample_data(x_raw, y_raw, sr=sr)
            x_mixed, y_mixed = add_noise(x_target, y_target, fdr, method='inner')
            x_100, y_100 = sample_data(x_mixed, y_mixed, target_num=100)
            logger.info(f'{fname}, data in sr: {sr:.2f}, fdr: {fdr:.2f}, N: {len(x_mixed)}')

            # 1. fit by loess with cross-valid
            x1, y1, mrd1, t1 = fit_by_loess(x_100, y_100, x_target, y_target)
            logger.info(f'{fname}, fit-loess,  mrd: {mrd1:.2f}, time: {t1:.2f}')

            # 2. fit by alr
            x2, y2, mrd2, t2 = fit_by_alr(x_100, y_100, x_target, y_target)
            logger.info(f'{fname}, fit-alr,    mrd: {mrd2:.2f}, time: {t2:.2f}')

            # 3. fit by ransac
            x3, y3, mrd3, t3 = fit_by_ransac_alr(x_100, y_100, x_target, y_target)
            logger.info(f'{fname}, fit-ransac, mrd: {mrd3:.2f}, time: {t3:.2f}')

            # 4. fit by calib
            x4, y4, mrd4, t4 = fit_by_calib(x_mixed, y_mixed, x_target, y_target, 99)
            logger.info(f'{fname}, fit-calib w/o  polish,  mrd: {mrd4:.2f}, time: {t4:.2f}')

            # plot
            axes[i, j].get_xaxis().set_visible(False)
            axes[i, j].get_yaxis().set_visible(False)

            axes[i, j].plot(x_mixed, y_mixed, 'o', markersize=0.5, color='grey', label='Data points')
            if 'x1' in locals():
                axes[i, j].plot(x1, y1, lw=2., label='Raw-LOESS')
            if 'x2' in locals():
                axes[i, j].plot(x2, y2, lw=2., label='Quantile-LOESS')
            if 'x3' in locals():
                axes[i, j].plot(x3, y3, lw=2., label='RANSAC-LOESS')
            if 'x4' in locals():
                axes[i, j].plot(x4, y4, lw=2., label='Calib-RT')

            # if i == len(sr_v) - 1 and j == 0:
            #     axes[i, j].legend()

            if j == 0:
                axes[i, j].get_yaxis().set_visible(True)
                ylim = axes[i, j].get_ylim()
            else:
                axes[i, j].set_ylim(ylim)
            if i == len(sr_v) - 1:
                axes[i, j].get_xaxis().set_visible(True)

    plt.tight_layout()
    plt.savefig(fname + '.jpg', dpi=350, bbox_inches='tight')
    plt.show()

def main():
    types = ['exp', 'distort_left', 'distort_right', 'linear', 'S']
    groups = [0, 1]

    for st in types:
        for g in groups:
            data = calib_rt.RTdatasets.get_pandas(sample_type=st,
                                                  index_in_group=g)
            plot_one_big(data, st+'_'+str(g))


if __name__ == '__main__':
    main()

