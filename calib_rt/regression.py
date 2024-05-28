# -*- coding:utf-8 -*-

import statsmodels.api as sm
import numpy as np
from scipy.interpolate import interp1d

class InputDataError(Exception):
    def __init__(self, message) -> None:
        super().__init__(message)


def fit_by_lowess(x, y, manual_frac=-1):
    data=np.column_stack((x,y))
    data = data[data[:, 0] != 0]
    x,y=data[:,0],data[:,1]

    if manual_frac == -1:
        frac=choose_frac(x,y)
        print(f"Choose lowess frac: {frac:.2f}")
    else:
        frac=manual_frac

    # by lowess
    lowess = sm.nonparametric.lowess
    y_lowess = lowess(y, x, frac)
    x_fit, y_fit = zip(*y_lowess)
    x_fit, y_fit = np.array(x_fit), np.array(y_fit)
    return x_fit, y_fit

def choose_frac(x, y):
    frac_v = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    mape_v = []
    for frac in frac_v:
        y_pred = sm.nonparametric.lowess(y, x, frac, return_sorted=False)
        mape = np.nanmean(np.abs(y - y_pred) / y)
        mape_v.append(mape)
    return frac_v[np.argmin(mape_v)]


class Predictor:
    def __init__(self,x_fit,y_fit) -> None:
        
        data=self.__make_x_unique(x_fit,y_fit)
        x_fit,y_fit=data[:,0],data[:,1]
        self.f_in = interp1d(x_fit, y_fit, kind ='linear')

        # extrapolate for start
        idx_start = np.argsort(x_fit)[:5]
        coefficients = np.polyfit(x_fit[idx_start], y_fit[idx_start], 1)
        self.f_start = np.poly1d(coefficients)

        # extrapolate for end
        idx_end = np.argsort(x_fit)[::-1][:5]
        coefficients = np.polyfit(x_fit[idx_end], y_fit[idx_end], 1)
        self.f_end = np.poly1d(coefficients)

    def __make_x_unique(self,x_fit,y_fit):
        data=np.column_stack((x_fit,y_fit))
        unique_data=dict()
        for x,y in data:
            if not unique_data.get(x):
                unique_data[x]=y
        return np.array(list(unique_data.items()))

    def predict(self,x_interp):
        x_max, x_min = self.f_in.x.max(), self.f_in.x.min()
        is_inner = (x_interp <= x_max) & (x_interp >= x_min)
        is_start = x_interp < x_min
        is_end = x_interp > x_max
        x_in = x_interp[is_inner]
        x_start = x_interp[is_start]
        x_end = x_interp[is_end]
        x_in_pred = self.f_in(x_in)
        x_start_pred = self.f_start(x_start)
        x_end_pred = self.f_end(x_end)
        pred = np.empty_like(x_interp)
        pred[is_inner] = x_in_pred
        pred[is_start] = x_start_pred
        pred[is_end] = x_end_pred
        return pred