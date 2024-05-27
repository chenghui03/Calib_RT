# -*- coding:utf-8 -*-

import statsmodels.api as sm
import numpy as np

def cal_mrd(y, y_pred):
    return np.nanmean(np.abs(y - y_pred) / y)


def fit_by_raw_lowess(x, y,frac=0.1):
    lowess = sm.nonparametric.lowess
    y_lowess = lowess(y, x, frac)
    x_fit, y_fit = zip(*y_lowess)
    x_fit, y_fit = np.array(x_fit), np.array(y_fit)
    return x_fit, y_fit


class Normalization(object):
    def __init__(self,x,y) -> None:
        self.params = {"x":(np.min(x),np.max(x)-np.min(x)),
                       "y":(np.min(y),np.max(y)-np.min(y))}
        self.x_normal = self.normal("x",x)
        self.y_normal = self.normal("y",y)

    def normal(self,key,x):
        keymin,keylen = self.params[key]
        return (x-keymin)/keylen
    
    def denormal(self,key,x):
        keymin,keylen = self.params[key]
        return keylen*x+keymin
    
    def get_normalized_data(self):
        return self.x_normal,self.y_normal
    
    def denormalize_data(self,x_normal,y_normal):
        x = self.denormal(x_normal)
        y = self.denormal(y_normal)
        return x,y