# -*- coding:utf-8 -*-

import statsmodels.api as sm
import numpy as np


def generate_noise(path, noise_prop, seed):
    data = np.load(path)
    x, y = data["x"], data["y"]
    datamatrix = np.concatenate((x[:, np.newaxis], y[:, np.newaxis]), axis=1)

    datasize = len(x)
    datasize_noise = datasize * noise_prop

    noise_xrange_left, noise_xrange_right = np.min(x), np.max(x)
    noise_yrange_low, noise_yrange_up = np.min(y), np.max(y)

    np.random.seed(seed)
    noisematrix = np.concatenate(( \
        np.random.uniform(noise_xrange_left, noise_xrange_right,
                          (datasize_noise, 1), ),
        np.random.uniform(noise_yrange_low, noise_yrange_up,
                          (datasize_noise, 1), ),), axis=1)

    datamatrix = np.concatenate((datamatrix, noisematrix), axis=0)

    return datamatrix[:, 0], datamatrix[:, 1]

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