# -*- coding:utf-8 -*-

import statsmodels.api as sm
import numpy as np
from scipy.interpolate import interp1d


class InputDataError(Exception):
    def __init__(self, message) -> None:
        super().__init__(message)

def fit_by_lowess(x, y,manual_frac=-1):
    data=np.column_stack((x,y))
    data = data[data[:, 0] != 0]
    x,y=data[:,0],data[:,1]

    if manual_frac==-1:
        frac=choose_frac(x,y)
        print(f"Choose frac:{frac:.2f} automatically")
    else:
        frac=manual_frac

    # 拟合 by lowess
    lowess = sm.nonparametric.lowess
    y_lowess = lowess(y, x, frac)
    x_fit, y_fit = zip(*y_lowess)
    x_fit, y_fit = np.array(x_fit), np.array(y_fit)
    return x_fit, y_fit


def choose_frac(x,y,delta_k_threshold=2):
    
    for frac in np.linspace(0,1,20)[2:]:
        lowess = sm.nonparametric.lowess(y, x, frac)  
        x_smoothed,y_smoothed =map(np.array,zip(*lowess)) 
        
        
        delta_x=x_smoothed[1:]-x_smoothed[:-1]
        delta_x = np.where(delta_x == 0, np.inf,delta_x)

        delta_y=y_smoothed[1:]-y_smoothed[:-1]
        
        k_list=delta_y/delta_x
        k_list=k_list[~np.isnan(k_list)]

        k_delta=k_list[1:]-k_list[:-1]
        if abs(np.nanmax(k_delta)) < delta_k_threshold :
            return frac
    else:
        raise InputDataError("Please check if your filtered data is approiate")

class Predictor:
    def __init__(self,x_fit,y_fit) -> None:
        
        data=self.__make_x_unique(x_fit,y_fit)
        x_fit,y_fit=data[:,0],data[:,1]
        self.f = interp1d(x_fit, y_fit, kind ='linear', fill_value='extrapolate')

    def __make_x_unique(self,x_fit,y_fit):
        data=np.column_stack((x_fit,y_fit))
        unique_data=dict()
        for x,y in data:
            if not unique_data.get(x):
                unique_data[x]=y
        return np.array(list(unique_data.items()))

    def predict(self,x_interp):
         return self.f(x_interp)