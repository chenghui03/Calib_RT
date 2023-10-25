# -*- coding:utf-8 -*-

'''
main function of calib_rt

.. calib_rt:
    url

'''

import numpy as np

from .screen import screen_by_hist, screen_by_graph, polish_ends
from .regression import fit_by_lowess, Predictor
from .utils import Normalization

class Calib_RT(object):
    """
    Calib_RT is used for TODO ...

    Attributes:
        fit(): fit by Calib_RT model
        predict(): predict by Calib_RT model
        setdata(): Set data for Calib_RT model
        result(): TODO
    """
    
    def __init__(self,bins:int=100,tol_bins:float=5) -> None:
        """
        Set params for Calib_RT model

        Args:
            bins(int): 
            tol_bins(float):
        
        Examples:
            >>> import calib_rt 
            >>> model = calib_rt.Calib_RT(bin=100,tol_bin=5)
        
        References:
            url
        """
        # TODO 输入检查

        self.bins = bins
        self.tol_bins = tol_bins

    def setdata(self,x,y) -> None:
        """
        Set data for Calib_RT model

        Args:
            x(array_like): 1-D array_like for Spectral library RT
            y(array_like): 1-D array_like for Measured RT
        
        Examples:
            >>> import calib_rt 
            >>> model = calib_rt.Calib_RT(bin=100,tol_bin=5)
            >>> model.setdata(x,y)
        
        References:
            url
        """
        # TODO 输入检查

        self.x = x
        self.y = y
        self.nomral = Normalization(x,y)
        self.x_normal,self.y_normal = self.nomral.get_normalized_data()

    def fit(self,x,y,manual_frac:float=-1) -> None:
        """
        fit Calib_RT model

        Args:
            x(array_like): 1-D array_like for Spectral library RT
            y(array_like): 1-D array_like for Measured RT
            manual_frac(float): 
        
        Examples:
            >>> import calib_rt 
            >>> model = calib_rt.Calib_RT(bin=100,tol_bin=5)
            >>> model.fit(x,y)
        
        References:
            url
        """
        # TODO 输入检查，同时将x,y转化为np.ndarray

        self.setdata(x,y)
        self.__fit(manual_frac)


    def __fit(self,manual_frac):

        x_hist,y_hist,rho_hist = screen_by_hist(self.x_normal,self.y_normal,
                                                self.bins)
        x_graph,y_graph = screen_by_graph(x_hist,y_hist,
                                          rho_hist)
        x_polish,y_polish = polish_ends(x_graph,y_graph,
                                        self.tol_bins)
        self.x_fit,self.y_fit = fit_by_lowess(x_polish,y_polish,manual_frac)
        self.Predor = Predictor(self.x_fit,self.y_fit)


    def result(self) -> str:
        """
        """
        # TODO ：用何指标衡量拟合的好坏，现有合适的 Spectral library RT 范围
        self.result_dict = dict()
        xmin,xmax = np.min(self.x_fit),np.max(self.x_fit)
        self.result_dict["good Spectral library RT range"] = np.array((xmin,xmax))


    def predict(self,x) -> np.ndarray:
        """
        predict Measured RT by Spectral library RT on the basis of Calib_RT model

        Args:
            x(array_like): 1-D array_like for Spectral library RT for predict
        
        Results:
            1-D np.ndarray Measured RT predict
        
        Examples:
            >>> import calib_rt 
            >>> model = calib_rt.Calib_RT(bin=100,tol_bin=5)
            >>> model.fit(x,y)
            >>> model.predict(x)
        
        References:
            url
        """
        # TODO 输入检查
        return self.__predict(x)
        

    def __predict(self,x):
        x_pred_normal = self.nomral.normal("x",x)
        y_pred_normal = self.Predor.predict(x_pred_normal)
        y_pred = self.nomral.denormal("y",y_pred_normal)
        return y_pred