# source from: https://github.com/olrodrig/ALR

import os
import numpy as np
import rpy2
from rpy2.robjects.packages import STAP
from rpy2.robjects.vectors import FloatVector

class Automated_Loess_Regression:
  def __init__(self, x, y, err_y=0, deg=2, alpha=0, outliers_det=False, n_sims=1000, average=True, verbose=False):
    """
    Automated loess regression (ALR).
    
    Execute the R code ALR.R, which is an automated routine to perform loess regressions to data with one predictor variable. 
    This routine takes into account the observed and intrinsic errors, along with the presence of possible outliers.
    """
    
    deg, outliers_det = int(deg), bool(outliers_det)
    
    if hasattr(alpha, '__len__') == False:  alpha = float(alpha) 
    
    #Convert Python vectors to R vectors
    if hasattr(alpha, '__len__') == True:  alpha = FloatVector(alpha) 
    rx, ry, rerr_y = FloatVector(x), FloatVector(y), 0
    if hasattr(err_y, '__len__') == True:
        rerr_y = FloatVector(err_y)
    else:
        if err_y != 0:
            err_y  = np.linspace(err_y, err_y, len(y))
            rerr_y = FloatVector(err_y)
    
    #Run Automated_Loess_Regression in ALR.R
    path = os.path.dirname(__file__)
    with open(path+'/ALR.R', 'r') as f:
        string = f.read()
    try:    
        ALR=STAP(string,"Automated_Loess_Regression").Automated_Loess_Regression(rx, ry, err_y=rerr_y, deg=deg, alpha=alpha, outliers_det=outliers_det, average=average, n_sims=n_sims, verbose=verbose)
    
        ###ALR ATTRIBUTES###
            
        ###INPUT DATA
        self.deg           = deg
        self.outliers_det  = outliers_det
        self.n_data        = len(x)
        self.x             = x
        self.y             = y
        self.err_y         = 0
        self.with_y_errors = False
        if hasattr(err_y, '__len__') == True:  
            self.err_y = err_y
            self.with_y_errors = True
        
        ###OUTPUT DATA
        self.x_ALR      = np.array(ALR.rx2('x_ALR'))
        self.y_ALR      = np.array(ALR.rx2('y_ALR'))
        self.err_y_ALR  = np.array(ALR.rx2('err_y_ALR'))
        self.y_ALR_sims = np.array(ALR.rx2('y_ALR_sims'))
        self.alpha      = ALR.rx2('alpha')[0]
        self.enp        = ALR.rx2('enp')[0]
        self.ssd        = ALR.rx2('ssd')[0]
        self.err_0      = ALR.rx2('err_0')[0]
        self.n_outliers = int(ALR.rx2('n_outliers')[0])
        self.n_fit      = int(self.n_data - self.n_outliers)
        if self.n_outliers != 0:
            self.x_outliers     = np.array(ALR.rx2('x_outliers'))
            self.y_outliers     = np.array(ALR.rx2('y_outliers'))
            self.err_y_outliers = np.array(ALR.rx2('err_y_outliers'))
    except rpy2.rinterface_lib.embedded.RRuntimeError:
        print('Execution halted')
  
  #interpolate the ALR fit to the x values
  def interp(self, x):
        
      out_of_range = False
      if hasattr(x, '__len__') == True:
          if min(x) < min(self.x_ALR) or max(x) > max(self.x_ALR):  out_of_range = True
      else:
          if x < min(self.x_ALR) or x > max(self.x_ALR): out_of_range = True
      
      if out_of_range:
          print('ERROR (in interp): x has values out of the valid range (i.e., ['+str(min(self.x_ALR))+', '+str(max(self.x_ALR))+']).')
      else:    
          y_ALR     = np.interp(x, self.x_ALR, self.y_ALR)
          err_y_ALR = np.interp(x, self.x_ALR, self.err_y_ALR)
          return y_ALR, err_y_ALR
  
  #plot the ALR regression
  def plot(self, ALRs=[], comparison_func=[], invert_y_axis=False, object_name='input data', xlabel='x', ylabel='y', figure_name=''):
      if len(ALRs) == 0:  ALRs = [self]
      plot_ALR(ALRs, comparison_func=comparison_func, invert_y_axis=invert_y_axis, object_name=object_name, xlabel=xlabel, ylabel=ylabel, figure_name=figure_name)
