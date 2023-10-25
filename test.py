# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import calib_rt


def main():

    # get datasets from calib_rt.RTdatasets
    print(calib_rt.RTdatasets.get_datasets_list(),end="\n\n")   # all datasets
    datasets = calib_rt.RTdatasets.get_pandas("S",1)            # get the datasets we want
    print(f"choose the 0 ones in linear :{datasets.Name}")
    print(datasets,end="\n\n")
    x = datasets["Spectral library RT"]
    y = datasets["Measured RT"]

    # use Calib_RT model
    model = calib_rt.Calib_RT(100,5)            # build and set params for Calib_RT model
    model.fit(x,y)                              # fit Calib_RT model
    z = np.linspace(np.min(x),np.max(x),1000) 
    y_pred = model.predict(z)                   # predict the Measured RT we want

    # draw figure and show/save
    plt.figure(figsize=(16,9))
    plt.plot(x, y, 'o', color='#808080', label='Raw', markersize=0.2)
    plt.plot(z,y_pred,color = 'red')
    plt.title(datasets.Name)
    plt.legend()
    plt.minorticks_on()
    plt.grid(True)
    plt.show()
    # plt.savefig(f"{datasets.Name}.png",dpi=300)
    plt.close()



if __name__ == "__main__":
    main()