# Calib-RT
<p align="left">
    <img alt="PyPI - Version" src="https://img.shields.io/pypi/v/pycalib_rt?style=flat&label=Calib-RT&color=green">
    <img title="python version" src="https://img.shields.io/badge/python-3.10-blue" alt="">
</p>

## Overview
Calib-RT is an open-source Python software package designed for RT (retention time) calibration. 
This package provides a flexible and robust solution for achieving accurate RT calibration across various data scales while handling a certain level of noise interference. 

The workflow diagram is below, providing an overview of the process. For a comprehensive and in-depth explanation, please refer to the associated paper for detailed insights and analysis.

<div align=center>
<img src="https://raw.githubusercontent.com/chenghui03/Calib_RT/93750fced36ee36786e61817d71d6ec5e2c5fdd3/img/workflow.svg" alt="Your Image" width="500">
</div>

## Installation
To use `calib_rt`, make sure you have the following dependencies installed:
- Python (>= 3.10)
- numpy (>= 1.26.0)
- pandas (>= 2.1.1)
- networkx (>= 3.1)
- statsmodels (>= 0.14.0)
- scipy (>= 1.11.3)

You can install the `calib_rt` package using pip:
```bash
pip install pycalib_rt 
```

## Usage
Here is an example of how to use `calib_rt` for RT calibration:

```python
import calib_rt

# basic information of all built-in datasets 
calib_rt.RTdatasets.get_datasets_list()  
         sample_type  datasets_num
   0   distort_left             2
   1  distort_right             2
   2            exp             2
   3         linear             2
   4              S             2

# use first of "S" type datasets
datasets = calib_rt.RTdatasets.get_pandas(sample_type="S",index_in_group=1)
x = datasets["Spectral library RT"]
y = datasets["Measured RT"]

# fit and predict
model = calib_rt.Calib_RT() 
model.fit(x,y)
y_pred = model.predict(x)         
```

## Performance test

<div align=center>
<img src="https://raw.githubusercontent.com/chenghui03/Calib_RT/main/img/performance-test-1.jpg" alt="Your Image" width="500">
</div>

Code for performance test is available in the [analysis](https://github.com/chenghui03/Calib_RT/tree/main/analyses).
For a detailed analysis of the test conclusion, please refer to our paper.

## References

[link of paper]()

## License
This project is licensed under the MIT License. See the LICENSE file for details.
