# Calib-RT
Calib-RT is an open-source Python software package designed for RT (retention time) calibration. 
This package provides a flexible and robust solution for achieving accurate RT calibration across various data scales while handling a certain level of noise interference. 

The workflow diagram is below, providing an overview of the process. For a comprehensive and in-depth explanation, please refer to the associated paper for detailed insights and analysis.

<div align=center>
<img src="https://raw.githubusercontent.com/chenghui03/Calib_RT/93750fced36ee36786e61817d71d6ec5e2c5fdd3/img/workflow.svg" alt="Your Image" width="500">
</div>

# Installation
You can install the calib_rt package using pip:
```bash
pip install calib_rt 
```

# Usage
Here is an example of how to use Calib-RT for retention time calibration:

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
datasets = calib_rt.RTdatasets.get_pandas(sample_type="S",1)   
x = datasets["Spectral library RT"]
y = datasets["Measured RT"]
# fit and predict
model = calib_rt.Calib_RT() 
model.fit(x,y)                  
y_pred = model.predict(x)        
```

# Performance test

<div align=center>
<img src="https://raw.githubusercontent.com/chenghui03/Calib_RT/main/img/performance.jpg" alt="Your Image" width="500">
</div>

For a detailed analysis of the test conclusion and all performance test results, please refer to the full paper.

# References
[link of paper]()

# License
This project is licensed under the MIT License. See the LICENSE file for details.
