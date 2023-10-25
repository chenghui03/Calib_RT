
# Calib-RT

<!-- TODO: 简述背景 -->

![workflow](https://gitee.com/chenghui03/calib_-rt/raw/release/img/workflow.svg)

# Document

<!-- TODO: 文档 -->

# Installation

To install the most recent release, run

`pip install calib_rt`

# Quick start

1. You can get datasets from calib_rt.RTdatasets, and get_datasets_list() will give you some basis information of all datasets.
   ```python
   >>> calib_rt.RTdatasets.get_datasets_list()  # all datasets
        sample_type  datasets_num
   0   distort_left             2
   1  distort_right             2
   2            exp             2
   3         linear             2
   4              S             2
   ```
2. Then you can choose a dataset by sample_type and it's index in group.
   ```python
   datasets = calib_rt.RTdatasets.get_pandas("S",1) # get the datasets we want
   datasets_info = datasets.Name                    # detail information can get from Name 
   x = datasets["Spectral library RT"]
   y = datasets["Measured RT"]
   ```
3. Finally, you can use Calib_RT just as sklearn model.
   ```python
   model = calib_rt.Calib_RT(100,5) # build and set params for Calib_RT model
   model.fit(x,y)                   # fit Calib_RT model
   y_pred = model.predict(x)        # predict the Measured RT we want
   ```

# Performance test

<!-- TODO: 性能测试 -->

![Stress](https://gitee.com/chenghui03/calib_-rt/raw/release/img/stress1.JPG)

# Cite
 
<!-- TODO: 论文相关 -->

[link of paper]()