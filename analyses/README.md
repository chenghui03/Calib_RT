# Calib-RT analyses

Before the analyses, ensure the necessary dependencies are installed using the following commands:

```bash
pip install matplotlib sklearn rpy2
```

## Scripts

`plot_competitors.py` compares the results of calib-rt and other regression method ([Raw-LOESS](https://www.statsmodels.org/stable/generated/statsmodels.nonparametric.smoothers_lowess.lowess.html#statsmodels.nonparametric.smoothers_lowess.lowess), [Quantile-LOESS](https://github.com/olrodrig/ALR), and [RANSAC-LOESS](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RANSACRegressor.html)) on sample datasets.


`plot_span_filtering.py` compares the effect of the span filter on the end extension of S-type data.


`plot_grid_size.py` tests different grid sizes on datasets with the most and the fewest data points. 
