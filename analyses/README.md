# Calib-RT analyses

Before the analyses, ensure the necessary dependencies are installed using the following commands:

```bash
pip install matplotlib sklearn rpy2
```

## Scripts

`plot_competitors.py` compares the result of calib-rt and other regression method (Raw-LOESS, Quantile-LOESS, and RANSAC-LOESS) on sample datasets.


`plot_span_filtering.py` compares the effect of span filter on the end extension on S type data.


`plot_grid_size.py` tests different grid sizes on datasets with the most and fewest data points.