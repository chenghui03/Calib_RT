# Calib-RT analyses

脚本使用 `matplotlib` 实现可视化，在运行脚本前请确保包含。你可以通过如下命令安装。

The script uses `matplotlib` for visualisation, make sure it is included before running the script. You can install it with the following command.

```bash
pip install matplotlib
```

## Evaluating performance

`plot_competitors.py` 实现将 calib-rt 和 Raw-LOESS、Quantile-LOESS、RANSAC-LOESS 三者在我们样例数据集上的回归分析作图。其中 Quantile-LOESS 的实现借助于 ALR，which code is clone from https://github.com/olrodrig/ALR 。由于 ALR 通过 R 语言实现，依赖于 `rpy2` package。而 RANSAC-LOESS 通过 `sklearn` 实现。在运行脚本前请确保你的环境中包含 `rpy2` 和 `sklearn`，你可以通过如下命令安装。

`plot_competitors.py` implements the plotting of calib-rt and the regression analyses of Raw-LOESS, Quantile-LOESS, and RANSAC-LOESS on our sample dataset. Quantile-LOESS is implemented with the help of ALR, whose code is clone from https://github.com/olrodrig/ALR . Since ALR is implemented in R, it depends on the `rpy2` package, while RANSAC-LOESS is implemented in `sklearn`. Make sure you have `rpy2` and `sklearn` in your environment before running the script, you can install them with the following command.

```bash
pip install sklearn
pip install rpy2
```

## Span filter affect performance

`plot_span_filtering.py` 在 S 型数据上比较了有无 span filter 对 Calib-RT 在末端延伸上的影响。

## Different grid sizes performance

`plot_grid_size.py` using different grid sizes on datasets with the most data 2 points (244,724, the first row subplots) and the fewest data points (21, the second row subplots). 