# Evaluate CAViaR by Quantile Regression
This is a group project of SDSC6013 Topics in Financial Engineering and Technology. We built a value-at-risk model directly modeling the quantile return directly by referring the paper CAViaR: Conditional Autoregressive Value at Risk by Regression Quantiles by Engle and Manganelli (2004).

## Disclaimer
This package `caviar` is currently under development and may contain bugs or incomplete features. Please use with caution and do not use in production environments.

## Known Issues
- [Adaptive model is instable.](https://github.com/yatshunlee/CAViaR-Project/issues/5)

You are welcome to report bug in https://github.com/yatshunlee/CAViaR-Project/issues. :)

## Quick Summary
We constructed two libraries: `caviar` and `var_tests` to model the value at risk and backtest the VaR estimate. For presentation, we constructed a dashboard to showcase how it can be possibly applied. If you are looking for some inspiration, we strongly suggest to take a look of the documentation below and the code in `notebook-example`.

Libraries:
1. CAViaR Model `caviar`
2. VaR Test `var_tests`

Demo Application:
- [Dashboard](https://youtu.be/W-qCPa_voCM)

Documentation:
- [Click here](./doc/README.md)

Contributor(s):
1. Angus Au-Yeung
2. Jasper Lee
