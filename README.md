# empyred

A simple pure implementation of Empirical Dynamic Modeling methods in Python. 
Two methods have been implemented: 
 - Simplex projection based on *"Nonlinear forecasting as a way of distinguishing
 chaos from measurement error in time series"* by Sugihara and May, 1990.
 - SMAP based on *"Nonlinear forecasting for the classification of natural time
 series"* by Sugihara, 1994.

> :warning: This package is not for production/research. It was created for educational purposes. If you'd like to use an EMD tool for research or production, you should use the [pyEDM](https://github.com/SugiharaLab/pyEDM).

In the example file accompaning the source code (empyred.py) four different
examples of timeseries can be found. 
- A sinusoidal signal
- Tent Map (default)
- Lorenz system (chaotic regime)
- 3 Species system

## Requirements
- Numpy
- Scipy
- Sklearn
- Matplotlib
