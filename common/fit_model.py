from numpy import arange
from pandas import read_csv
from scipy.optimize import curve_fit
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
import numpy as np


def objective(x, a, b, c):
    return a * x + b * x**2 + c


def fit(time_avg_small_obs, x=[], eva=[], gt=[]):

    popt, error = curve_fit(objective, x, time_avg_small_obs)
    a, b, c = popt
    print('y = %.15f * x + %.15f * x**2 + %.15f' % (a, b, c))

    pyplot.scatter(x, time_avg_small_obs, label='observation')

    pred = [objective(i, a, b, c) for i in eva]

    pyplot.plot(eva, pred, '--', color='red', label='prediction')

    pyplot.scatter(eva, gt, color='blue', label='truth')

    print(mean_squared_error(pred, gt, squared=False))

    #pred = [objective(i, a/1.8, b/1.8, c/1.8) for i in eva]
    #pyplot.plot(eva, pred, '--', color='red', label='prediction')

    # pyplot.legend()

    # pyplot.show()

    return  a, b, c

def fit_return(time_avg_small_obs, x=[]):
    popt, error = curve_fit(objective, x, time_avg_small_obs)
    a, b, c = popt
    return a, b, c
