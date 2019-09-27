from sklearn.metrics import mutual_info_score, normalized_mutual_info_score, adjusted_mutual_info_score
# from sklearn.feature_selection import mutual_info_regression
from scipy.stats import chi2_contingency
import numpy as np
import xarray as xr
import time
from itertools import combinations
import pandas as pd

def main():
    mean = 'lev'
    sel = 'time'

    for f in ['monthly.nc', 'daily.nc', '10h.nc']:
        combs = ["{}x{}".format(x,y) for (x,y) in combinations(['tm1', 'vm1', 'um1', 'qm1'], 2)]
        ds = xr.open_dataset(f)
        df = pd.DataFrame(index=range(0, getattr(ds, sel).size), columns = combs, dtype=float)
        for ix in range(0,  getattr(ds, sel).size):
            for (v1, v2) in combinations(['tm1', 'vm1', 'um1', 'qm1'], 2):
                d1 = getattr(ds, v1).mean(mean).isel(**{sel:ix}).data.flatten()
                d2 = getattr(ds, v2).mean(mean).isel(**{sel:ix}).data.flatten()

                d1 = preprocess(d1)
                d2 = preprocess(d2)
                # start = time.time()
                mi = mutual_information(d1, d2, 10, 'sklearn')
                # t = (time.time() - start) / 1000.0
                df['{}x{}'.format(v1,v2)][ix] = mi
                print("MI: {:.3f} ({} & {}) {}".format(mi, v1, v2, ix))
        df.to_csv('{}.{}.{}.nmi.csv'.format(f, mean, sel))


def calc_MI_sklearn(x, y, bins):
    c_xy = np.histogram2d(x, y, bins)[0]
    mi = normalized_mutual_info_score(x, y, average_method='arithmetic') # nats? bits?
    return mi

def preprocess(data, bins=50):
    data = data[~np.isnan(data)]  # kick out nans
    mini, maxi = data.min(), data.max()

    borders = pd.np.linspace(mini, maxi, bins, endpoint=False)
    discrete = np.digitize(data, borders, right=True)
    return discrete

def calc_MI_scipy(x, y, bins):
    c_xy = np.histogram2d(x, y, bins)[0]
    g, p, dof, expected = chi2_contingency(c_xy, lambda_="log-likelihood")
    mi = 0.5 * g / c_xy.sum() # nats? bits?
    return mi


def calc_MI_numpy(X, Y, bins):

    def shannon_entropy(c):
        c_normalized = c / float(np.sum(c))
        c_normalized = c_normalized[np.nonzero(c_normalized)]
        H = -sum(c_normalized* np.log(c_normalized))
        return H

    c_XY = np.histogram2d(X,Y,bins)[0]
    c_X  = np.histogram(X,bins)[0]
    c_Y  = np.histogram(Y,bins)[0]

    H_X  = shannon_entropy(c_X)
    H_Y  = shannon_entropy(c_Y)
    H_XY = shannon_entropy(c_XY)

    MI   = H_X + H_Y - H_XY
    return MI

def mutual_information(x, y, bins=10, method='sklearn'):
    if method.strip() in ('sklearn',):
        func = calc_MI_sklearn
    elif method.strip() in ('numpy', 'np'):
        func = calc_MI_numpy
    elif method.strip() in ('scipy'):
        func = calc_MI_scipy
    # elif method in ('regression', 'regr'):
    #     func = calc_MI_regression
    #     bins = 3
    else:
        msg = "Method '{}' is invalid.".format(method)
        raise Exception(msg)

    try:
        result = func(x, y, bins)
    except ValueError as err:
        print(str(err))
    else:
        return result


if __name__ == '__main__':
    main()
