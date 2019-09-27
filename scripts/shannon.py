import xarray as xr
import pandas as pd
import numpy as np
from shannon.discrete import entropy as shannon_entropy
import pyeeg


def main():
    ds = xr.open_dataset('/home/ucyo/Developments/big_files/IMK_MESSy______20141101_0000_6h02_pl.nc')
    variables = ['vm1', 'qm1','tm1', 'um1']
    bins = [2**x*100 for x in range(0, 32, 1)]

    df = pd.DataFrame(index=bins, columns=variables, dtype=float)
    df.index.name = 'Bins'
    df.name = 'Shannon'

    for var in variables:
        for b in bins:
            try:
                se = shannon_entropy(preprocess(getattr(ds, var), b))
                df[var][b] = se
                print("Added: {} {}: {}".format(var, b, se))
            except MemoryError:
                pass
            else:
                df.to_csv('shannon.csv')

# def sample_entropy(x):
#     return pyeeg.samp_entropy(x, 2, np.std((x))*.2)

def preprocess(data, bins):
    data = data.values[~np.isnan(data.values)]  # kick out nans
    mini, maxi = data.min(), data.max()

    borders = pd.np.linspace(mini, maxi, bins, endpoint=False)
    discrete = np.digitize(data, borders, right=True)
    return discrete

# def entropies(data, bins):
#     data = preprocess(data, bins)
#     h_entropy = shannon_entropy(data)
#     sampleen = sample_entropy(discrete)

#     result = namedtuple("Discrete", "borders, shannon, sample")
#     return result(borders, h_entropy, sampleen)

if __name__ == '__main__':
    main()
