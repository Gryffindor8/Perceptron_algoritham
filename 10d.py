import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
from pylab import show

# first make some fake data with same layout as yours
data = pd.DataFrame(np.random.randn(1000, 10), columns=['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10'])
scatter_matrix(data, alpha=0.2, figsize=(6, 6), diagonal='hist')

show()
