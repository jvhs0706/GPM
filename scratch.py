import pandas as pd
import numpy as np

df = pd.read_csv('logs/hist-da.csv')

theoretical = np.sqrt(df['num_bins'].values) * df['sigma'].values
actual = df['l2'].values

theoretical = theoretical.reshape(3, 4, 5).mean(axis = -1)
actual = actual.reshape(3, 4, 5).mean(axis = -1)
print(theoretical)
print(actual)