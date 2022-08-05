import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import json
import math
import copy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

def numpy_nan_median(a):
    if np.all(a!=a):
        return np.NaN  
    else:
        return np.nanmedian(a)

statsPath = '~/Downloads/BodyBrainData/BodyBrain_stats.csv'
rawDataPath = '~/Downloads/BodyBrainData/BodyBrain.csv'

individualsPerRun = 250000
totalRuns = 20
pop_size = 50

BodyBrain_stats = pd.read_csv(statsPath)


with pd.read_csv(rawDataPath, chunksize=individualsPerRun) as reader:
    medians_column = []
    for i, chunk in tqdm(enumerate(reader), total=totalRuns, desc="Processing chunk:"):
        chunk.drop(['md5', 'id'], axis=1, inplace = True)
        raw_columns = chunk.columns.tolist()
        count = 0
        # For each generation
        for j in range(0, min(individualsPerRun, len(chunk)), pop_size):
            # Select gen+1 generation data
            gen_data = chunk.iloc[j:j + pop_size,:]
            desired_indicators = [indicator.replace(' ', '_') for indicator in BodyBrain_stats['Indicator'][count*25 : (count + 1)*25]]
            count += 1
            for indicator_key in desired_indicators:
                try:
                    if indicator_key in raw_columns:
                        arr = np.array(gen_data[indicator_key])
                        if arr.dtype == np.dtype('O'):
                            arr = arr.astype(np.float64)
                        medians_column += [numpy_nan_median(arr)]
                    else:
                        medians_column += [0]
                except Exception as e:
                    print(e)
                    medians_column += [np.nan]




BodyBrain_stats['Median'] = medians_column
BodyBrain_stats.to_csv(statsPath, index = False)


