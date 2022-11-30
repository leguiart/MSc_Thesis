import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Initializing the random number generator for reproducibility
random_seed = 10
np.random.seed(random_seed)
random.seed(random_seed) 

point_set = []
for _ in range(3):
    points = 10 * np.random.random_sample((5,2)) - 5
    points = np.column_stack((np.arange(len(points)), points))
    df = pd.DataFrame(points, columns = ['id', 'X_0', 'X_1'])
    point_set += [df]

for i, points in enumerate(point_set):
    plt.figure(figsize=(12, 8))

    p1 = sns.scatterplot(x='X_0', y='X_1', 
                    data=points, s=120)

    for line in range(0,points.shape[0]):
        p1.text(points.X_0[line]+0.05, points.X_1[line], 
        points.id[line], horizontalalignment='left', 
        size='medium', color='black', weight='semibold')
    
    print(points)
    plt.title(f'Point set {i + 1}')

    plt.show()