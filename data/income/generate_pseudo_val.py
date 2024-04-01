import numpy as np
from sklearn_extra.cluster import CLARA
from collections import Counter

np.random.seed(0)

x_val = np.load("x_val.npy")

km = CLARA(n_clusters=2, metric="manhattan", random_state=0).fit(x_val)
labels = km.predict(x_val)
np.save('pseudo_val_y.npy', labels)
print(Counter(labels))