import numpy as np
import pandas as pd

from sklearn.cluster import KMeans

data = pd.read_csv("./covtype-dev.data")

unlabeled = data.drop(columns=["Cover_Type"],axis=1)
print(unlabeled.columns)

kmeans = KMeans(7, random_state=42)

kmeans.fit(unlabeled)

print(kmeans.cluster_centers_)
print(kmeans.labels_)

# print(data)