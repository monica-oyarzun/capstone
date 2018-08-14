from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import TruncatedSVD

from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples

from pre_processing import *

# public_df and subset (with columns to use for model) are imported from pre_processing

# dummify categoricals
for col in cols_to_dummify:
    dummify_clustering(subset, col)

# set scaler using MinMax and transform subset
scaler = MinMaxScaler(feature_range = (0,1))
scaler.fit(subset)
scaled_subset = scaler.transform(subset)

# perform PCA to bring components down to 8
svd = TruncatedSVD(n_components=8, n_iter=7)
X_svd = svd.fit_transform(scaled_subset)

# kMeans model after PCA
n_clusters = 9
kmeans = KMeans(n_clusters)
predictions = kmeans.fit_predict(X_svd)
silhouette_score = silhouette_score(X_svd, predictions, metric='euclidean')
print ('predictions: ', np.unique(predictions, return_counts=True))
print ('silhouette score: ', silhouette_score)

# find top features from clustering
centroids = svd.inverse_transform(kmeans.cluster_centers_)
top_centroids = np.argsort(centroids)[:, -1:-11:-1]
for i in range(n_clusters):
    print ('Cluster', i+1, 'total: ', np.unique(predictions == i, return_counts=True)[1][1])
    print (subset[predictions == i].iloc[:, top_centroids[i]].columns)
    print ()

