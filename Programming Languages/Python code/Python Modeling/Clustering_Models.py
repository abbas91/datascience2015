# Clustering Model #


# -- Prototype-based clustering

# K-mean
from sklearn.cluster import KMeans
km = KMeans(n_clusters=3, # k
	        init='random', # initate points
	        n_init=10, # runs to choose lowest SSE #
	        max_iter=300, # iters in each run
	        tol=1e-04, # define converge
	        random_state=0)
Y_km = km.fit_predict(X)


# K-mean ++
from sklearn.cluster import KMeans
km = KMeans(n_clusters=3, # k
	        init='k-means++', # initiate points far away from each other
	        n_init=10, # runs to choose lowest SSE #
	        max_iter=300, # iters in each run
	        tol=1e-04, # define converge
	        random_state=0)
Y_km = km.fit_predict(X)



# FCM (Fuzzy C-means) - one memeber can have multiple groups
" ?? Not in sklearn, find outside "













# -- Hierarchical clustering

# Agglomerative methods - start with n groups -> 1 groups

" implement in scipy "
# calculate distences matrix
from scipy.sptial.distance import pdist, squareform
row_dist = pd.DataFrame(squareform(
	       pdist(df, metric='euclidean')),
           columns=labels, index=labels)

"""
    A  B  C  D 
ID1 
ID2
ID3

"""

"""
    ID1  ID2  ID3
ID1 0    1.3  4.5
ID2 1.3  0    7.2
ID3 4.5  7.2  0
"""

from scipy.cluster.hierarchy import linkage
# option 1
row_clusters = linkage(pdist(df, metric='euclidean'), # condensed distence 
	                   method='complete') # ------------- choose an inkage method
# option 2
row_clusters = linkage(df.values,
	                   method='complete', # input sample matrix
	                   metric='euclidean')

pd.DataFrame(row_clusters,
	         columns=['row label 1',
	                  'row label 2',
	                  'distance',
	                  'no. of items in clust.'],
	         index=['cluster %d' %(i+1) for i in
	                range(row_clusters.shape[0])])





" implement in sklearn "

from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters=2,
	                         affinity='euclidean',
	                         linkage='complete')
labels = ac.fit_predict(X)
labels # each observation
" [1 0 0 1 0]"











# Divisive methods - start with 1 group -> n groups

" Not included here. Find outside ??? "















# -- Density-based clustering < not assume spharel shape, good for complex shape >

# DBSCAN
from sklearn.cluster import DBSCAN
db = DBSCAN(eps=0.2,
	        min_samples=5,
	        metric='euclidean')
Y_db = db.fit_predict(X)




















# -- Graph-based clustering

" Not included here. Find outside ??? "

# SpectralClustering
from sklearn.cluster import SpectralClustering












# -- Evaluation metrics

# [Within-cluster SSE(distortion)] for K-mean methods

distortions = []
for i in range(1,11):
	km = KMeans(n_clusters=i, # k
	        init='k-mean++', # initate points
	        n_init=10, # runs to choose lowest SSE #
	        max_iter=300, # iters in each run
	        tol=1e-04, # define converge
	        random_state=0)
	km.fit(X)
	distortions.append(km.inertia_)
plt.plot(range(1,11), distortions, marker='o')
plt.xlabel('Number of cluster')
plt.ylable('Distoration')
plt.show()

" Use elbow method to define K "




# [Silhouette coefficient] for alls

km = KMeans(n_clusters=3, # k
	        init='k-mean++', # initate points
	        n_init=10, # runs to choose lowest SSE #
	        max_iter=300, # iters in each run
	        tol=1e-04, # define converge
	        random_state=0)
Y_km = km.fit_predict(X)

import numpy as np
from matplotlib import cm
from sklearn.metrics import silhouette_samples # or silhouette_scores = numpy.mean(silhouette_samples(...))
cluster_labels = np.unique(Y_km)
n_clusters = cluster_labels.shape[0]
silhouette_vals = silhouette_samples(X,
	                                 Y_km,
	                                 metric='euclidean')
y_ax_lower, y_ax_upper = 0, 0
yticks = []
for i, c in enumerate(cluster_labels):
	c_silhouette_vals = silhouette_vals[Y_km == c]
	c_silhouette_vals.sort()
	y_ax_upper += len(c_silhouette_vals)
	color = cm.jet(i / n_clusters)
	plt.barh(range(y_ax_lower, y_ax_lower),
		     c_silhouette_vals,
		     height=1.0,
		     edgecolor='none',
		     color=color)
	yticks.append((y_ax_lower + y_ax_upper) / 2)
	y_ax_lower += len(c_silhouette_vals)
silhouette_avg = np.mean(silhouette_avg)
plt.axvline(silhouette_avg,
	        color="red",
	        linestyle="--")
plt.yticks(yticks, cluster_labels + 1)
plt.ylabel('Cluster')
plt.xlabel('Silhouette coefficient')
plt.show()





























































