# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 21:50:56 2021

@author: Felix
"""

from scipy.cluster.hierarchy import dendrogram, linkage 
from matplotlib import pyplot as plt 
import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy.spatial import distance
import seaborn as sns
import matplotlib as mpl
import math

from sklearn.cluster import KMeans

	
import scipy.sparse as sp

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils.extmath import  stable_cumsum






def q1():
    c1 = np.loadtxt(r"C:\Users\Felix\Desktop\5140 Data Mining\assign4\c1.txt")
    
    single = linkage(c1, 'single', optimal_ordering = True)
    
    complete = linkage(c1,'complete', optimal_ordering = True)
    
    mean = linkage(c1, 'median', optimal_ordering = True)
    
    plt.figure(0)
    
    plt.scatter(c1[:,0], c1[:,1])
    plt.title("C1 Scatter")
    for i in range(len(c1[:,0])):
        if(i == 12):
            plt.annotate(f'{i}', (c1[i,0]+.3, c1[i,1]+.4))
        else:
            plt.annotate(f'{i}', (c1[i,0]+.3, c1[i,1]))
    
    
    plt.figure(1)
    plt.title("Single")
    dn_single = dendrogram(single)
    
    plt.figure(2)
    plt.title("Complete")
    dn_complete = dendrogram(complete)
    
    plt.figure(3)
    plt.title("Mean")
    dn_median = dendrogram(mean)






#%matplotlib inline

sns.set_style("darkgrid")
mpl.rcParams['figure.figsize'] = (6,4)
mpl.rcParams['figure.dpi'] = 200


#From Kaggle Gonzalez notebook
def gonzalez(data, cluster_num, technique = 'max'):
    clusters = []
    clusters.append(data[0]) # let us assign the first cluster point to be first point of the data
    while len(clusters) is not cluster_num:
        if technique is 'max':
            clusters.append(max_dist(data, clusters)) 
        if technique is 'norm':
            clusters.append(norm_dist(data, clusters)) 
        # we add the furthest point from ALL current clusters
    return (clusters)



def max_dist(data, clusters):
    distances = np.zeros(len(data)) # we will keep a cumulative distance measure for all points
    for cluster_id, cluster in enumerate(clusters):
        for point_id, point in enumerate(data):
            if distance.euclidean(point,cluster) == 0.0:
                distances[point_id] = -math.inf # this point is already a cluster (obselete)
            if not math.isinf(distances[point_id]):
                # if a point is not obselete, then we add the distance to its specific bin
                distances[point_id] = distances[point_id] + distance.euclidean(point,cluster) 
                # return the point which is furthest away from all the other clusters
    return data[np.argmax(distances)]




def norm_dist(data, clusters):
    distances = np.zeros(len(data)) # we will keep a cumulative distance measure for all points
    for point_id, point in enumerate(data):
        for cluster_id, cluster in enumerate(clusters):
            if distance.euclidean(point,cluster) == 0.0:
                distances[point_id] = -math.inf # this point is already a cluster (obselete)
            if not math.isinf(distances[point_id]):
                # if a point is not obselete, then we add the distance to its specific bin
                distances[point_id] = distances[point_id] + math.pow(distance.euclidean(point,cluster),2) 
                # return the point which is furthest away from all the other clusters
    for distance_id, current_distance in enumerate(distances):
        if not math.isinf(current_distance): 
            distances[distance_id] = math.sqrt(current_distance/len(data))
    return data[np.argmax(distances)]





def q2_a():

    c2 = pd.read_csv(r"C:\Users\Felix\Desktop\5140 Data Mining\assign4\c2.csv", names=['x0', 'x1'], delim_whitespace=True )
    
    data_set = []
    for index, row in c2.iterrows():
            data_set.append([row['x0'], row['x1']]) 
    data_set = np.array(data_set)
    
    
    
    cluster_points = gonzalez(data_set, 4)
    print('Cluster Centeroids:', cluster_points)
    
    
    
    
    
    cluster_distance = np.full(len(data_set), np.inf)
    for point_idx, point in enumerate(data_set):
        for cluster_idx, cluster_point in enumerate(cluster_points):
            if cluster_distance[point_idx] is math.inf:
                cluster_distance[point_idx] = distance.euclidean(point,cluster_point)
                continue
            if distance.euclidean(point,cluster_point) < cluster_distance[point_idx]:
                cluster_distance[point_idx] = distance.euclidean(point,cluster_point)
    print('4-max cost:', np.max(cluster_distance))
    
    cost = (sum(cluster_distance**2) / len(cluster_distance))**.5
    print('4-means cost:', cost)
    
    
    
    from collections import defaultdict
    
    cluster_sets = defaultdict(list)
    for point in data_set:
        temp_index = -1;
        closest_center = math.inf
        for index, center in enumerate(cluster_points):
            temp_dist = distance.euclidean(center, point)
            
            if(temp_dist < closest_center):
                closest_center = temp_dist
                temp_index = index
        cluster_sets[temp_index].append(point)
    
    cluster_0 = np.array(cluster_sets[0])
    cluster_1 = np.array(cluster_sets[1])
    cluster_2 = np.array(cluster_sets[2])
    cluster_3 = np.array(cluster_sets[3])
    
    plt.title("Gonzalez 4 center")
    plt.scatter(cluster_0[:,0], cluster_0[:,1], color = 'red')
    plt.scatter(cluster_1[:,0], cluster_1[:,1], color = 'blue')
    plt.scatter(cluster_2[:,0], cluster_2[:,1], color = 'yellow')
    plt.scatter(cluster_3[:,0], cluster_3[:,1], color = 'magenta')
    
    
    
    for index, point in enumerate(cluster_points):
            plt.scatter(point[0],point[1], marker='*', c='black', s=50)






    
    
    
    
#From https://austindavidbrown.github.io/post/2019/01/k-means-in-python/    
# Lloyds algorithm for k-means
def kmeans_ll(X, k, i, max_iter = 100, tolerance = 10**(-3)):
  n_samples = X.shape[0]
  n_features = X.shape[1]
  classifications = np.zeros(n_samples, dtype = np.int64)

  # Choose initial cluster centroids randomly
  I = np.random.choice(n_samples, k)
  
  #First 4 data points
  #centroids = X[0:4,:]
  
  #Gonzalez
  #centroids = np.array([[13.51372985, 45.03355641],[58.90178825, 89.90968034],[64.97202851, 15.22366739],[41.14955242, 89.2626484]])
  centroids_all = q2_ab()
  centroids = centroids_all[i]
  loss = 0
  for m in range(0, max_iter):
    # Compute the classifications
    for i in range(0, n_samples):
      distances = np.zeros(k)
      for j in range(0, k):
        distances[j] = np.sqrt(np.sum(np.power(X[i, :] - centroids[j], 2))) 
      classifications[i] = np.argmin(distances)

    # Compute the new centroids and new loss
    new_centroids = np.zeros((k, n_features))
    new_loss = 0
    for j in range(0, k):
      # compute centroids
      J = np.where(classifications == j)
      X_C = X[J]
      new_centroids[j] = X_C.mean(axis = 0)

      # Compute loss
      for i in range(0, X_C.shape[0]):
        new_loss += np.sum(np.power(X_C[i, :] - centroids[j], 2))

    # Stopping criterion            
    if np.abs(loss - new_loss) < tolerance:
      return new_centroids, classifications, new_loss / len(X)
    
    centroids = new_centroids
    loss = new_loss

  print("Failed to converge!")
  return centroids, classifications,


def q2_c():
    c2 = pd.read_csv(r"C:\Users\Felix\Desktop\5140 Data Mining\assign4\c2.csv", names=['x0', 'x1'], delim_whitespace=True )

    data_set = []
    for index, row in c2.iterrows():
            data_set.append([row['x0'], row['x1']]) 
    data_set = np.array(data_set)
    
    all_losses = []
    
    trials = 20
    for i in range(trials):
        centers, classifications, loss = kmeans_ll(data_set, 4, i)    
        all_losses.append(loss)
        
        plt.figure(i)
        fig = plt.figure(figsize=(10, 6))
        fig.tight_layout()
        s1 = plt.subplot(1, 2, 1)
        s1.set_title("Lloyd's: Gonzalez initialized")
        s1.scatter(data_set[:, 0], data_set[:, 1], c = classifications, s = 2)
        s1.scatter(centers[:, 0], centers[:,1], c = "r", s = 20)

    losses_arr = np.array(all_losses)
    loss_arr = np.cumsum(losses_arr) / sum(losses_arr)
    
    
    
#    x = np.linspace(0,1,trials)
#    plt.title("LLoyds 20 Initialized Kmeans++ outputs Cdf:")
#    plt.plot(x, loss_arr)
    
    # Plot





#Modified from the sklearn library, Returns the Centers to start Kmeans with
def _kmeans_plusplus(X, n_clusters, x_squared_norms,
	                     random_state, n_local_trials=None):
	    """Computational component for initialization of n_clusters by
	    k-means++. Prior validation of data is assumed.
	
	    Parameters
	    ----------
	    X : {ndarray, sparse matrix} of shape (n_samples, n_features)
	        The data to pick seeds for.
	
	    n_clusters : int
	        The number of seeds to choose.
	
	    x_squared_norms : ndarray of shape (n_samples,)
	        Squared Euclidean norm of each data point.
	
	    random_state : RandomState instance
	        The generator used to initialize the centers.
	        See :term:`Glossary <random_state>`.
	
	    n_local_trials : int, default=None
	        The number of seeding trials for each center (except the first),
	        of which the one reducing inertia the most is greedily chosen.
	        Set to None to make the number of trials depend logarithmically
	        on the number of seeds (2+log(k)); this is the default.
	
	    Returns
	    -------
	    centers : ndarray of shape (n_clusters, n_features)
	        The inital centers for k-means.
	
	    indices : ndarray of shape (n_clusters,)
	        The index location of the chosen centers in the data array X. For a
	        given index and center, X[index] = center.
	    """
	    n_samples, n_features = X.shape
	
	    centers = np.zeros((n_clusters, n_features), dtype=X.dtype)
	
	    # Set the number of local seeding trials if none is given
	    if n_local_trials is None:
	        # This is what Arthur/Vassilvitskii tried, but did not report
	        # specific results for other than mentioning in the conclusion
	        # that it helped.
	        n_local_trials = 2 + int(np.log(n_clusters))
	
	    # Pick first center randomly and track index of point
	    center_id = 0
	    indices = np.full(n_clusters, -1, dtype=int)
	    if sp.issparse(X):
	        centers[0] = X[center_id].toarray()
	    else:
	        centers[0] = X[center_id]
	    indices[0] = center_id
	
	    # Initialize list of closest distances and calculate current potential
	    closest_dist_sq = euclidean_distances(
	        centers[0, np.newaxis], X)#, Y_norm_squared=x_squared_norms,
	        #squared=True)
	    current_pot = closest_dist_sq.sum()
	
	    # Pick the remaining n_clusters-1 points
	    for c in range(1, n_clusters):
	        # Choose center candidates by sampling with probability proportional
	        # to the squared distance to the closest existing center
	        rand_vals = random_state.random_sample(n_local_trials) * current_pot
	        candidate_ids = np.searchsorted(stable_cumsum(closest_dist_sq),
	                                        rand_vals)
	        # XXX: numerical imprecision can result in a candidate_id out of range
	        np.clip(candidate_ids, None, closest_dist_sq.size - 1,
	                out=candidate_ids)
	
	        # Compute distances to center candidates
	        distance_to_candidates = euclidean_distances(
	            X[candidate_ids], X)#, Y_norm_squared=x_squared_norms, squared=True)
	
	        # update closest distances squared and potential for each candidate
	        np.minimum(closest_dist_sq, distance_to_candidates,
	                   out=distance_to_candidates)
	        candidates_pot = distance_to_candidates.sum(axis=1)
	
	        # Decide which candidate is the best
	        best_candidate = np.argmin(candidates_pot)
	        current_pot = candidates_pot[best_candidate]
	        closest_dist_sq = distance_to_candidates[best_candidate]
	        best_candidate = candidate_ids[best_candidate]
	
	        # Permanently add best center candidate found in local tries
	        if sp.issparse(X):
	            centers[c] = X[best_candidate].toarray()
	        else:
	            centers[c] = X[best_candidate]
	        indices[c] = best_candidate
	
	    return centers, indices
	
    
    
    
    
    


def kmeans_pl_pl(plot):
    c2 = pd.read_csv(r"C:\Users\Felix\Desktop\5140 Data Mining\assign4\c2.csv", names=['x0', 'x1'], delim_whitespace=True )
    
    data_set = []
    for index, row in c2.iterrows():
            data_set.append([row['x0'], row['x1']]) 
    data_set = np.array(data_set)
    
    #Note the norm doesnt' do anything but I got it working and was afraid to take it out
    start_centers, st_cent_indices = _kmeans_plusplus(data_set,4,np.linalg.norm(data_set, axis =1), np.random.RandomState(None), None)
    #These centers are built from kmeans++ modified where the start point is always X[0] as detailed by assign4
    print(start_centers)
    kmeans = KMeans(n_clusters = 4, init = start_centers, n_init = 1).fit(data_set)
    
    
    cluster_map = pd.DataFrame()
    cluster_map['data_index'] = c2.index.values
    cluster_map['cluster'] = kmeans.labels_
    
    y_kmeans = kmeans.predict(data_set)
    
   # plt.figure(plot)
    #plt.title("Kmeans++")
    #plt.scatter(data_set[:,0], data_set[:,1], c= y_kmeans, s=50, cmap ='viridis')
    
    centers = kmeans.cluster_centers_
    
    #plt.scatter(centers[:,0], centers[:,1], c='black', s=100, alpha = .8)
    # sqrt (sum of distances(center, each point ^2)/Size(X)) 
    cluster_0 = cluster_map[cluster_map.cluster == 0].to_numpy()
    cluster_0_data = data_set[cluster_0[:,0]]
    
    cluster_0_distances = (euclidean_distances(centers[0,np.newaxis], cluster_0_data))
    cluster_0_max = np.max(cluster_0_distances)
    cluster_0_cost = math.sqrt((cluster_0_distances**2).sum()) / len(data_set)
    
    cluster_1 = cluster_map[cluster_map.cluster == 1].to_numpy()
    cluster_1_data = data_set[cluster_1[:,0]]
    
    cluster_1_distances = (euclidean_distances(centers[0,np.newaxis], cluster_1_data))
    cluster_1_max = np.max(cluster_1_distances)
    cluster_1_cost = math.sqrt((cluster_0_distances**2).sum()) / len(data_set)
    
    cluster_2 = cluster_map[cluster_map.cluster == 2].to_numpy()
    cluster_2_data = data_set[cluster_2[:,0]]
    
    cluster_2_distances = (euclidean_distances(centers[0,np.newaxis], cluster_2_data))
    cluster_2_max = np.max(cluster_2_distances)
    cluster_2_cost = math.sqrt((cluster_2_distances**2).sum()) / len(data_set)
    
    cluster_3 = cluster_map[cluster_map.cluster == 0].to_numpy()
    cluster_3_data = data_set[cluster_3[:,0]]
    
    cluster_3_distances = (euclidean_distances(centers[0,np.newaxis], cluster_3_data))
    cluster_3_max = np.max(cluster_3_distances)
    cluster_3_cost = math.sqrt((cluster_3_distances**2).sum()) / len(data_set)
    
    abs_max = max(cluster_0_max,cluster_1_max,cluster_2_max,cluster_3_max)
    
    
    cluster_cost_sum = cluster_0_cost + cluster_1_cost + cluster_2_cost + cluster_3_cost 
    
    #inertia = kmeans.inertia_
    return cluster_cost_sum , centers, abs_max



def q2_ab():
    inertial_values = []
    centers = []
    trials = 20
    for i in range(trials):
        cost, center, max_cost = (kmeans_pl_pl(i))
        centers.append(center)
        inertial_values.append(cost)

        #print(f'Trial: {i}\n: Max:{max_cost}\n Means:{cost}\nCenters:]n{centers}')
    inertial_values_arr = np.array(inertial_values)
    inertial_values_arr =np.cumsum(inertial_values_arr) / sum(inertial_values)
    
    x = np.linspace(0,1,trials)
    #plt.title("Kmeans++ Cost CDF")
    #plt.plot(x, inertial_values_arr)
    
    return centers
    
    
######################### Uncomment for values #######################################
#q1()
#q2_a()
#q2_ab()    
#q2_c()    








c1 = np.loadtxt(r"C:\Users\Felix\Desktop\5140 Data Mining\assign4\test.txt")

single = linkage(c1, 'single', optimal_ordering = True)

#complete = linkage(c1,'complete', optimal_ordering = True)

#mean = linkage(c1, 'median', optimal_ordering = True)

plt.figure(0)

plt.scatter(c1[:,0], c1[:,1])
plt.title("C1 Scatter")
for i in range(len(c1[:,0])):
    if(i == 12):
        plt.annotate(f'{i}', (c1[i,0]+.3, c1[i,1]+.4))
    else:
        plt.annotate(f'{i}', (c1[i,0]+.3, c1[i,1]))


plt.figure(1)
plt.title("Single")
dn_single = dendrogram(single)

plt.figure(2)
plt.title("Complete")
dn_complete = dendrogram(complete)

plt.figure(3)
plt.title("Mean")
dn_median = dendrogram(mean)


