
from abc import ABC, abstractmethod
from scripts.Greedy_Codes.utils import *
import numpy as np
from scipy.spatial import distance

class Assignment(ABC):
    @abstractmethod
    def assign_cluster(self, clus_data,K,f):
        """
        :param clus_data:    Data with previous cluster assignment
        :param K:            Number of clusters   

        """

        raise NotImplementedError


class ArbitraryAssign(Assignment):
  
    def assign_cluster(self, clus_data, K, f):
        clus_data = clus_data.copy()
        new_model = []
        loss_best_pre = 0  # Initialize loss_best_pre

        for _, row in clus_data.iterrows():
            subset = row.iloc[-K:]
        
            min_col = subset.idxmin()                            # e.g., 'loss1'
            
            # Extract the numeric part from 'loss1' to get the cluster index
            try:
                cluster_idx = int(min_col.replace('loss', ''))
            except ValueError:
                print(f"Invalid column format: {min_col}")
                cluster_idx = -1  # Assign a default or handle appropriately
            
            new_model.append(cluster_idx)
        
        # Calculate loss_best_pre
        for i in range(K):
            loss_column = f'loss{i+1}'
            if loss_column in clus_data.columns:
                loss_best_pre += clus_data[clus_data['model'] == i+1][loss_column].sum()
            else:
                raise KeyError(f"Column {loss_column} does not exist in clus_data.")
        
        loss_best_pre /= clus_data.shape[0]

        return new_model, loss_best_pre

class ClosestCentroid(ArbitraryAssign):


    def assign_cluster(self,clus_data,K,f):
        
        clus_data = clus_data.copy()
        new_model = []
        clus_data['model'], loss_best_pre = super().assign_cluster(clus_data,K,f)

        centroid = centroids(clus_data, K, f, with_y = False)
        
        centroid_dist = np.zeros((len(clus_data),K))

        for i in range(K):
            centroid_dist[:,i] = np.sum((clus_data.iloc[:,0:f] - centroid[i])**2,axis = 1)
        
        new_model = np.argmin(centroid_dist, axis=1)+1

        return new_model, loss_best_pre

class BoundingBox(ArbitraryAssign):


    def assign_cluster(self,clus_data,K,f):

        clus_data = clus_data.copy()
        new_model = []
        clus_data['model'], loss_best_pre = super().assign_cluster(clus_data,K,f)


        # centroid = medians(clus_data, K, f, with_y = False)
        centroid = centroids(clus_data, K, f, with_y = False)

        # print(centroid)
        centroid_dist = np.zeros((len(clus_data),K))

        for i in range(K):
            for j in range(len(clus_data)):
                centroid_dist[j,i] = distance.chebyshev(clus_data.iloc[j,0:f], centroid[i])
                # centroid_dist[j,i] = distance.cityblock(clus_data.iloc[j,0:f], centroid[i])

        
        
        new_model = np.argmin(centroid_dist, axis=1)+1
 
        # for i in range(len(new_model)):
        #     print(centroid_dist[i,:], new_model[i])

        return new_model, loss_best_pre
