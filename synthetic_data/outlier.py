from abc import ABC, abstractmethod
import numpy as np

class Outlier(ABC):
    def __init__(self):
        self.N = None
        self.y = None
        self.z = None
        self.random = None

    @abstractmethod
    def add_outliers(self, N:int, K:int, y:np.ndarray, cluster_label:np.ndarray, random_state=None):
        """
        Adding outliers to the target variable y 
        
        params:
        N: Number of data points
        K: Number of clusters
        y: Target variable
        cluster_label: Cluster labels
        random_state: Random seed for reproducibility

        return:
        y: Target variable with outliers

        """
        pass


class ByIQR(Outlier):
    '''
        using the 1.5 * IQR to 3 IQR to generate outliers
    '''
    def __init__(self, outlier_ratio:float):
        super().__init__()
        self.outlier_ratio = outlier_ratio

    def add_outliers(self, N, K, y, cluste_label, random_state=None):
        
        self.random = random_state
        self.N = N
        self.K = K
        self.y = y
        self.z = cluste_label

        if self.outlier_ratio > 0:
            # Number of outliers per cluster
            num_outliers = int(self.N * self.outlier_ratio)
            num_outliers_k = num_outliers // self.K

            for k in range(self.K):
                indices = np.where(self.z == k)[0] 
                outlier_indices = self.random.choice(indices, num_outliers_k, replace=False)
                # Add outliers to the data based on range of y values in the cluster
                iqr = np.percentile(self.y[indices], 75) - np.percentile(self.y[indices], 25)
                negative_upper_bound = np.percentile(self.y[indices], 25) - 2 * iqr
                positive_upper_bound = np.percentile(self.y[indices], 75) + 2 * iqr
                negative_lower_bound = np.percentile(self.y[indices], 25) - 3 * iqr
                positive_lower_bound = np.percentile(self.y[indices], 75) + 3 * iqr

                # Adding outliers to the data within outlier_indices
                self.y[outlier_indices] = self.y[outlier_indices] + \
                self.random.choice([self.random.uniform(negative_lower_bound, negative_upper_bound), 
                                    self.random.uniform(positive_lower_bound, positive_upper_bound)], num_outliers_k)
        return self.y
    

class ByIQRDistance(Outlier):
    '''
        using the 1.5 * IQR to 3 IQR to generate outliers times (1 + outlier_ratio), as outlier_ratio increase, the distance between the outlier and the cluster increases
    '''
    def __init__(self, outlier_ratio:float, outlier_distance:float):
        super().__init__()
        self.outlier_ratio = outlier_ratio
        self.outlier_distance = outlier_distance

    def add_outliers(self, N, K, y, cluste_label, random_state=None):
        
        self.random = random_state
        self.N = N
        self.K = K
        self.y = y
        self.z = cluste_label
        if self.outlier_ratio > 0:
            # Number of outliers per cluster
            num_outliers = int(self.N * self.outlier_ratio)
            num_outliers_k = num_outliers // self.K

            for k in range(self.K):
                indices = np.where(self.z == k)[0] 
                outlier_indices = self.random.choice(indices, num_outliers_k, replace=False)
                # Add outliers to the data based on range of y values in the cluster
                iqr = np.percentile(self.y[indices], 75) - np.percentile(self.y[indices], 25)
                negative_upper_bound = np.percentile(self.y[indices], 25) - self.outlier_distance * iqr
                positive_upper_bound = np.percentile(self.y[indices], 75) + self.outlier_distance * iqr
                negative_lower_bound = np.percentile(self.y[indices], 25) - (1 + self.outlier_distance) * iqr
                positive_lower_bound = np.percentile(self.y[indices], 75) + (1 + self.outlier_distance) * iqr

                # Adding outliers to the data within outlier_indices
                self.y[outlier_indices] = self.y[outlier_indices] + \
                self.random.choice([self.random.uniform(negative_lower_bound, negative_upper_bound), 
                                    self.random.uniform(positive_lower_bound, positive_upper_bound)], num_outliers_k)
                # self.y[outlier_indices] = self.y[outlier_indices] * (1 + self.outlier_distance)
        return self.y
                
