from abc import ABC, abstractmethod
import numpy as np

class ClusterLabel(ABC):
    def __init__(self):
        self.X = None
        self.D = None
        self.N = None
        self.random = None
        self.z = None

    @abstractmethod
    def cluster_labels(self, N:int, D:int, points_per_k:np.ndarray, random_state=None):
        """
        Generate cluster labels for each data point
        
        params:
        N: Number of data points
        K: Number of clusters
        points_per_k: Number of data points per cluster
        random_state: Random seed for reproducibility

        return:
        X: Data points
        z: Cluster labels
        """
        pass



class EqualRange(ClusterLabel):
    '''
    Iniltizing data point X and generate cluster labels for each data point based on the number of clusters where X in different clusters are uniformly distributed within same range
    '''

    def __init__(self):
        super().__init__()

    def cluster_labels(self, N, D, points_per_k, random_state=None):
        self.N = N
        self.D = D
        self.random = random_state
        # initialize data points and cluster labels
        self.X = self.random.uniform(-10, 10, size=(self.N, self.D)) # data points
        self.z = np.zeros((self.N)) # cluster labels
        
        self.z = np.repeat([i for i in range(len(points_per_k))], points_per_k)
        # Handle any remaining points
        if len(self.z) < self.N:
            remaining = self.N - len(self.z)
            self.z = np.concatenate([self.z,
                self.random.choice([i for i in range(len(points_per_k))], remaining)])
        self.random.shuffle(self.z)

        return self.X, self.z

class UnequalRange(ClusterLabel):

        '''
        Generate cluster labels for each data point based on the number of clusters

        For each cluster, initialize data points self.X separately to ensure each cluster can have different shapes

        '''
        def __init__(self, range_by_k = None):
            super().__init__()
            self.range_by_k = range_by_k 

        def cluster_labels(self, N, D, points_per_k, random_state=None):
            self.N = N
            self.D = D
            self.K = len(points_per_k)
            self.z = np.repeat([i for i in range(len(points_per_k))],  points_per_k)
            self.random = random_state
            # Handle any remaining points
            if len(self.z) < self.N:
                remaining = self.N - len(self.z)
                self.z = np.concatenate([self.z,
                    self.random.choice([i for i in range(len(points_per_k))], remaining)])
            self.random.shuffle(self.z)

            self.X = np.zeros((self.N, self.D))
            # random create k ranges without overlap
            for k in range(self.K):
                indices = np.where(self.z == k)[0]
                if self.range_by_k is not None:
                    if len(self.range_by_k[k]) == 1:
                        # custom range for each cluster
                        start_point = [self.range_by_k[k][0][0]] * self.D
                        end_point = [self.range_by_k[k][0][1]] * self.D
                    else:
                        start_point = [self.range_by_k[k][d][0] for d in range(self.D)]
                        end_point = [self.range_by_k[k][d][1] for d in range(self.D)]
                else:
                    # randomly assign range for each cluster
                    start_point = [self.random.uniform(-1, 1, 1)  - k for _ in range(self.D)]
                    end_point = [self.random.uniform(start_point, 10, 1) + k for _ in range(self.D)]
                    
                x_current = []
                for d in range(self.D):
                    x_current.append(self.random.uniform(start_point[d], end_point[d], len(indices)))
                
                self.X[indices] = np.array(x_current).T

            return self.X, self.z
        
class SelfDefinedX(ClusterLabel):
    '''
    Generate cluster labels for each data point based on the number of clusters

    '''
    def __init__(self, X, z):
        super().__init__()
        self.X = X
        self.z = z

    def cluster_labels(self, N, D, points_per_k, random_state=None):
        return self.X, self.z
        

