from abc import ABC, abstractmethod
import numpy as np

class ClusterSize(ABC):
    def __init__(self):
        self.N = None
        self.K = None
    
    @abstractmethod
    def cluster_size(self, N, K, random_state=None):
        """
        N: Number of data points
        K: Number of clusters
        random_state: Random seed for reproducibility
        """

        pass

class FixedClusterSize(ClusterSize):

    def __init__(self):
        super().__init__()

    def cluster_size(self, N, K, random_state=None):
        self.N = N
        self.K = K
        points_per_cluster = self.N // self.K
        return [points_per_cluster for _ in range(self.K)]
    
class ImbalancedClusterSize(ClusterSize):

    def __init__(self, distribution):
        super().__init__()
        self.distribution = distribution

    def cluster_size(self, N, K, random_state=None):
        self.N = N
        self.K = K
        self.random = random_state
        try:
            if self.distribution is None:
                self.distribution = self.random.dirichlet(np.ones(self.K))
    
            assert np.sum(self.distribution) == 1, 'Distribution of data points per cluster should sum to 1'
            assert len(self.distribution) == self.K, 'Distribution of data points per cluster should match the total number of clusters'

            points_per_cluster = [int(self.N * p) for p in self.distribution]
            #handle any remaining points
            remaining = self.N - np.sum(points_per_cluster)
            points_per_cluster[-1] += remaining
            return points_per_cluster
        
        except AssertionError as e:
            print(e)
            raise ValueError('Invalid input parameters')