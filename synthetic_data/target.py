from abc import ABC, abstractmethod
import numpy as np

class RegTarget(ABC):
    def __init__(self):
        super().__init__()

        self.cluster_params = None # the dictionary containing regression weights and bias for each cluster to generate target variable y
        self.X = None # data points
        self.N = None # number of data points
        self.z = None # cluster labels

    @abstractmethod
    def regression_target(self, X:np.ndarray, cluster_params:dict, z:np.ndarray, random_state=None) -> np.ndarray:
        """
        Generate target variable y based on the regression weights and bias for each cluster
        y = X * weights + bias
        
        params:
        X: Data points
        cluster_params: Dictionary containing regression weights and bias for each cluster
        z: Cluster labels
        
        return:
        y: Target variable
        """
        pass

class GaussianNoise(RegTarget):
    '''
    Add Gaussian noise to the target variable y
    '''

    def __init__(self, noise_std:float):
        super().__init__()
        self.noise_std = noise_std

    def regression_target(self, X, cluster_params, z, random_state=None):
        self.X = X
        self.cluster_params = cluster_params
        self.z = z
        self.random = random_state

        self.N = len(self.X)
        self.K = np.max(self.z) + 1  
        self.y = np.zeros((self.N))

        for k in range(self.K):
            indices = np.where(self.z == k)[0]
            weights_k = self.cluster_params[k]['weights']
            bias_k = self.cluster_params[k]['bias']
            self.y[indices] = self.X[indices] @ weights_k + bias_k
            noise = self.random.normal(0, self.noise_std, len(indices))
            self.y[indices] = self.y[indices] + noise

        return self.y
    
    def add_noise(self, y, noise_std, random_state=None):
        if random_state is not None:
            self.random = random_state
        else:
            self.random = np.random.RandomState(np.random.randint(0, 1000))
            
        return y + self.random.normal(0, noise_std, len(y))

