from abc import ABC, abstractmethod
import numpy as np

class RegParams(ABC):
    def __init__(self):
        self.K = None
        self.random = None

    @abstractmethod
    def regression_params(self, K:int, D:int, random_state=None) -> dict:
        """
        Generate regression weights and bias for each cluster
        
        params:
        K: Number of clusters
        D: Number of features
        random_state: Random seed for reproducibility

        return:
        cluster_params: Dictionary containing regression weights and bias for each cluster
        {k: {'weights': weights_k, 'bias': bias}}
        
        """
        pass

class Uniform(RegParams):
    '''
    Generate random regression weight and bias for each cluster using uniform distribution
    '''
    def __init__(self):
        super().__init__()

    def regression_params(self, K, D, random_state=None):
        '''
        Generate regression weight and bias for each cluster based on self.K
        '''
        self.K = K
        self.D = D
        self.random = random_state

        self.cluster_params = {}
    
        for k in range(self.K):
            # Randomly assign bias and weights for each cluster
            weights_k = self.random.uniform(-5, 5, size=self.D)
            bias = self.random.uniform(-3, 3, 1)
            self.cluster_params[k] = {'weights': weights_k, 'bias': bias}
        
        return self.cluster_params
    
class DifferentSlope(RegParams):
    def __init__(self):
        super().__init__()

    def regression_params(self, K, D, random_state=None):
        '''
        Generate regression weight and bias for each cluster
        '''
        self.K = K
        self.D = D
        self.random = random_state

        self.cluster_params = {}

        for k in range(1, self.K + 1):
            # Ensure that the weights have different shapes
            weights_k = []
            if k % 2 == 0:
                for d in range(self.D):
                    if d % 2 == 0:
                        weights_k.append(self.random.uniform(-1, -2))
                    else:
                        weights_k.append(self.random.uniform(1, 2))
            else:
                for d in range(self.D):
                    if d % 2 == 0:
                        weights_k.append(self.random.uniform(1, 2))
                    else:
                        weights_k.append(self.random.uniform(-1, -2))

            self.cluster_params[k-1] = {'weights': np.array(weights_k), 'bias': [0]}
        
        return self.cluster_params
    

class SelfDefinedSlope(RegParams):
    def __init__(self, slope_by_k, bias_by_k):
        super().__init__()
        self.slope_by_k = slope_by_k
        self.bias_by_k = bias_by_k

    def regression_params(self, K, D, random_state=None):
        '''
        Generate regression weight and bias for each cluster
        '''
        self.K = K
        self.D = D
        self.random = random_state

        self.cluster_params = {}

        for k in range(self.K):
            # Ensure that the weights have different shapes
            weights_k = self.slope_by_k[k]
            bias = self.bias_by_k[k]
            self.cluster_params[k] = {'weights': weights_k, 'bias': bias}

        return self.cluster_params