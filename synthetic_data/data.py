from abc import ABC, abstractmethod
from  synthetic_data.cluster_size import *
from  synthetic_data.outlier import *
from  synthetic_data.cluster_label import *
from  synthetic_data.regression_params import *
from  synthetic_data.target import *
import numpy as np
import json
import os

class Data(ABC):
    '''
    A parent class for both SyntheticData and RealData classes
    '''

    def __init__(self, **kwargs):
        self.N = None # number of data points across all clusters
        self.K = None # number of clusters
        self.D = None # dimension of the data

        self.X = None # data points (NxD)
        self.y = None # regression target (Nx1)
        self.z = None # cluster labels (Nx1)

        self.random = np.random.RandomState(kwargs.get('random_state', 0))
        self.name = self.__class__.__name__ # name of the dataset

        self.train = None # index for training data
        self.val = None # index for validation data

        self.noise_std = 0.00 # noise level for synthetic data
        self.outlier_ratio = 0.00 # outlier ratio for synthetic data
    @abstractmethod
    def load_data(self, file_path:str):
        '''
        Load data from a json file with the following format:
        {
            'X': data points (NxD),
            'y': regression target (Nx1),
            'z': cluster labels (Nx1)
            'other': (optional) other parameters (noise_std, outlier_ratio, etc.)
        }
        '''
        pass

    @abstractmethod
    def save_data(self, file_path:str):
        '''
        Save data to a json file with the following format:
        {
            'X': data points (NxD),
            'y': regression target (Nx1),
            'z': cluster labels (Nx1)
            'other': other parameters (noise_std, outlier_ratio, etc.)
        }
        '''
        pass

class SyntheticData(Data, ABC):
    '''
    A class to generate or load synthetic data with regression target and cluster labels
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # additional attributes for synthetic data
        self.input_arg = {} # input arguments for generating synthetic data
        self.cluster_params = {} # regression weights and bias for each cluster

        self.noise_std = None # noise level for synthetic data
        self.outlier_ratio = None # outlier ratio for synthetic data

        self.ground_truth_mse = None # ground truth mse error
        self.ground_truth_r2 = None # ground truth r2 score

        self.loss = 'MSE'

        # customizable class for generating synthetic data
        self.cluster_size_generator = None # class to generate number of data points per cluster
        self.cluster_label_generator = None # class to generate cluster labels for each data point
        self.regression_params_generator = None # class to generate regression weights and bias for each cluster
        self.regression_target_generator = None # class to generate regression target
        self.outlier_generator = None # class to generate outliers

        self.regression_weights = {} # regression weights and bias for each cluster

    def set_generator(self, generator:ClusterSize|ClusterLabel|RegParams|Outlier|RegTarget):
        '''
        Set the class to generate
        '''
        if isinstance(generator, ClusterSize):
            self.cluster_size_generator = generator
        elif isinstance(generator, ClusterLabel):
            self.cluster_label_generator = generator
        elif isinstance(generator, RegParams):
            self.regression_params_generator = generator
        elif isinstance(generator, Outlier):
            print('setting outlier generator')
            self.outlier_generator = generator
        elif isinstance(generator, RegTarget):
            self.regression_target_generator = generator

        self.name += f'_{generator.__class__.__name__}'

    def _generate_regression_params(self):
        '''
        Generate regression weight and bias for each cluster based on self.K
        '''
        if self.regression_params_generator == None:
            self.regression_params_generator = Uniform()

        self.cluster_params = self.regression_params_generator.regression_params(self.K, self.D, self.random)


    def _generate_outliers(self):
        '''
        Add outliers to the data basd on the outlier ratio provided
        '''
        # obtain the number of outliers per cluster
        if self.outlier_generator == None:
            self.outlier_generator = ByIQR(outlier_ratio=self.outlier_ratio)

        self.y = self.outlier_generator.add_outliers(self.N, self.K, self.y, self.z, self.random)
                
    def _generate_num_points_per_cluster(self) -> list:
        '''
        Generate number of data points per cluster based on total number of data points and number of clusters or based on distribution of data points per cluster
        '''
        if self.cluster_size_generator == None:
            self.cluster_size_generator = FixedClusterSize()
            
        return self.cluster_size_generator.cluster_size(self.N, self.K, self.random)
        
    def _generate_cluster_labels(self):
        '''
        Iniltizing data point X and generate cluster labels for each data point based on the number of clusters
        '''
        if self.cluster_label_generator == None:
            self.cluster_label_generator = EqualRange()

        points_per_k = self._generate_num_points_per_cluster()
        self.X, self.z = self.cluster_label_generator.cluster_labels(self.N, self.D, points_per_k, self.random)

    def _generate_regression_target(self, y = None):
        '''
        Generate regression target (y) for each cluster based on the regression weights 
        '''
        if self.regression_target_generator == None:
            self.regression_target_generator = GaussianNoise(noise_std=self.noise_std)
        if y is not None:
            self.y =self.regression_target_generator.add_noise(y, self.noise_std, self.random)
        else:
            self.y = self.regression_target_generator.regression_target(self.X, self.cluster_params, self.z, self.random)
    
    def _generate_ground_truth_mse(self):
        '''
        Calculate the ground truth mse and r2 error
        '''
        y_pred = np.zeros((self.N))
        for k in range(self.K):
            indices = np.where(self.z == k)[0]
            weights_k = self.cluster_params[k]['weights']
            bias_k = self.cluster_params[k]['bias']
            y_pred[indices] = self.X[indices] @ weights_k + bias_k

        if self.loss == 'MSE':
            mse = np.mean((self.y - y_pred)**2)
            print('mse', mse)
        else:
            mse = np.mean(np.abs(self.y - y_pred))  

        r2 = 1 - mse / np.var(self.y)
        
        self.ground_truth_mse = mse
        self.ground_truth_r2 = r2
        

    def generate_data(self, N:int, K:int, D:int, noise_std:float = 1.00, 
                      outlier_ratio:float = 0.00, overwrite = False, 
                      validation = False, use_exist = False, **kwargs):
        '''
        Generate synthetic data by adding gaussian noise as bias into each cluster's regression target

        Parameters:
        N: int - number of data points across all clusters
        K: int - number of clusters
        D: int - dimension of the data

        noise_std: float - noise level for synthetic data
        outlier_ratio: float - outlier ratio for synthetic data

        overwrite: bool - whether to overwrite the existing data
        validation: bool - whether to split the data into training and validation set
        '''

        try :
            if self.X is not None and not overwrite and not use_exist:
                print('Error: Data already exists. Set overwrite=True to overwrite the existing data')
                return
            elif self.X is not None and overwrite:
                print('Warning: Overwriting the existing data')
                self.X = None
                self.y = None
                self.z = None
             
            for input in [N, K, D, noise_std]:
                assert input > 0, f'Invalid input value for {input}'
            
            if self.outlier_generator.__class__.__name__ == 'ByIQR':
                assert 0 <= outlier_ratio <= 1, 'Outlier ratio should be between 0 and 1'

            assert N >= K, 'There should be at least one data point per cluster'

            self.input_arg = kwargs # store input arguments for accessing later

            self.noise_std = noise_std
            self.outlier_ratio = outlier_ratio

            if not use_exist:

                self.N = N
                self.K = K
                self.D = D

                # Generate regression weights and bias for each cluster
                self._generate_regression_params()

                # Generate cluster labels
                self._generate_cluster_labels()
                
                # Generate y for each cluster
                self._generate_regression_target()
            
            else:
                assert self.y is not None, 'y should be provided to use existing data'
                self._generate_regression_target(self.y) # add noise to the existing y

            # Add outliers to the data based on the outlier ratio provided   
            self._generate_outliers()

            #predict the y_pred using ground truth weights and bias and calculate the ground truth mse error based on the regression target
            self._generate_ground_truth_mse()

            self.name += f'_{N}_{K}_{D}_{noise_std}_{outlier_ratio}'

            val_flag = False

            if use_exist:
                val_flag = self.val is not None

            if validation and not val_flag:
                print('Splitting data into training and validation set')
                total_val = int(0.4 * self.N)
                total_val_per_cluster = int(total_val / self.K)
                self.train = []
                self.val = []
                for k in range(self.K):
                    indices = np.where(self.z == k)[0]
                    val_indices = self.random.choice(indices, total_val_per_cluster, replace=False)
                    train_indices = np.setdiff1d(indices, val_indices)
                    self.train.extend(train_indices)
                    self.val.extend(val_indices)
            
            elif not use_exist:
                self.train = np.arange(self.N)
                self.val = None

        except AssertionError as e:
            raise ValueError('Invalid input parameters')
        
        
    def load_data(self, file_path:str):
        '''
        Load synthetic data from a json file with the following format:
        {
            'X': data points (NxD),
            'y': regression target (Nx1),
            'z': cluster labels (Nx1)
            'other': (optional) other parameters (noise_std, outlier_ratio, etc.)
        }
        '''
        data_dic = json.load(open(file_path, 'r'))
        
        self.N = np.shape(data_dic['X'])[0] # number of data points across all clusters
        self.K = len(np.unique(data_dic['z'])) # number of clusters
        self.D = np.shape(data_dic['X'])[1] # dimension of the data
        for k, weights in enumerate(data_dic['regression_weights']):
            self.cluster_params[k] = {'bias': weights[0], 'weights': np.array(weights[1:])}

        self.input_arg = data_dic['input_args'] # input arguments for generating synthetic data

        if 'noise_std' in self.input_arg.keys():
            self.noise_std = self.input_arg['noise_std']
        if 'outlier_ratio' in self.input_arg.keys():
            self.outlier_ratio = self.input_arg['outlier_ratio']

        if 'train' in data_dic.keys():
            self.train = np.array(data_dic['train'])
            self.val = np.array(data_dic['val']) if data_dic['val'] is not None else None
        else:
            self.train = np.arange(self.N)
            self.val = None
        

        print(f'Number of data points : {self.N}, Number of clusters : {self.K}, Dimension of the data : {self.D}')

        self.X = np.array(data_dic['X']) #data points
        self.y = np.array(data_dic['y']) #ground truth regression target
        self.z = np.array(data_dic['z']) #ground truth cluster labels

    
    def save_data(self, file_path:str):
        '''
        Save synthetic data to a json file with the following format:
        {
            'X': data points (NxD),
            'y': regression target (Nx1),
            'z': cluster labels (Nx1)
            'other': other parameters (noise_std, outlier_ratio, etc.)
        }
        '''
        input_args = {'N': self.N, 'K': self.K, 'D': self.D, 'noise_std': self.noise_std, 'outlier_ratio': self.outlier_ratio}
        #append others with self.input_arg
        if len(self.input_arg) > 0:
            input_args.update(self.input_arg)

        data_dic = {
            'X': self.X.tolist(),
            'y': self.y.tolist(),
            'z': self.z.tolist(),
            'regression_weights': [[self.cluster_params[k]['bias']] + self.cluster_params[k]['weights'].tolist() for k in range(self.K)],
            'ground_truth_mse': self.ground_truth_mse,
            'ground_truth_r2': self.ground_truth_r2,
            'train': self.train,
            'val': self.val if self.val is not None else None,
            'input_args':  input_args
        }
        if not file_path.endswith('.json'):
            file_path = os.path.join(file_path, f'{self.name}.json')
        
        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super(NpEncoder, self).default(obj)
    
        with open(file_path, 'w') as f:
            json.dump(data_dic, f, cls=NpEncoder)
        


