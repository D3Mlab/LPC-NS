from  synthetic_data.data import SyntheticData
from  synthetic_data.cluster_size import *
from  synthetic_data.outlier import *
from  synthetic_data.cluster_label import *
from  synthetic_data.regression_params import *
from  synthetic_data.target import *

from evaluation.eval import Evaluation
from scripts.model import LpcNsMip, LpcNsQbpo, GobalOpt

import numpy as np
import os

'''
This script is used to run the experiments for RQ2: Performance analysis on synthetic datasets - Level of Guassian Noise in Target Variable.

Notes: This script requires gurobi to be installed with valid license to run the MIP model.

'''

parameter = 'noise_std'

models = [
['MIP','MSE','lpc_ns_mip'], # model name, loss function, path to save the results
['QPBO','QPBO','lpc_ns_qpbo'],
['GlobalOpt','MSE','globalopt'],
]

print('Running experiments for RQ2 (K = 2): Performance analysis on synthetic datasets - Level of Guassian Noise in Target Variable')

for random_state in [42, 12, 3, 4, 2, 5, 6, 7, 8, 9]:
    for value in [0.6, 1.5, 2.3, 3, 3.5, 4]:
            for model in models:
                print(f'Running {model[0]} model with {model[1]} loss function for noise_std = {value} and random_state = {random_state}')
                synthetic = SyntheticData(random_state=42) # Initialize the synthetic data class

                # set the seed for same X across different trails (different trails only change random states in noise in Y not X)
                np.random.seed(2) 
                n_samples = int(200/2)
                # To ensure non-separable data in feature space, we generate data with same mean and variance from gaussian distribution
                x1 = np.random.normal([1]*3, [2]*3, size=(n_samples ,3 )) # mean = 1, variance = 2 dimension = 3 cluster 1
                x2 = np.random.normal([1]*3, [2]*3, size=(n_samples ,3 )) # mean = 1, variance = 2 dimension = 3 cluster 2
                x = np.concatenate([x1, x2]) 
                z = np.concatenate([[0]*n_samples, [1]*n_samples])
                # regression coefficients
                reg_coef =  SelfDefinedSlope([np.array([2, 4, 3])/10, 
                    np.array([-2, -4, -3])/10], [np.array([-10]),np.array([10])])
                
                synthetic.set_generator(reg_coef)
                synthetic.set_generator(SelfDefinedX(x, z)) 

                path = os.path.join('experiments/results', model[-1]) # path to save the results

                if model[0] == 'GlobalOpt':
                    model_class = GobalOpt(verbose =True) # Initialize the model class
                elif model[0] == 'MIP':
                    model_class = LpcNsMip(verbose =True) # LPC-NS-MIP
                else:
                    model_class = LpcNsQbpo(verbose =True) # LPC-NS-QBPO
                
                evaluation = Evaluation.evaluate_model(synthetic,model_class,
                                                        verbose = True, validate=True,
                                                        dataset=1, 
                                                        random_state=random_state, # random state for the noise in Y
                                                        whiten=False, 
                                                        K = 2, # number of clusters
                                                        loss = model[1], # loss function
                                                        parameter=parameter, # parameter to be varied
                                                        parameter_value=value, # value of the parameter
                                                        lambda_values=0.01, # lambda value for the model
                                                        save_local=True, # save the results locally
                                                        file_dir=path,
                                                        N = 200, D = 3)
                
print('Running experiments for RQ2 (K = 3): Performance analysis on synthetic datasets - Level of Guassian Noise in Target Variable')

models = [
['MIP','MSE','lpc_ns_mip'],
['QPBO','QPBO','lpc_ns_qpbo'],
['GlobalOpt','MSE','globalopt'],
]


print('Running experiments for RQ2 (K = 3): Performance analysis on synthetic datasets - Level of Guassian Noise in Target Variable')

n_samples = int(120/3)
d = 3
np.random.seed(2)
x1 = np.random.normal([1]*d, [1]*d, size=(n_samples, d))
x2 = np.random.normal([1]*d, [1]*d, size=(n_samples, d))
x3 = np.random.normal([1]*d, [1]*d, size=(n_samples, d))

x = np.concatenate([x1, x2, x3])
z = np.concatenate([[0]*n_samples, [1]*n_samples, [2]*n_samples])

reg_coef = SelfDefinedSlope([np.random.uniform(-1,-2,d), np.random.uniform(2,1,d), 
                        np.array([0.1]*d), np.random.uniform(-3,3,d)],
                        [np.array([-10]),np.array([1]),np.array([10]),np.array([5])])

for random_state in [42, 12, 3, 4, 2, 5, 6, 7, 8, 9]:
    for value in [0.6, 1.1, 1.7, 2.3, 2.9]:
            for model in models:
                print(f'Running {model[0]}')
                synthetic = SyntheticData(random_state=random_state)
                synthetic.set_generator(reg_coef)
                synthetic.set_generator(SelfDefinedX(x, z)) 

                path = os.path.join('experiments/results', model[-1]) # path to save the results

                if model[0] == 'GlobalOpt':
                    model_class = GobalOpt(verbose =True) # Initialize the model class
                elif model[0] == 'MIP':
                    model_class = LpcNsMip(verbose =True) # LPC-NS-MIP
                else:
                    model_class = LpcNsQbpo(verbose =True) # LPC-NS-QBPO
                
                evaluation = Evaluation.evaluate_model(synthetic,model_class,
                                                        verbose = True, validate=True,
                                                        dataset=1, 
                                                        random_state=random_state, # random state for the noise in Y
                                                        whiten=False, 
                                                        K = 3, # number of clusters
                                                        loss = model[1], # loss function
                                                        parameter=parameter, # parameter to be varied
                                                        parameter_value=value, # value of the parameter
                                                        lambda_values=0.01, # lambda value for the model
                                                        save_local=True, # save the results locally
                                                        file_dir=path,
                                                        N = 120, D = 3)