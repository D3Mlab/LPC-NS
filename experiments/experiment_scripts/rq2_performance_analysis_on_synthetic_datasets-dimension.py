import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from  synthetic_data.data import SyntheticData
from  synthetic_data.cluster_size import *
from  synthetic_data.outlier import *
from  synthetic_data.cluster_label import *
from  synthetic_data.regression_params import *
from  synthetic_data.target import *

from evaluation.eval import Evaluation
from scripts.model import LpcNsMip, LpcNsQbpo, GobalOpt

import numpy as np

'''
This script is used to run the experiments for RQ2: Performance analysis on synthetic datasets - Number of Dimension Variables.

Notes: This script requires gurobi to be installed with valid license to run the MIP model.

'''

models = [
['MIP','MSE','lpc_ns_mip'], # model name, loss function, path to save the results
['QPBO','QPBO','lpc_ns_qpbo'],
['GlobalOpt','MSE','globalopt'],
]

parameter = 'D' 

random_states = [42, 12, 3, 4, 2, 5, 6, 7, 8, 9]
for random_state in random_states: # random state to generate the synthetic data with different noise in Y
    for value in [4, 8, 12, 16]: # different dimension of X 
            for model in models: # different models
                print(f'Running {model[0]} model with {model[1]} loss function for D = {value} and random_state = {random_state}')
                n_samples = int(100/2) # number of samples in each cluster
                D = value
                d = D

                # To ensure non-separable data in feature space, we generate data with same mean and variance from gaussian distribution
                np.random.seed(2) #fixed X but varying random noise in Y
                x1 = np.random.normal([1]*D, [2]*D, size=(n_samples ,D ))
                x2 = np.random.normal([1]*D, [2]*D, size=(n_samples ,D ))
                x = np.concatenate([x1, x2])
                z = np.concatenate([[0]*n_samples, [1]*n_samples])
                reg_coef = SelfDefinedSlope([np.random.uniform(-1,-2,d), np.random.uniform(2,1,d)],
                                        [np.array([10]),np.array([-10])])
                
                # Initialize the synthetic data class
                synthetic = SyntheticData(random_state=random_state)
                synthetic.set_generator(reg_coef)
                synthetic.set_generator(SelfDefinedX(x, z)) 
                path = os.path.join('experiments/results/', model[-1])

                # Initialize the model class
                if model[0] == 'GlobalOpt':
                    model_class = GobalOpt(verbose =True) # Initialize the model class
                elif model[0] == 'MIP':
                    model_class = LpcNsMip(verbose =True) # LPC-NS-MIP
                else:
                    model_class = LpcNsQbpo(verbose =True) # LPC-NS-QBPO

                evaluation = Evaluation.evaluate_model(synthetic,
                                                        model_class,
                                                        verbose = True, K = 2, 
                                                        validate=True,
                                                        lambda_values=0.01, 
                                                        dataset=1, 
                                                        parameter_value=value,
                                                        loss = model[1],
                                                        whiten = False,
                                                        random_state=random_state,
                                                        parameter='D', 
                                                        save_local=True,
                                                        show_plot=False,
                                                        file_dir= path,
                                                        N = 100, 
                                                        noise_std=0.5)

print('Done running the experiments for RQ2 (K = 2): Performance analysis on synthetic datasets - Number of Dimension Variables')

print('Running experiments for RQ2 (K = 3): Performance analysis on synthetic datasets - Number of Dimension Variables')
for random_state in random_states:
    for value in [2,4,6,8]:
            for model in models:
                print(f'Running {model[0]} model with {model[1]} loss function for D = {value} and random_state = {random_state}')
                d = value
                n_samples = int(42//3)

                np.random.seed(2) #fixed X but varying random noise in Y
                x1 = np.random.normal([1]*d, [2]*d, size=(n_samples ,d ))
                x2 = np.random.normal([1]*d, [2]*d, size=(n_samples ,d ))
                x3 = np.random.normal([1]*d, [2]*d, size=(n_samples ,d ))
                
                x = np.concatenate([x1, x2, x3])
                z = np.concatenate([[0]*n_samples, [1]*n_samples, [2]*n_samples])

                reg_coef = SelfDefinedSlope([np.random.uniform(-1,-2,d), np.random.uniform(2,1,d), 
                                      np.array([0.1]*d), np.random.uniform(-3,3,d)],
                                        [np.array([-10]),np.array([1]),np.array([10]),np.array([5])])
                
                synthetic = SyntheticData(random_state=random_state)
                synthetic.set_generator(reg_coef)
                synthetic.set_generator(SelfDefinedX(x, z))
                path = os.path.join('experiments/results/', model[-1])
                
                if model[0] == 'GlobalOpt':
                    model_class = GobalOpt(verbose =True) # Initialize the model class
                elif model[0] == 'MIP':
                    model_class = LpcNsMip(verbose =True) # LPC-NS-MIP
                else:
                    model_class = LpcNsQbpo(verbose =True) # LPC-NS-QBPO
                   

                evaluation = Evaluation.evaluate_model(synthetic,
                                                model_class,
                                                verbose = True, 
                                                K = 3, validate=False,
                                                lambda_values=1, 
                                                dataset=1, 
                                                parameter_value=value,
                                                loss = model[1],
                                                whiten = False,
                                                random_state=random_state,
                                                parameter='D', 
                                                save_local=True,
                                                show_plot=False,
                                                file_dir= path,
                                                N = 42, 
                                                noise_std=0.5)