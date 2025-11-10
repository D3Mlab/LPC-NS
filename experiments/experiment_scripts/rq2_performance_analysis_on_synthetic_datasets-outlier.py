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
import os

'''
This script is used to run the experiments for RQ2: Performance analysis on synthetic datasets - Outlier Ratio.

Notes: This script requires gurobi to be installed with valid license to run the MIP model.

'''

parameter = 'outlier_ratio' # parameter to vary

models = [
['MIP','MSE','lpc_ns_mip'],  # model name, loss function, path to save the results
['QPBO','QPBO','lpc_ns_qpbo'],
['GlobalOpt','MSE','globalopt'],
]

random_states = [42, 12, 3, 4, 2, 5, 6, 7, 8, 9]

print('Running experiments for RQ2 (K = 2): Performance analysis on synthetic datasets - Outlier Ratio')
for random_state in random_states: # random state to generate the synthetic data with different noise in Y
    for value in [0.1, 0.2, 0.3, 0.4, 0.5]: # different outlier ratio
            for model in models: # different models
                print(f'Running {model[0]} model with {model[1]} loss function for outlier_ratio = {value} and random_state = {random_state}')
                n_samples = int(200/2)

                np.random.seed(2)
                # To ensure non-separable data in feature space, we generate data with same mean and variance from gaussian distribution
                x1 = np.random.normal([1]*3, [2]*3, size=(n_samples ,3))
                x2 = np.random.normal([1]*3, [2]*3, size=(n_samples ,3))

                x = np.concatenate([x1, x2])
                z = np.concatenate([[0]*n_samples, [1]*n_samples])
                
                synthetic = SyntheticData(random_state=random_state)
                # regression coefficients
                reg_coef =  SelfDefinedSlope([np.array([2, 4, 3])/10, 
                    np.array([-2, -4, -3])/10], [np.array([-10]),np.array([10])])
                synthetic.set_generator(reg_coef)
                synthetic.set_generator(SelfDefinedX(x, z)) 
                # save the results
                path = os.path.join('experiments/results', model[-1])

                if model[0] == 'GlobalOpt':
                    model_class = GobalOpt(verbose =True) # Initialize the model class
                elif model[0] == 'MIP':
                    model_class = LpcNsMip(verbose =True) # LPC-NS-MIP
                else:
                    model_class = LpcNsQbpo(verbose =True) # LPC-NS-QBPO

                # Evaluate the model
                evaluation = Evaluation.evaluate_model(synthetic,
                                                    model_class,
                                                    verbose = True, K = 2, 
                                                    validate=True, # 40% OOD sample
                                                    lambda_values=0.01, 
                                                    dataset=1, 
                                                    parameter_value=value, 
                                                    loss = model[1],
                                                    whiten = False,
                                                    random_state=random_state,
                                                    parameter=parameter, 
                                                    save_local=True,
                                                    show_plot=False,
                                                    file_dir= path,
                                                    N = 200, # number of samples
                                                    noise_std=0.3, # noise in Y
                                                    D = 3) # dimension of X
                
                
print('Running experiments for RQ2 (K = 3): Performance analysis on synthetic datasets - Outlier Ratio')
parameter = 'outlier_ratio'
models = [
['MIP','MSE','lpc_ns_mip'], # model name, loss function, path to save the results
['QPBO','QPBO','lpc_ns_qpbo'],
['GlobalOpt','MSE','globalopt'],
]

for random_state in random_states:
    for value in [0.1, 0.2, 0.3, 0.4, 0.5]:
            for model in models:
                print(f'Running {model[0]}')
                synthetic = SyntheticData(random_state=random_state)
                n_samples = int(120/3)
                d = 3
                
                x1 = np.random.normal([1]*d, [2]*d, size=(n_samples ,d ))
                x2 = np.random.normal([1]*d, [2]*d, size=(n_samples ,d ))
                x3 = np.random.normal([1]*d, [2]*d, size=(n_samples ,d ))
                
                x = np.concatenate([x1, x2, x3])
                z = np.concatenate([[0]*n_samples, [1]*n_samples, [2]*n_samples])

                i = SelfDefinedSlope([np.random.uniform(-1,-2,d), np.random.uniform(2,1,d), 
                                      np.array([0.1]*d), np.random.uniform(-3,3,d)],
                                        [np.array([-10]),np.array([1]),np.array([10]),np.array([5])])
                synthetic.set_generator(i)
                synthetic.set_generator(SelfDefinedX(x, z))

                path = os.path.join('experiments/results', model[-1])

                if model[0] == 'GlobalOpt':
                    model_class = GobalOpt(verbose =True) # Initialize the model class
                elif model[0] == 'MIP':
                    model_class = LpcNsMip(verbose =True) # LPC-NS-MIP
                else:
                    model_class = LpcNsQbpo(verbose =True) # LPC-NS-QBPO
                
                evaluation = Evaluation.evaluate_model(synthetic,
                                                        model_class,
                                                        verbose = True, K = 3, 
                                                        validate=False,
                                                        lambda_values=0.01, 
                                                        dataset=1, 
                                                        parameter_value=value,
                                                        loss = model[1],
                                                        whiten = False,
                                                        random_state=random_state,
                                                        parameter=parameter, 
                                                        save_local=True,
                                                        show_plot=False,
                                                        file_dir= path,
                                                        N = 42, noise_std=2.5, 
                                                        D = 3)