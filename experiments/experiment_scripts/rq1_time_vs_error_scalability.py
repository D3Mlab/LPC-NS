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
This script is used to run the experiments for RQ1: How do objective error and optimization times change as the number of samples increases?

Notes: This script requires gurobi to be installed with valid license to run the MIP model.

'''

print('Running RQ1: How do objective error and optimization times change as the number of samples increases?')
#How do objective error and optimization times change as the number of samples increases? 
parameter = 'N' # parameter to change
print('Running RQ1 with K = 2')

models = [
['QPBO','QPBO','lpc_ns_qbpo'], # model name, loss function, path to save the results
['MIP','MSE','lpc_ns_mip'], 
['GlobalOpt','MSE','globalopt'],
]

random_states = [42, 12, 3, 4, 2, 5, 6, 7, 8, 9]

for random_state in random_states: # random states for different experiments trials
    for value in [50, 100, 150, 200]: # sample sizes
        for model in models: 
            print(f'===== Running RQ1 with {model[0]} with {value} samples in the random state {random_state} =====')
            n_samples = int(value/2) # number of samples per clusters
            synthetic = SyntheticData(random_state=random_state) # Initialize the synthetic data class
            np.random.seed(2) # This random seed is fixed to ensure X is the same across different N
            
            0
            x1 = np.random.normal([1]*2, [2]*2, size=(n_samples ,2)) # cluster 1 samples
            x2 = np.random.normal([1]*2, [2]*2, size=(n_samples ,2)) # cluster 2 samples
            x = np.concatenate([x1, x2]) # combine the data
            z = np.concatenate([[0]*n_samples, [1]*n_samples]) # cluster labels

            # Define the regression coefficients
            reg_coef = SelfDefinedSlope([np.array([0.2, 0.4]), np.array([-0.2, -0.4])], 
                                        [np.array([-10]),np.array([10])])
            synthetic.set_generator(reg_coef)
            synthetic.set_generator(SelfDefinedX(x, z)) 

            path = os.path.join('experiments/results', model[-1]) # path to save the results

            if model[0] == 'GlobalOpt':
                model_class = GobalOpt(verbose =True) # Initialize the model class
            elif model[0] == 'MIP':
                model_class = LpcNsMip(verbose =True) # LPC-NS-MIP
            else:
                model_class = LpcNsQbpo(verbose =True) # LPC-NS-QBPO

                evaluation = Evaluation.evaluate_model(synthetic,
                                                        model_class,
                                                        verbose =True,
                                                        dataset=1,
                                                        random_state=int(random_state), 
                                                        whiten=False,
                                                        K = 2, # number of clusters
                                                        loss = model[1], # loss function
                                                        parameter=parameter, # N
                                                        parameter_value=value, # N values
                                                        lambda_values=1, # lambda values
                                                        save_local=True, # save the results
                                                        file_dir=path, # path to save the results
                                                        D = 2, # dimension of the data
                                                        noise_std = 3.5) # noise_std = 3.5
                
# Additional Samples for LPC-NS-QBPO since GlobalOpt reaches the time limit (2hrs) at 200 samples

print('Running RQ1 with K = 2 with additional samples for LPC-NS-QBPO and LPC-NS-MIP')

models = [
['QPBO','QPBO','lpc_ns_qbpo'],
['MIP','MSE','lpc_ns_mip'],
]
for random_state in random_states:
    for value in [400, 600, 800, 1000, 1500, 2000]:
                for model in models:
                    print(f'===== Running RQ1 with {model[0]} with {value} samples in the random state {random_state} =====')
                    n_samples = int(value/2)
                    synthetic = SyntheticData(random_state=random_state)
                    np.random.seed(2) # This random seed is fixed to ensure X is the same across different N
                    x1 = np.random.normal([1]*2, [2]*2, size=(n_samples ,2)) # cluster 1
                    x2 = np.random.normal([1]*2, [2]*2, size=(n_samples ,2)) # cluster 2
                    x = np.concatenate([x1, x2]) # combine the data
                    z = np.concatenate([[0]*n_samples, [1]*n_samples]) # cluster labels

                    reg_coef = SelfDefinedSlope([np.array([0.2, 0.4]), np.array([-0.2, -0.4])], 
                                                [np.array([-10]),np.array([10])])
                    synthetic.set_generator(reg_coef)
                    synthetic.set_generator(SelfDefinedX(x, z)) 

                    path = os.path.join('experiments/results', model[-1]) # path to save the results

                    if model[0] == 'GlobalOpt':
                        model_class = GobalOpt(verbose =True) # GlobalOpt 
                    elif model[0] == 'MIP':
                        model_class = LpcNsMip(verbose =True) # MIP
                    else:
                        model_class = LpcNsQbpo(verbose =True)

                        evaluation = Evaluation.evaluate_model(synthetic,
                                                    model_class,
                                                    verbose =True,
                                                    dataset=1,
                                                    random_state=int(random_state), 
                                                    whiten=False,
                                                    K = 2,
                                                    loss = model[1], # loss function
                                                    parameter=parameter, # N
                                                    parameter_value=value, # N values
                                                    lambda_values=1, # lambda values
                                                    save_local=True, # save the results
                                                    file_dir=path, # path to save the results
                                                    D = 2, # dimension of the data
                                                    noise_std = 3.5) # noise_std = 3.5
                
print('Running RQ1 with K = 3')
parameter = 'N'
models = [
['QPBO','QPBO','lpc_ns_qpbo'],
['MIP','MSE','lpc_ns_mip'],
['GlobalOpt','MSE','global_opt'],
]
for random_state in [42, 12, 3, 4, 2, 5, 6, 7, 8, 9]:
    for value in [45, 60, 75, 90]:
        for model in models:
            print(f'Running RQ1 K = 3 with {model[0]} with {value} samples in the random state {random_state}')
            np.random.seed(2)
            n_samples = int(value/2)
            d = 2
            synthetic = SyntheticData(random_state=random_state)
            n_samples = int(value/3)
            x1 = np.random.normal([1]*d, [2]*d, size=(n_samples ,d ))
            x2 = np.random.normal([1]*d, [2]*d, size=(n_samples ,d ))
            x3 = np.random.normal([1]*d, [2]*d, size=(n_samples ,d ))
            np.random.seed(2)
            x = np.concatenate([x1, x2, x3])
            z = np.concatenate([[0]*n_samples, [1]*n_samples, [2]*n_samples])

            i = SelfDefinedSlope([np.random.uniform(-1,-2,d),
                                   np.random.uniform(2,1,d),
                                     np.array([0.1]*d)],
                                    [np.array([-10]),np.array([1]), 
                                     np.array([10])])
            synthetic.set_generator(i)
            synthetic.set_generator(SelfDefinedX(x, z))

            path = os.path.join('experiments/results', model[-1])
            
            if model[0] == 'GlobalOpt':
                model_class = GobalOpt(verbose =True) # GlobalOpt 
            elif model[0] == 'MIP':
                model_class = LpcNsMip(verbose =True) # MIP
            else:
                model_class = LpcNsQbpo(verbose =True)

            evaluation = Evaluation.evaluate_model(synthetic,
                                                        model_class,
                                                        verbose =True,
                                                        dataset=1,
                                                        random_state=random_state, 
                                                        whiten=False,
                                                        K = 3,
                                                        loss = model[1],
                                                        parameter=parameter, 
                                                        parameter_value=value,
                                                        lambda_values=0.01, 
                                                        save_local=True,
                                                        file_dir=path, D = d, 
                                                        noise_std = 3.5)
print('Running RQ1 with different K values')
parameter = 'K'

models = [
['QPBO','QPBO','lpc_ns_qbpo'],
 ['MSE','MSE','lpc_ns_mip'],
['GlobalOpt','MSE','globalopt'],
]

for random_state in random_states:
    for value in [2, 3, 4]:
        for model in models:
            print(f'Running {model[0]}')
            np.random.seed(2)
            d = 2
            synthetic = SyntheticData(random_state=random_state)
            n_samples = int(36/value)
            x = []
            for k in range(value):
                x.append(np.random.normal([1]*d, [2]*d, size=(n_samples ,d  )))
            x = np.concatenate(x)
            z = np.concatenate([[i]*n_samples for i in range(value)])

            i = SelfDefinedSlope([np.random.uniform(-1,-2,d), 
                                  np.random.uniform(2,1,d),
                                    np.array([0.1]*d),
                                    np.random.uniform(-3,-4,d),
                                    np.random.uniform(3,4,d)],
                                    [np.array([-10]),np.array([1]),
                                     np.array([10]),np.array([5]),np.array([-5])])
            
            synthetic.set_generator(i)
            synthetic.set_generator(SelfDefinedX(x, z))

            path = os.path.join('experiments/results', model[-1])

            if model[0] == 'GlobalOpt':
                model_class = GobalOpt(verbose =True) # GlobalOpt 
            elif model[0] == 'MIP':
                model_class = LpcNsMip(verbose =True) # MIP
            else:
                model_class = LpcNsQbpo(verbose =True)

            evaluation = Evaluation.evaluate_model(synthetic,
                                                    model_class,
                                                    verbose =True,
                                                    dataset=1,
                                                    random_state=random_state, 
                                                    whiten=False,
                                                    loss = model[1],
                                                    parameter=parameter, 
                                                    parameter_value=value,
                                                    lambda_values=0.01, 
                                                    save_local=True,
                                                    file_dir=path, D = d, 
                                                    noise_std = 3.5, N = 36)