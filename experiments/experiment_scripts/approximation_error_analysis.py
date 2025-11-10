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



models = [
['MIP','MSE','lpc_ns_mip'],  # model name, loss function, path to save the results
['QPBO','QPBO','lpc_ns_qpbo'],
['GlobalOpt','MSE','globalopt'],
]

for random_state in [42, 12, 3, 4, 2, 5, 6, 7, 8, 9]:
    count = 0
    for value in [1,2,3,4]:
        np.random.seed(2)
        n_samples = int(70/2)
        if count == 0:
            x1 = np.random.normal((1, 1), (1, 1), size=(n_samples ,2))
            x2 = x1
        elif count == 3:
            x1 = np.random.normal((1, 1), (0.7, 0.5), size=(n_samples ,2))
            x2 = np.random.normal((3.2, 3.2), (0.5, 0.7), size=(n_samples ,2))
        elif count == 2:
            x1 = np.random.normal((1, 1), (0.5, 0.5), size=(n_samples ,2))
            x2 = np.random.normal((1, 1), (2, 2), size=(n_samples ,2))
        elif count == 1:
            x1 = np.random.normal((1, 1), (2, 2), size=(n_samples ,2))
            x2 = np.random.normal((1, 1), (2, 2), size=(n_samples ,2))

        z = np.concatenate([[0]*n_samples, [1]*n_samples])
        x = np.concatenate([x1, x2])

        # calculate the XTX - X2TX - X1TX
        xt = (x.T @ x)/2
        x1t = x1.T @ x1
        x2t = x2.T @ x2
        xtx = np.sum([np.linalg.norm(xt - x2t, ord=2), np.linalg.norm(xt - x1t, ord=2)]).round(2)

        count+=1    
        for model in models:
            path = os.path.join('experiments/results', model[-1])

            synthetic = SyntheticData(random_state=random_state)
            i = SelfDefinedSlope([np.array([2, 4]), 
            np.array([-2, -4])], [np.array([-5]),np.array([+5])])
            synthetic.set_generator(i)
            synthetic.set_generator(SelfDefinedX(x, z)) 
            
            if model[0] == 'GlobalOpt':
                model_class = GobalOpt(verbose =True) # Initialize the model class
            elif model[0] == 'MIP':
                model_class = LpcNsMip(verbose =True) # LPC-NS-MIP
            else:
                model_class = LpcNsQbpo(verbose =True) # LPC-NS-QBPO

                evaluation = Evaluation.evaluate_model(synthetic,
                                                model_class,
                                                verbose = True, validate=False,
                                                dataset=1,
                                                random_state=random_state, 
                                                whiten=False,
                                                K = 2,
                                                loss = model[1],
                                                parameter='XTX', 
                                                parameter_value=xtx,
                                                lambda_values=0.01, 
                                                save_local=True,
                                                file_dir=path,
                                                N = 70, D = 2, 
                                                noise = 2,
                                                CLR=True) # enble Clusterwise Linear Regression for comparison
