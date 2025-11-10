import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from real_data.real import RealData
from evaluation.eval import Evaluation
from scripts.model import LpcNsMip, LpcNsQbpo
import numpy as np
import json

dataset_list = [
(368, 'Type'),
(519, 'sex'), 
(60, 'selector'),
(1, 'Sex'), 
(87, 'motor'),
(162, 'month'),
(275, 'hr'), 
(320, 'Fjob'),
(597, 'day'),
(390, 'period'),
(925, 'Age'),
(189, 'sex'), 
(89, 'largest spot size')
]

class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NpEncoder, self).default(obj)
        
file_path = 'experiments/results/real_data' # path to save results
for dataset in dataset_list:
    for lambda_reg in [10, 0.1, 0.01, 0.001, 0.0001]: # regularization parameter
        real = RealData() #load data
        print(f'==================={dataset}===================')
        real.load_data(ID=dataset[0], feature=dataset[1], data_path='real_data/cleaned_data_collection')
        
        print(f'sample size: {real.N}, dimension: {real.X.shape[1]}')

        # Greedy
        print('===================Greedy===================')
        model = LpcNsQbpo(verbose= True, loss='QPBO', lambda_reg=lambda_reg)
        model.whiten = False 
        model.loss = 'QPBO'
        evaluation = Evaluation(real, model, verbose= True)
        evaluation.fit(use_greedy=True, verbose= True, use_milp=False, error = 'QPBO')
        evaluation.evaluate()
        dir = f'{file_path}/greedy/{dataset[0]}_{dataset[1]}_{lambda_reg}'
        os.makedirs(dir, exist_ok=True)
        json.dump(evaluation.evaluation_results, open(dir+"/evaluation_results.json", 'w'), cls=NpEncoder)
        json.dump(evaluation.regression_weights, open(dir+"/regression_weights.json", 'w'), cls=NpEncoder)
        json.dump(evaluation.cluster_assignments, open(dir+"/cluster_assignments.json", 'w'), cls=NpEncoder)
        
        # lPC-NS-QPBO
        print('===================lPC-NS-QPBO===================')
        model = LpcNsQbpo(verbose= True, lambda_reg=lambda_reg)
        model.whiten = False
        evaluation = Evaluation(real, model, verbose= True)
        evaluation.fit(use_greedy=False, verbose= True, error = 'QPBO')
        evaluation.evaluate()
        dir = f'{file_path}/lpc_ns_qpbo/{dataset[0]}_{dataset[1]}_{lambda_reg}'
        os.makedirs(dir, exist_ok=True)
        json.dump(evaluation.evaluation_results, open(dir+"/evaluation_results.json", 'w'), cls=NpEncoder)
        json.dump(evaluation.regression_weights, open(dir+"/regression_weights.json", 'w'), cls=NpEncoder)
        json.dump(evaluation.cluster_assignments, open(dir+"/cluster_assignments.json", 'w'), cls=NpEncoder)
        
        # lPC-NS-MIP
        print('===================lPC-NS-MIP===================')
        model = LpcNsMip(verbose= True, lambda_reg=lambda_reg) 
        model.whiten = False
        evaluation = Evaluation(real, model, verbose= True)
        evaluation.fit(use_greedy=False, verbose= True, error = 'MSE')
        evaluation.evaluate()
        dir = f'{file_path}/lpc_ns_mip/{dataset[0]}_{dataset[1]}_{lambda_reg}'
        os.makedirs(dir, exist_ok=True)
        json.dump(evaluation.evaluation_results, open(dir+"/evaluation_results.json", 'w'), cls=NpEncoder)
        json.dump(evaluation.regression_weights, open(dir+"/regression_weights.json", 'w'), cls=NpEncoder)
        json.dump(evaluation.cluster_assignments, open(dir+"/cluster_assignments.json", 'w'), cls=NpEncoder)