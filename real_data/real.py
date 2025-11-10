from synthetic_data.data import Data
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import f_regression
from ucimlrepo import fetch_ucirepo 
import json
import os

class RealData(Data):

    '''
    Class to load real data, data cleaning from UCI repository, and cluster analysis
    
    '''

    def load_data(self, ID, feature, random_state=42, data_path=None):
        """
        Load real data from local files

        Parameters
        ----------
        ID : int
            ID of the dataset
        feature : str
            categorical feature that used for data partition
        random_state : int
            random state
        data_path : str
            path to the data
        """
        df = pd.read_csv(os.path.join(data_path, f'Data_{ID}_{feature}.csv'))
        data_json = json.load(open(os.path.join(data_path,f'Data_{ID}_{feature}.json')))
        self.X = np.array(data_json['X'])
        self.z = np.array(data_json['z'], dtype=int)
        self.N, self.D = self.X.shape
        self.y = np.array(data_json['y'])
        self.train = df.index.tolist()
        self.K = len(np.unique(self.z))
        regression_weight = []
        for i in range(len(np.unique(self.z))):
            regression_weight.append({'bias': [data_json['weights'][str(i)][0]],
                                       'weights': data_json['weights'][str(i)][1:]}) 
        self.cluster_params = regression_weight    
        self.D = self.X.shape[1]
        self.N = self.X.shape[0]
        self.feature_name = df.columns.tolist()[:-1]
        self.data_dict = {}
        self.random = np.random.RandomState(random_state)

    def random_subset(self, N):
        """
        Randomly select a subset of the data

        Parameters
        ----------
        N : int
            number of data points
        """
        cluster_sizes = int(N / self.K)
        idx_subset = np.array([])
        for k in range(self.K):
            idx = np.where(self.z == k)[0]
            if len(idx) < cluster_sizes:
                idx_subset = np.hstack((idx_subset, idx))
            else:
                idx_subset = np.hstack((idx_subset, self.random.choice(idx, cluster_sizes, replace=False)))
                
        idx_subset = idx_subset.astype(int).flatten()
        #shuffle the data
        self.random.shuffle(idx_subset)
        self.X = self.X[idx_subset, :]
        self.z = self.z[idx_subset]
        self.y = self.y[idx_subset]
        self.N = self.X.shape[0]
        self.train = np.arange(self.N)
    
    def get_data_from_uci(self, id, save_path):
        """
        Load real data from UCI repository

        Parameters
        ----------
        id : int
            ID of the dataset
        save_path : str
            path to save the data
        """

        data = fetch_ucirepo(id=id) 
        
        # data (as pandas dataframes) 
        if id == 390:
            X = data.data.features.iloc[63:, :6].reset_index(drop=True)
            df = data.data.original.iloc[63:, :].reset_index(drop=True)
            y = data.data.targets.iloc[63:, :].reset_index(drop=True)
        else:
            X = data.data.features
            df = data.data.original
            y = data.data.targets

        if id == 320:
            y = data.data.targets.iloc[:, 2].to_frame()

        if id == 368:
            y_coloim = y.columns[0]
            df = pd.concat([X, y], axis=1)
            df = df[df['Type'] != 'Video'].reset_index(drop=True)
            X = df.drop(columns=[y.columns[0]])
            y = df[y.columns[0]].to_frame()

        if id == 162:
            y_col = y.columns[0]
            df = pd.concat([X, y], axis=1)
            df = df[~df['month'].isin(['apr','jan','dec','may'])].reset_index(drop=True)
            X = df.drop(columns=[y.columns[0]])
            y = df[y.columns[0]].to_frame()


        # replace Yes and No into 1 and 0 in y
        y = y.replace({'Yes': 1, 'No': 0})
        #replace '%' in X
        X = X.replace({'%': ''}, regex=True)
        X = X.replace({'?': 0})
        y = y.replace({'%': ''}, regex=True)
        y = y.replace({'?': 0})

        #replace inf in X
        X = X.replace({np.inf: 0}, regex=True)
        y = y.replace({np.inf: 0}, regex=True)

        #replace NaN in X
        X = X.replace({np.nan: 0}, regex=True)
        y = y.replace({np.nan: 0}, regex=True)

        possible_label = np.array([])
        for i, row in data.variables.iterrows():
            if row['name'] in df.columns:
                if row['type'] == 'Categorical':
                    if row['name'] not in y.columns:
                        possible_label = np.hstack((possible_label, row['name']))
                if row['type'] == 'Binary':
                    if row['name'] not in y.columns:
                        possible_label = np.hstack((possible_label, row['name']))
                    if row['name'] in X.columns:
                        X[row['name']] = X[row['name']].replace({'Yes': 1, 'No': 0})
                if row['type'] == 'Continuous' or row['type'] == 'Integer':
                    if row['name'] in X.columns:
                        #X[row['name']].fillna(0, inplace=True)
                        X[row['name']] = X[row['name']].astype(float)
                    if row['name'] in y.columns:
                        y[row['name']] = y[row['name']].astype(float)
        
        label_dict = {}
        for label in possible_label:
            if not label in y.columns.values:
                label_lst = df[label].unique()
                indices_lst = []
                label_final_lst = []
                for l in label_lst:
                    ind = df[df[label] == l].index.tolist()
                    indices_lst.append(df[df[label] == l].index.tolist())
                    label_final_lst.append(l)
                label_lst = [str(i) + ':'+str(len(l)) for i, l in zip(label_final_lst, indices_lst)]
                if len(label_lst) > 1:
                    if [len(l) > 0 for l in indices_lst] == [True]*len(indices_lst):
                        label_dict[label] = (label_lst, indices_lst)
        
        if len(label_dict) == 0:
            print('No categorical or binary columns found')
        else:
            result = self._cluster_analysis(X, y.iloc[:, 0].to_frame(), label_dict, possible_label, id, save_path)
            self.data_dict[id] = result
    
    def _cluster_analysis(self, X, y, label_dict, possible_label, id, save_path):



        norm_dict = {}
        tem = [col_name for col_name in X.columns if X[col_name].dtype == 'object' and col_name not in y.columns]
        possible_label = np.hstack((tem, possible_label))
        for label in label_dict.keys():
            weights_lst = []
            mean_lst = []
            df_cleaned = pd.concat([X, y], axis=1)
            if label in df_cleaned.columns:
                df_cleaned = df_cleaned.drop(columns=[label])
            if  'Date' in df_cleaned.columns:
                df_cleaned = df_cleaned.drop(columns=['Date'])
            if 'Time' in df_cleaned.columns:
                df_cleaned = df_cleaned.drop(columns=['Time'])
            if 'date' in df_cleaned.columns:
                df_cleaned = df_cleaned.drop(columns=['date'])
            if 'dteday' in df_cleaned.columns:
                df_cleaned = df_cleaned.drop(columns=['dteday'])

            for other_label in possible_label:
                if other_label != label:
                    if other_label in df_cleaned.columns:
                        #one hot encoding
                        df_cleaned = pd.get_dummies(df_cleaned, columns=[other_label], drop_first=True, dtype=float)

            #impute each column with the mean of the column
            for col in df_cleaned.columns:
                if not np.any([i in col for i in possible_label]):
                    # numerical columns
                    df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mean())
                    df_cleaned[col] = (df_cleaned[col] - df_cleaned[col].mean())/df_cleaned[col].std()

            # fig, ax = plt.subplots(1, len(df_cleaned.columns)-1, figsize=((len(df_cleaned.columns)-1)*5, 5))
            
            #feature selection
            if len(df_cleaned.columns) > 15:
                X_select = df_cleaned.drop(columns=[y.columns[0]])
                y_select = df_cleaned[y.columns[0]]
                f_stats, p_values = f_regression(X = X_select, y = y_select)
                #select the top 15 features that p_values < 0.05 with the highest f_stats
                f_stats = np.argsort(f_stats)[::-1]
                top_15 = []
                for index in f_stats:
                    if p_values[index] < 0.05 and len(top_15) < 15:
                        top_15.append(index)

                df_cleaned = pd.concat([X_select.iloc[:, top_15], y_select], axis=1)
            indices = np.random.choice(len(df_cleaned), np.min([len(df_cleaned), 600]), replace=False)
            print(len(indices))
            cluster_label = np.zeros(len(y))
            for c in range(len(label_dict[label][1])):
                cluster_label[label_dict[label][1][c]] = c
            df_cleaned['Cluster'] = cluster_label
            df_cleaned = df_cleaned.iloc[indices, :].reset_index(drop=True)
            cluster_label = df_cleaned['Cluster']
            df_cleaned = df_cleaned.drop(columns=['Cluster'])
            cluster_pred = np.zeros(df_cleaned.shape[0])
            X_select = df_cleaned.drop(columns=[y.columns[0]]).values
            XTX =  X_select.T @ X_select / np.unique(cluster_label).shape[0]
            XTX_diff_lst = np.zeros(np.unique(cluster_label).shape[0])
            for i in range(np.unique(cluster_label).shape[0]):
                X_cluster = df_cleaned.iloc[np.where(cluster_label == i)[0]]
                if not X_cluster.shape[0] == 0:
                    X_cluster = X_cluster.drop(columns=[y.columns[0]])
                    y_cluster = df_cleaned.iloc[np.where(cluster_label == i)[0]]
                    y_cluster = y_cluster[y.columns[0]]
                    X_array = X_cluster.values
                    XTX_diff_lst[i] = (np.linalg.norm(XTX - X_array.T @ X_array, ord=2))/ X_array.shape[0]
                    mean_lst.append([y_cluster.mean()])
                    
                    # fit a linear regression model to the data
                    model = LinearRegression()
                    model.fit(X = X_cluster, y = y_cluster)

                    # get the model coefficients
                    coefficients = model.coef_
                    # get the model intercept
                    intercept = model.intercept_
                    # store the model coefficients and intercept
                    weights_lst.append((coefficients, intercept))
                    pred = model.predict(X_cluster)
                    cluster_pred[np.where(cluster_label == i)[0]] = pred

            #compare with normal linear regression
            X_current = df_cleaned.drop(columns=[y.columns[0]])
            y_current = df_cleaned[y.columns[0]]
            model = Ridge(alpha=0.5)
            model.fit(X_current, y_current)
            coefficients = model.coef_
            intercept = model.intercept_
            pred = model.predict(X_current)
            baseline_mse = mean_squared_error(y_current, pred)
            cluster_mse = mean_squared_error(y_current, cluster_pred)
            #l2 norm
            basline_norm = np.linalg.norm(np.hstack((intercept, coefficients)))
            #l2 norm for each cluster
            norm_lst = [np.linalg.norm(np.hstack((w[1], w[0]))) for w in weights_lst]
            #pairwise difference
            dist = np.zeros((len(norm_lst), len(norm_lst)))
            for i in range(len(norm_lst)):
                for j in range(len(norm_lst)):
                    w1 = weights_lst[i]
                    w2 = weights_lst[j]
                    difference = np.hstack((w1[1], w1[0])) - np.hstack((w2[1], w2[0]))
                    dist[i][j] = np.linalg.norm(difference)

            max_diff = np.max([i for i in np.triu(dist, k=1).flatten() if i != 0])
            max_index = np.where(dist == max_diff)[0][0], np.where(dist == max_diff)[1][0]
            indices_cluster_0 = np.where(cluster_label == max_index[0])[0]
            indices_cluster_1 = np.where(cluster_label == max_index[1])[0]
            print(label_dict[label][0])
            x1 = df_cleaned.iloc[indices_cluster_0].drop(columns=[y.columns[0]]).values
            x2 = df_cleaned.iloc[indices_cluster_1].drop(columns=[y.columns[0]]).values
            x_all =  df_cleaned.iloc[np.hstack((indices_cluster_0, indices_cluster_1)), :].drop(columns=[y.columns[0]]).values  
            XTX_diff = np.linalg.norm(x_all.T @ x_all/2 - x1.T @ x1, ord=2) + np.linalg.norm(x_all.T @ x_all / 2 - x2.T @ x2, ord=2) / x_all.shape[0]
            norm_dict[label] = (label_dict[label][0], 
                                label_dict[label][0][max_index[0]] + ' ' + label_dict[label][0][max_index[1]],
                                basline_norm, max_diff/(len(X_current.columns) + 1), 
                                pd.DataFrame({label + ' ' + i:np.hstack((w[1], w[0])) for i, w in zip(label_dict[label][0], weights_lst) if i in [label_dict[label][0][max_index[0]], label_dict[label][0][max_index[1]]]},
                                index = np.hstack(('intercept', X_current.columns))), 
                                (baseline_mse, cluster_mse),
                                XTX_diff 
            )

            df_cleaned = pd.concat([df_cleaned.drop(columns=[y.columns[0]]), df_cleaned[y.columns[0]]], axis=1)
            #add cluster label
            df_cleaned['Cluster'] = cluster_label
            #seclect row only in the two clusters
            print('before cleaning', len(df_cleaned))
            df_cleaned = df_cleaned.iloc[np.hstack((indices_cluster_0, indices_cluster_1)), :].reset_index(drop=True)
            print(np.unique(df_cleaned['Cluster'], return_counts=True))
            cluster_label = df_cleaned['Cluster']
            df_cleaned = df_cleaned.drop(columns=['Cluster'])
            print('after cleaning', len(df_cleaned))
            #cluster mapper
            cluster_mapper = {max_index[0]: 0, max_index[1]: 1}
            cluster_label = np.array([cluster_mapper[i] for i in cluster_label])
            
            file_dir = save_path
            os.makedirs(file_dir, exist_ok=True)
            df_cleaned.to_csv(f'{file_dir}/Data_{id}_{label}.csv', index=False)
            cluster_label.dump(f'{file_dir}/Cluster_{id}_{label}.npy')
            X_current = df_cleaned.drop(columns=[y.columns[0]])
            y_current = df_cleaned[y.columns[0]]
            print(len(cluster_label))
            data_dict = {'X': X_current.values.tolist(), 'y': y_current.values.tolist(), 'z': cluster_label.tolist(),
                        'weights': {cluster_mapper[i[0]]:np.hstack((w[1], w[0])).tolist() for i, w in zip(enumerate(label_dict[label][0]), weights_lst) if i[0] in [max_index[0], max_index[1]]},
                        'XTX': XTX_diff}
            json.dump(data_dict, open(f'{file_dir}/Data_{id}_{label}.json', 'w'))
        #sort by min_diff
        norm_dict = dict(sorted(norm_dict.items(), key=lambda item: item[1][3], reverse=True))
        return norm_dict

    def print_data_information(self):
        best_diffs = {}
        all_diffs = []
        for key, item in self.data_dict.items():
            print('Dataset: ', key)
            best_diff = 1e-10
            best_k = 0
            for k, v in item.items():
                print('Cluster: ', k)
                print('Max Average Diff: ', v[3], 'Max Average Differene Categories: ', v[1])
                print('Baseline MSE: ', v[5][0], 'Cluster MSE: ', v[5][1])
                #print('Max Feature Diff: ', v[6][0], 'Max Featur Diff Categories: ', v[6][1], 'Feature: ', v[6][2])
                print('Weights: ', v[4])
                print('XTX: ', v[6])
                print('---------------------------------')
                
                if v[3] > best_diff:
                    best_diff = v[3]
                    best_k = k
                all_diffs.append((key, k))
            best_diffs[key] = (best_diff, best_k)

        data_list = []
        all_diff = dict(sorted(best_diffs.items(), key=lambda item: item[1][0], reverse=False))
        for key, value in all_diff.items():
            print('Dataset: ', key)
            print('Best Cluster: ', value[1])
            print('Best Average Diff: ', value[0])
            print('---------------------------------')
            data_list.append((key, value[1]))

    def save_data(self, file_path):
        """
        Save the data to the file

        Parameters
        ----------
        file_path : str
            path to save the data
        """
        json.dump(self.X.tolist(), open(f'{file_path}/X.json', 'w'))