import numpy as np
import gurobipy as gp
from gurobipy import GRB

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scripts.Greedy_Codes.model import *
from scripts.Greedy_Codes.utils import *

from pyscipopt import Model

class LinearPredictiveClustering(ABC):

    '''
    Parent class for LPC-NS-MIP, LPC-NS-QPBO, and GLobalOpt
    '''

    def __init__(self, n=50, p=1, K=2, lambda_reg=0.7, noise_std=1.0,
                  random_state=None, verbose=True, loss='MSE'):
        """
        Initializes the LinearPredictiveClustering with specified parameters.

        Parameters:
        - n: Total number of data points
        - p: Number of input features
        - K: Number of clusters
        - lambda_reg: Regularization parameter for Ridge Regression
        - noise_std: Standard deviation of Gaussian noise
        - random_state: Seed for reproducibility
        """
        self.n = n
        self.p = p
        self.K = K

        self.lambda_reg = lambda_reg
        self.noise_std = noise_std
        self.random_state = random_state

        self.loss = loss

        # Set random seed for reproducibility
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Initialize placeholders for data and parameters
        self.X = None
        self.y = None
        self.true_cluster_assignments = None
        self.cluster_params = None
        self.X_augmented = None
        self.U = None
        self.Sigma_values = None
        self.V = None
        self.XV = None
        self.A = None
        self.A_k = [None] * self.K
        self.Q = None

        self.cluster_assignments = None
        self.w_star_list = []
        self.w_star_original_k_np = None
        self.w_star = None
        self.mse = None
        self.ridge_model = None
        self.w_star_sklearn = None
        self.mse_sklearn = None
        self.w_star_refit = []
        self.verbose = verbose

        self.whiten = False

    def _print(self, text: str, **kwargs):
        '''
        an helper function to print and store the intermediate log if verbose is set to True
        '''
        if self.verbose:
            print(text,**kwargs)

    def _get_pca_components(self, X, n_components=2):
        """
        Helper function to compute PCA components of X.

        Parameters:
        - X: n x p array of input features
        - n_components: number of PCA components to compute

        Returns:
        - X_pca: n x n_components array
        """
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        return X_pca
    
    def prepare_augmented_data(self):
        """
        Adds a column of ones to X for the bias term and computes SVD and related matrices.
        """
        if self.X is None or self.y is None:
            raise ValueError("Data not generated yet. Call generate_synthetic_data() first.")

        self.X = self.X.round(2)
        self.y = self.y.round(2)

        # Add a column of ones to X for the bias term
        self.X_augmented = np.hstack((np.ones((self.n, 1)), self.X))  # n x (p+1)
        self.p_augmented = self.X_augmented.shape[1]  # p + 1

        # Step 1: Compute the SVD of X_augmented
        self.U, self.Sigma_values, self.Vt = np.linalg.svd(
            self.X_augmented, full_matrices=False
        )
        self.U_y, self.Sigma_values_y, self.Vt_y = np.linalg.svd(
            self.y[:, np.newaxis], full_matrices=False
        )
        self.V = self.Vt.T  # V is (p+1) x (p+1)
        self.V_y = self.Vt_y.T  # V_y is 1 x 1
        self.Sigma = np.diag(self.Sigma_values)
        self.Sigma_inv = np.linalg.inv(self.Sigma)
        # Step 2: Compute XV (whitened data)
        self.XV = self.X_augmented
        self.yV = self.y[:, np.newaxis]

        I = np.eye(self.p_augmented)
        self.A = np.linalg.inv((self.XV.T @ self.XV)/self.K + self.lambda_reg * I)  # (p+1) x (p+1)

        # Step 4: Compute Matrix Q
        A = np.linalg.inv((self.XV.T @ self.XV)/self.K + self.lambda_reg * I)  # (p+1) x (p+1)
        self.A_XVT = A @ self.XV.T  # (p+1) x n
        self.XV_A_XVt = self.XV @ A  @ self.XV.T # n x (p+1)
        self.XV_AA_XVt = self.XV @ A @ A @ self.XV.T # n x n
        self.Q =  self.yV * self.yV.T * self.XV_A_XVt 

        # Ensure Q is symmetric
        self.Q = (self.Q + self.Q.T) / 2

        #calculate y^Ty
        self.yTy = self.y[:, np.newaxis] * self.y[np.newaxis, :]

    @abstractmethod
    def perform_gurobi_miqp(self):
        """
        Solves the Cluster-wise Regression using Gurobi MIQP and retrieves cluster assignments.
        """
        pass
                        
    def compute_cluster_weights(self):
        """
        Computes the regression weights for each cluster based on cluster assignments.
        """
        if self.cluster_assignments is None:
            raise ValueError("Cluster assignments not available. Call perform_gurobi_miqp() first.")
        
        y_pred_star = np.zeros(self.n)  # Initialize predicted y values
        y_pred = np.zeros(self.n)  # Initialize predicted y values

        self._print("Cluster Assignments from Gurobi MIQP using {}".format(self.loss))
        #if self.loss == 'MSE':
        self.w_star_list = []  # Reset the list
        self.w_list = []
        self.A_k = []
        for k in range(self.K):
            # Create z_k vector
            z_k = (self.cluster_assignments == k).astype(float)  # n-dimensional vector
            self.A_k.append(np.linalg.inv(self.XV.T @ np.diag(z_k) @ self.XV + self.lambda_reg * np.eye(self.p + 1)))
            indices = np.where(self.cluster_assignments == k)[0]
            
            w_k =  np.linalg.inv(self.XV.T @ self.XV + self.lambda_reg * np.eye(self.p + 1)) @ self.XV.T @ np.diag(z_k) @ self.y # estimate the weights for each cluster
            w_star_k =  np.linalg.inv(self.XV.T @ np.diag(z_k) @ self.XV + self.lambda_reg * np.eye(self.p + 1)) @ self.XV.T @ np.diag(z_k) @ self.y # optimal weights for each cluster

            # Make predictions
            y_pred_star[indices] = self.X_augmented[indices] @ w_star_k
            y_pred[indices] = self.X_augmented[indices] @ w_star_k

            self.w_star_list.append(w_star_k)
            self.w_list.append(w_k)
            
        mse_star = np.mean((self.y - y_pred_star) ** 2)
        mse =  np.mean((self.y - y_pred) ** 2)
        
        r2_star = 1 - np.sum((self.y - y_pred_star) ** 2) / np.sum((self.y - np.mean(self.y)) ** 2)
        r2 = 1 - np.sum((self.y - y_pred) ** 2) / np.sum((self.y - np.mean(self.y)) ** 2)

        self.mse_milp_star = mse_star
        self.r2_milp_star = r2_star
        self.mse_milp = mse
        self.r2_milp = r2

    def perform_numpy_ridge_regression(self):
        """
        Performs Ridge Regression using NumPy and computes Mean Squared Error.
        """
        if self.X_augmented is None:
            raise ValueError("Data not prepared yet. Call prepare_augmented_data() first.")

        # Compute optimal weights w* using the closed-form solution
        self.w_star = self.A @ (self.XV.T @ self.y)  # (p+1)-dimensional vector [bias, weights]

        # Make predictions
        y_pred = self.XV @ self.w_star  # n-dimensional vector

        # Compute Mean Squared Error (MSE)
        self.mse = np.mean((self.y - y_pred) ** 2)

        self._print("\nRidge Regression using NumPy:")
        self._print("Optimal weights w* (including bias):")
        self._print(self.w_star)

        # Transform back to original space
        self.w_star_original_k_np = self.V @ self.w_star
        self._print(f"\nWeights (including bias) in original space:")
        self._print(self.w_star_original_k_np)

        self._print("\nMean Squared Error on training data:")
        self._print(self.mse)

    def perform_sklearn_ridge_regression(self):
        """
        Performs Ridge Regression using scikit-learn and computes Mean Squared Error.
        """

        if self.X is None or self.y is None:
            raise ValueError("Data not generated yet. Call generate_synthetic_data() first.")
        self.mse_sklearn = 1e10
        self.r2_sklearn = -1e10
        best_lambda = 0
        for lambda_reg in np.linspace(0.1, 1e5, 1000):
            self.ridge_model = Ridge(alpha=lambda_reg, fit_intercept=True)
            self.ridge_model.fit(self.X, self.y)  # Using original X (without the column of ones)
            self.w_star_sklearn = np.hstack((self.ridge_model.intercept_, self.ridge_model.coef_))  # [bias, weights]
            y_pred_sklearn = self.ridge_model.predict(self.X)
            mse = np.mean((self.y - y_pred_sklearn) ** 2)
            r2 = 1 - np.sum((self.y - y_pred_sklearn) ** 2) / np.sum((self.y - np.mean(self.y)) ** 2)
            
            if mse < self.mse_sklearn:
                self.mse_sklearn = mse
                best_lambda = lambda_reg
            if r2 > self.r2_sklearn:
                self.r2_sklearn = r2

        self._print("############################################################################")
        self._print("\nRidge Regression using scikit-learn:")
        self._print("Optimal weights w* in X Space (including bias):")
        self._print(self.w_star_sklearn)

        self._print(f"\nMean Squared Error on training data (scikit-learn) with lambda = {best_lambda}:")
        self._print(self.mse_sklearn)

    def perform_refit_sklearn_ridge_regression(self, ground_truth = False):
        """
        Performs Ridge Regression using scikit-learn to refit the regression model within each cluster using the cluster assignments from Gurobi MIQP.

        Parameters:
        if ground_truth is True, the cluster assignments will be based on the true cluster assignments
        
        """
        if self.X is None or self.y is None:
            raise ValueError("Data not generated yet. Call generate_synthetic_data() first.")
        
        if self.cluster_assignments is None and not ground_truth:
            raise ValueError("Cluster assignments not available. Call perform_gurobi_miqp() first.")
        
        if ground_truth:
            # refit the regression model using the true cluster assignments
            cluster_assignments = self.true_cluster_assignments
            K = np.max(cluster_assignments) + 1 # number of clusters in ground truth label
        else:
            # refit the regression model using the cluster assignments from Gurobi MIQP
            cluster_assignments = self.cluster_assignments
            K = self.K # number of clusters in the cluster assignments from Gurobi MIQP
        
        w_star_refit = [] # initialize the list to store the refit weights

        for k in range(K):
            # retrieve the data points for the cluster k
            indices = np.where(cluster_assignments == k)[0]
            if len(indices) > 0: # prevent empty clusters
                X_k = self.X[indices]
                y_k = self.y[indices]
                # Use Ridge Regression for MSE
                best_lambda = 0
                best_mse = 1e10
                for lambda_reg in np.linspace(1e-10, 1e5, 1000):
                    model_k = Ridge(alpha=lambda_reg, fit_intercept=True)
                    model_k.fit(X_k, y_k)  # Using original X (without the column of ones)
                    y_pred_k = model_k.predict(X_k)
                    mse = np.mean((y_k - y_pred_k) ** 2)
                    if mse < best_mse:
                        best_mse = mse
                        best_lambda =  lambda_reg
                    model_k = Ridge(alpha=best_lambda, fit_intercept=True)
                model_k.fit(X_k, y_k)  # Using original X (without the column of ones)
                w_star_refit.append(np.hstack((model_k.intercept_, model_k.coef_))) # [bias, weights]
            else:
                # if the cluster is empty, append zeros
                w_star_refit.append(np.zeros(self.p + 1))

        if not ground_truth: # store the refit results only if the cluster assignments are not based on the true cluster assignments
            self.w_star_refit = w_star_refit

        return w_star_refit


    def compute_mse_vs_components(self):
        """
        Computes MSE for different numbers of top components kept in the SVD and plots the results.
        """
        if self.X_augmented is None:
            raise ValueError("Data not prepared yet. Call prepare_augmented_data() first.")

        n, p_augmented = self.X_augmented.shape

        # Prepare lists to store results
        L_values = []
        mse_values = []

        # Test different numbers of top columns of V
        min_L = max(p_augmented - 4, 1)
        for L in range(p_augmented, min_L - 1, -1):  # Decrease L from p_augmented to min_L
            self._print(f"\nTesting with L = {L} components")

            # Extract the first L columns of V and Sigma_values
            V_L = self.V[:, :L]
            Sigma_values_L = self.Sigma_values[:L]

            # Compute XV_L (n x L)
            XV_L = self.X_augmented @ V_L

            # Compute A_L (L x L)
            Sigma_squared_L = Sigma_values_L ** 2
            A_L = np.diag(1 / (Sigma_squared_L + self.lambda_reg))

            # Compute w_star_L (L x 1)
            w_star_L = A_L @ (XV_L.T @ self.y)

            # Compute y_pred_L (n x 1)
            y_pred_L = XV_L @ w_star_L

            # Compute MSE_L
            mse_L = np.mean((self.y - y_pred_L) ** 2)
            self._print(f"MSE with L = {L}: {mse_L}")

            # Store results
            L_values.append(L)
            mse_values.append(mse_L)

        # Plot MSE vs L
        plt.figure(figsize=(8, 6))
        plt.plot(L_values, mse_values, marker='o')
        plt.xlabel('Number of Components L')
        plt.ylabel('Mean Squared Error (MSE)')
        plt.title('MSE vs Number of Components L')
        plt.gca().invert_xaxis()  # Optional, to show decreasing L from left to right
        plt.grid(True)
        plt.show()


    def run_all(self):
        """
        Executes all steps in sequence.
        """
        self._print("Generating synthetic data...")
        self.generate_synthetic_data()
        self.visualize_true_clusters()

        self._print("\nPreparing augmented data and computing matrices...")
        self.prepare_augmented_data()

        self._print("\nPerforming Gurobi MIQP for cluster-wise regression...")
        self.perform_gurobi_miqp()

        self._print("\nComputing cluster-specific regression weights...")
        self.compute_cluster_weights()
        self._print("\nPerforming Ridge Regression using scikit-learn...")
        self.perform_sklearn_ridge_regression()


class  LpcNsQbpo(LinearPredictiveClustering):

    '''

    Class for LPC-NS-QBPO Model

    '''

    def __init__(self, n=50, p=1, K=2, lambda_reg=0.7, noise_std=1.0, loss = 'QBPO', random_state=42, verbose=True):
        """
        Initializes the GlobalOpt with specified parameters.

        Parameters:
        - n: Total number of data points
        - p: Number of input features
        - self.K: Number of clusters
        - self.lambda_reg: Regularization parameter for Ridge Regression
        - noise_std: Standard deviation of Gaussian noise
        - random_state: Seed for reproducibility
        """

        super().__init__(n, p, K, lambda_reg, noise_std, loss=loss, random_state=random_state, verbose=verbose)


    def perform_gurobi_miqp(self):
        """
        Solves the Cluster-wise Regression using Gurobi MIQP and retrieves cluster assignments.
        """
        if self.Q is None:
            raise ValueError("Matrices not prepared yet. Call prepare_augmented_data() first.")

        model = gp.Model("ClusterWiseRegression")

        if not self.verbose:
            model.setParam('OutputFlag', 0)  # Suppress Gurobi output for cleaner output
        
        model.setParam('MIPGap', 0.05)  # Set MIP gap to 5% for faster computation
        model.setParam('TimeLimit', 3600)  # Set time limit to 1 hour

        z = model.addVars(self.n, self.K, vtype=GRB.BINARY, name="z")
        obj = gp.QuadExpr()
        for k in range(self.K):
            for i in range(self.n):
                for j in range(self.n):
                    obj += self.Q[i,j] * z[i,k] * z[j,k]

        model.setObjective(obj, GRB.MAXIMIZE)

        model.addConstrs((gp.quicksum(z[i, k] for k in range(self.K)) == 1 for i in range(self.n)),name="assignment")   
        model.optimize()
            
        if model.status != GRB.OPTIMAL:
            self._print("\nGurobi MIQP did not converge to an optimal solution.")
        else:
            self._print("\nGurobi MIQP converged to an optimal solution.")
        
        # Retrieve Cluster Assignments from Gurobi MIQP
        self.cluster_assignments = np.zeros(self.n, dtype=int)
        for i in range(self.n):
            for k in range(self.K):
                if z[i, k].X > 0.5:
                    self.cluster_assignments[i] = k
                    break

class LpcNsMip(LinearPredictiveClustering):

    '''

    Class for LPC-NS-MIP Model

    '''

    def __init__(self, n=50, p=1, K=2, lambda_reg=0.7, noise_std=1.0, loss = 'MSE', random_state=42, verbose=True):
        """
        Initializes the GlobalOpt with specified parameters.

        Parameters:
        - n: Total number of data points
        - p: Number of input features
        - self.K: Number of clusters
        - self.lambda_reg: Regularization parameter for Ridge Regression
        - noise_std: Standard deviation of Gaussian noise
        - random_state: Seed for reproducibility
        """

        super().__init__(n, p, K, lambda_reg, noise_std, loss=loss, random_state=random_state, verbose=verbose)

    def perform_gurobi_miqp(self):
        """
        Solves the LPC using Gurobi MIQP and retrieves cluster assignments.
        """
        if self.Q is None:
            raise ValueError("Matrices not prepared yet. Call prepare_augmented_data() first.")

        model = gp.Model("ClusterWiseRegression")

        if not self.verbose:
            model.setParam('OutputFlag', 0)  # Suppress Gurobi output for cleaner output
        
        model.setParam('MIPGap', 0.05)  # Set MIP gap to 5% for faster computation
        model.setParam('TimeLimit', 3600)  # Set time limit to 1 hour
        
        z = model.addVars(self.n, self.K, vtype=GRB.BINARY, name="z")
        residual = model.addVars(self.n, lb=-GRB.INFINITY, name="residual") 
        w_astar =  model.addVars(self.K, self.p + 1, name="w", lb=-GRB.INFINITY)
        for i in range(self.n):
            model.addConstr(gp.quicksum(z[i, k] for k in range(self.K)) == 1, name=f"cluster_assign_{i}")
        obj = gp.QuadExpr()
        for k in range(self.K):
            w = self.A @ self.XV.T @ np.diag([z[n, k] for n in range(self.n)]) @ self.y
            for j in range(self.p + 1):
                model.addConstr(w_astar[k, j] == w[j], name=f"w_{k}_{j}")
            res = self.y - (self.XV @ w)
            for i in range(self.n):
                model.addConstr((z[i, k] == 1) >> (residual[i] == res[i]), name=f"residual_{i}_{k}")
        obj += gp.quicksum(residual[i]**2 for i in range(self.n)) 
        + self.lambda_reg * gp.quicksum(w_astar[k, j]**2 for k in range(self.K) for j in range(self.p + 1))
        model.setObjective(obj, GRB.MINIMIZE)   
        model.optimize()
            
        if model.status != GRB.OPTIMAL:
            self._print("\nGurobi MIQP did not converge to an optimal solution.")
        else:
            self._print("\nGurobi MIQP converged to an optimal solution.")
        
        # Retrieve Cluster Assignments from Gurobi MIQP
        self.cluster_assignments = np.zeros(self.n, dtype=int)
        for i in range(self.n):
            for k in range(self.K):
                if z[i, k].X > 0.5:
                    self.cluster_assignments[i] = k
                    break
                        
class GobalOpt(LinearPredictiveClustering):

    '''

    Class for GlobalOpt Model

    '''

    def __init__(self, n=50, p=1, K=2, lambda_reg=0.7, noise_std=1.0, loss = 'MSE', 
                 random_state=42, verbose=True):
        """
        Initializes the GlobalOpt with specified parameters.

        Parameters:
        - n: Total number of data points
        - p: Number of input features
        - self.K: Number of clusters
        - self.lambda_reg: Regularization parameter for Ridge Regression
        - noise_std: Standard deviation of Gaussian noise
        - random_state: Seed for reproducibility
        """
        super().__init__(n, p, K, lambda_reg, noise_std, loss=loss, random_state=random_state, verbose=verbose)


    
    def perform_gurobi_miqp(self):
        """
        Solves the Cluster-wise Regression using Gurobi MIQP and retrieves cluster assignments.
        """
        if self.Q is None:
            raise ValueError("Matrices not prepared yet. Call prepare_augmented_data() first.")

        # Create the Gurobi model
        model = gp.Model("ClusterWiseRegression")
        if not self.verbose:
         model.setParam('OutputFlag', 0)  # Suppress Gurobi output for cleaner output

        z = model.addVars(self.n, self.K, vtype=GRB.BINARY, name="z")
        residual = model.addVars(self.n, lb=-GRB.INFINITY, name="residual")
        w = model.addVars(self.K, self.p + 1, name="w", lb=-GRB.INFINITY)
        #abs_w = model.addVars(self.K, self.p + 1, name="abs_w", lb=0)
        # Add cluster assignment constraints
        for i in range(self.n):
            model.addConstr(gp.quicksum(z[i, k] for k in range(self.K)) == 1)
        obj = gp.QuadExpr()
        # Residual constraints
        for k in range(self.K):
            for i in range(self.n):
                res = self.yV[i,0] - gp.quicksum(self.XV[i, j] * w[k, j] for j in range(self.p + 1))
                model.addGenConstrIndicator(z[i, k], True, residual[i] == res, name=f"residual_{i}_{k}")
                # model.addConstr(residual[i, k] == res[i]*z[i, k], name=f"residual_{i}_{k}")
        obj += gp.quicksum(residual[i]**2 for i in range(self.n)) + self.lambda_reg * gp.quicksum(w[k, j]**2 for k in range(self.K) for j in range(self.p + 1))
        model.setObjective(obj, GRB.MINIMIZE)   

        model.setParam('MIPGap', 0.05)  # Set MIP gap to 5% for faster computation
        model.setParam('TimeLimit', 7200)  # Set time limit to 1 hour
        model.optimize()
        
        # Retrieve Cluster Assignments from Gurobi MIQP
        self.cluster_assignments = np.zeros(self.n, dtype=int)
        if model.status == GRB.OPTIMAL:
            print("Optimal solution found.")
        else:
            print("Time limit reached. Suboptimal solution found.")

        for i in range(self.n):
            for k in range(self.K):
                if z[i, k].X > 0.5:
                    self.cluster_assignments[i] = k
                    break
        self.w_star_list = [np.zeros(self.p + 1) for _ in range(self.K)]
        for k in range(self.K):
            for j in range(self.p + 1):
                self.w_star_list[k][j] = w[k, j].X

        self._print(self.cluster_assignments)

    def compute_cluster_weights(self):
        """
        Computes the regression weights for each cluster based on cluster assignments.

        Since the weights are already computed during Gurobi MIQP, this function simply prints the weights.

        Note: Call perform_gurobi_miqp() first to retrieve cluster assignments.

        Returns:
        - None
        """
        if self.cluster_assignments is None:
            raise ValueError("Cluster assignments not available. Call perform_gurobi_miqp() first.")
        
        y_pred = np.zeros(self.n)  # Initialize predicted y values

        self._print("Cluster Assignments from Gurobi MIQP using {}".format(self.loss))

        for k in range(self.K):
            # Create z_k vector
            indices = np.where(self.cluster_assignments == k)[0]
            w_star_k = self.w_star_list[k]
            # Make predictions
            y_pred[indices] = self.X_augmented[indices] @ w_star_k

            self._print(f"\nCluster {k} weights w_{k}^* (including bias) in original space:")
            self._print( self.w_star_list[k])


# SCIP PBO solver class used for performance comparison between SCIP and Gurobi
class PBOClusterRegressionModel:
    def __init__(self, n=50, p=1, K=2, lambda_reg=0.7, noise_std=1.0, random_state=None, verbose=True):
        """
        Initializes the OPBClusterRegressionModel with specified parameters.

        Parameters:
        - n: Total number of data points
        - p: Number of input features
        - K: Number of clusters
        - lambda_reg: Regularization parameter for Ridge Regression
        - noise_std: Standard deviation of Gaussian noise
        - random_state: Seed for reproducibility
        - verbose: Verbosity of output
        """
        self.n = n
        self.p = p
        self.K = K
        self.lambda_reg = lambda_reg
        self.noise_std = noise_std
        self.random_state = random_state
        self.verbose = verbose

        # Set random seed for reproducibility
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Initialize placeholders for data and parameters
        self.X = None
        self.y = None
        self.true_cluster_assignments = None
        self.cluster_params = None
        self.X_augmented = None
        self.U = None
        self.Sigma_values = None
        self.V = None
        self.XV = None
        self.A = None
        self.Q = None

        self.cluster_assignments = None
        self.w_star_list = []
        self.mse_milp = None
        self.r2_milp = None

    def _print(self, text: str, **kwargs):
        if self.verbose:
            print(text, **kwargs)

    def prepare_augmented_data(self):
        """
        Adds a column of ones to X for the bias term and computes SVD and related matrices.
        """
        if self.X is None or self.y is None:
            raise ValueError("Data not generated yet. Call generate_synthetic_data() first.")

        # Add a column of ones to X for the bias term
        self.X_augmented = np.hstack((np.ones((self.n, 1)), self.X))  # n x (p+1)
        self.p_augmented = self.X_augmented.shape[1]  # p + 1

        # Step 1: Compute the SVD of X_augmented
        self.U, self.Sigma_values, Vt = np.linalg.svd(
            self.X_augmented, full_matrices=False
        )
        self.V = Vt.T  # V is (p+1) x (p+1)

        # Step 2: Compute XV (whitened data)
        self.XV = self.X_augmented @ self.V  # n x (p+1)

        # Step 3: Compute Matrix A
        Sigma_squared = self.Sigma_values ** 2  # (p+1)-dimensional vector
        self.A = np.diag(1 / (Sigma_squared + self.lambda_reg))  # (p+1) x (p+1)

        # Step 4: Compute Matrix Q
        XV_A = self.XV @ self.A  # n x (p+1)
        self.Q = self.y[:, np.newaxis] * self.y[np.newaxis, :] * (XV_A @ self.XV.T)  # n x n matrix

        # Ensure Q is symmetric
        self.Q = (self.Q + self.Q.T) / 2

    def generate_opb_file(self, output_filename='cluster_regression.opb'):
        """
        Generates an OPB file for the MIQP problem in minimization form.

        Parameters:
        - output_filename: Name of the output OPB file
        """
        n = self.n
        K = self.K
        Q = self.Q  # Ensure that self.Q is computed

        with open(output_filename, 'w') as f:
            f.write("* MIQP for Cluster-wise Linear Regression in OPB Format\n\n")

            # Objective Function (Convert maximization to minimization by negating the coefficients)
            f.write("min: \n")
            objective_terms = []
            for k in range(K):
                for i in range(n):
                    for j in range(i, n):  # To avoid duplicate terms
                        coef = -Q[i, j]  # Negate the coefficient for minimization
                        if abs(coef) > 1e-6:
                            var_i = f"z_{i}_{k}"
                            var_j = f"z_{j}_{k}"
                            if i == j:
                                # For terms like z_i_k * z_i_k (z_i_k squared, which is z_i_k for binary variables)
                                term = f"{coef:+} {var_i}"
                            else:
                                term = f"{coef:+} {var_i} {var_j}"
                            objective_terms.append(term)
            # Write the objective terms
            # Combine terms into lines of reasonable length
            max_line_length = 80
            current_line = ''
            for term in objective_terms:
                if len(current_line) + len(term) + 1 > max_line_length:
                    f.write(current_line + '\n')
                    current_line = term
                else:
                    if current_line == '':
                        current_line = term
                    else:
                        current_line += ' ' + term
            if current_line != '':
                f.write(current_line + '\n')
            f.write(";\n\n")

            # Assignment Constraints
            f.write("* Assignment Constraints\n")
            for i in range(n):
                vars_in_constraint = [f"+1 z_{i}_{k}" for k in range(K)]
                constraint = " ".join(vars_in_constraint) + " = 1;"
                f.write(constraint + "\n")

            # All variables are binary by default in OPB format


    def solve_opb_model(self, opb_filename='cluster_regression.opb', solver='gurobi'):
        """
        Solves the OPB model using Gurobi or SCIP and retrieves cluster assignments.

        Parameters:
        - opb_filename : str
            Path to the OPB file
        - solver : str
            'gurobi' or 'scip'
        """
        if solver.lower() == 'gurobi':
            model = gp.read(opb_filename)
            model.setParam('OutputFlag', 1 if self.verbose else 0)

            model.setParam('TimeLimit', 10800)  # 3 hours
            model.setParam('MIPGap', 0.05)  # 5% optimality gap
            model.optimize()

            z_values = {}
            for v in model.getVars():
                z_values[v.varName] = v.X

        elif solver.lower() == 'scip':
            
            scip_model = Model()
            if not self.verbose:
                scip_model.setParam('display/verblevel', 0)

            scip_model.setParam('limits/time', 10800)  # 3 hours
            scip_model.setParam('limits/gap', 0.05)  # 5% optimality gap
            scip_model.readProblem(opb_filename)
            scip_model.optimize()

            z_values = {}
            for v in scip_model.getVars():
                z_values[v.name] = scip_model.getVal(v)

        else:
            raise ValueError(f"Unsupported solver '{solver}'. Use 'gurobi' or 'scip'.")

        
        self.cluster_assignments = np.zeros(self.n, dtype=int)
        for i in range(self.n):
            for k in range(self.K):
                var_name = f"z_{i}_{k}"
                if z_values.get(var_name, 0) > 0.5:
                    self.cluster_assignments[i] = k
                    break

        self._print("Cluster Assignments from OPB Model:")
        self._print(self.cluster_assignments)


    def compute_cluster_weights(self):
        """
        Computes the regression weights for each cluster based on cluster assignments.
        """
        if self.cluster_assignments is None:
            raise ValueError("Cluster assignments not available. Call solve_opb_model() first.")

        self.w_star_list = []  # Reset the list
        y_pred = np.zeros(self.n)  # Initialize predicted y values
        for k in range(self.K):
            # Create z_k vector
            z_k = (self.cluster_assignments == k).astype(float)  # n-dimensional vector
            indices = np.where(self.cluster_assignments == k)[0]

            if len(indices) == 0:
                # Skip empty clusters
                self.w_star_list.append(np.zeros(self.p + 1))
                continue

            # Compute w_k^* in transformed space
            w_star_k_transformed = self.A @ (self.XV.T @ (z_k * self.y))

            # Map w_k^* back to original space
            w_star_k = self.V @ w_star_k_transformed

            # Make predictions
            y_pred[indices] = self.X_augmented[indices] @ w_star_k

            self.w_star_list.append(w_star_k)
            self._print(f"\nCluster {k} weights w_{k}^* (including bias) in original space:")
            self._print(w_star_k)

        mse = np.mean((self.y - y_pred) ** 2)
        r2 = 1 - np.sum((self.y - y_pred) ** 2) / np.sum((self.y - np.mean(self.y)) ** 2)

        self.mse_milp = mse
        self.r2_milp = r2

    
    def perform_refit_sklearn_ridge_regression(self, ground_truth=False):
        """
        Refits the model using sklearn Ridge regression on each cluster.
        
        Parameters:
        ground_truth : bool
            If True, uses true cluster assignments instead of predicted ones.
        """
        if ground_truth:
            cluster_assignments = self.true_cluster_assignments
        else:
            if self.cluster_assignments is None:
                raise ValueError("No cluster assignments available. Run solve_opb_model first.")
            cluster_assignments = self.cluster_assignments

        w_star_refit = []
        for k in range(self.K):
            # Get data points for current cluster
            indices = np.where(cluster_assignments == k)[0]
            
            if len(indices) > 0:
                # Fit Ridge regression for current cluster
                ridge = Ridge(alpha=self.lambda_reg)
                X_k = self.X[indices]
                y_k = self.y[indices]
                ridge.fit(X_k, y_k)
                
                # Store weights (intercept and coefficients)
                w_star_refit.append(
                    np.hstack([ridge.intercept_, ridge.coef_])
                )
            else:
                # Handle empty clusters
                w_star_refit.append(np.zeros(self.p + 1))
        
        if not ground_truth:
            self.w_star_refit = w_star_refit
            
        return w_star_refit