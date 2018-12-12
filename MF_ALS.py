import numpy as np
import scipy.sparse as sp
from MF import MF

class MF_ALS(MF):
    """
    Implementation of a simple matrix factorization model trained
    using Alternating Least Squares (ALS)
    """

    def __init__(self, data=None, test_purpose=False):
        """
        Initializes internal data structures and hyperparameters.
        Args:
            test_purpose: True for testing, False for creating submission
        """
        super().__init__(data=data, test_purpose=test_purpose, num_features=8)
        self.init_hyperparams()

    def train(self): # ALS
        """
        Optimizes Mean Squared Error loss function using Alternating Least Squares
        (ALS) to learn two feature matrices that factorizes the given training data.
        Returns:
            predictions_df: The predictions of the model on the test data as a Pandas 
                Data Frame.
        """
        print('Learning the matrix factorization using ALS ...')
        for i in range(self.num_epochs):
            self.update_user_features()
            self.update_item_features()
            self.train_rmses.append(self.compute_rmse())
            print('Iteration: {}, RMSE on training set: {}'.format(i + 1, self.train_rmses[-1]))
            if self.is_converged():
                print('The training process converged to a threshold.'); break
        print('... Final RMSE on training set: {}'.format(self.train_rmses[-1]))
        if self.test_purpose: 
            print('Test RMSE: {}'.format(self.compute_rmse(is_train=False)))
        predictions_df = self.get_predictions()
        return predictions_df

    def update_user_features(self):
        """
        Updates the user feature matrix by solving the normal equations of ALS.
        """
        num_users = self.data.num_users()
        num_nonzero_rows = self.data.train_sp.getnnz(axis=1)
        lambda_I = self.lambda_user * sp.eye(self.num_features)
        updated_user_features = np.zeros((num_users, self.num_features))
        for user, items in self.data.observed_by_row_train: # optimize one group
            Z = self.item_features[items]
            Z_T_Z_regularized = Z.T.dot(Z) + num_nonzero_rows[user] * lambda_I
            X = self.data.get_rating(user, items)
            X_Z = X.dot(Z)
            W_star = np.linalg.solve(Z_T_Z_regularized, X_Z.T)
            updated_user_features[user] = W_star.T
        self.user_features = updated_user_features

    def update_item_features(self):
        """
        Updates the item feature matrix by solving the normal equations of ALS.
        """
        num_items = self.data.num_items()
        num_nonzero_columns = self.data.train_sp.getnnz(axis=0)
        lambda_I = self.lambda_item * sp.eye(self.num_features)
        updated_item_features = np.zeros((num_items, self.num_features))
        for item, users in self.data.observed_by_col_train:
            Z = self.user_features[users]
            Z_T_Z_regularized = Z.T.dot(Z) + num_nonzero_columns[item] * lambda_I
            X = self.data.get_rating(users, item)
            X_Z = X.T.dot(Z)
            W_star = np.linalg.solve(Z_T_Z_regularized, X_Z.T)
            updated_item_features[item] = W_star.T
        self.item_features = updated_item_features

    def predict(self, user, item):
        """
        Predicts a rating for the specified user, item pair.
        Args:
            user: The specified user
            item: The specified item
        Returns:
            The predicted rating for the specified user, item pair
        """
        return super().predict(user, item)

    def init_hyperparams(self):
        """
        Initializes the hyperparameters used in ALS.
        """
        self.lambda_user = 0.081
        self.lambda_item = 0.081
        self.num_epochs = 24
    
# Testing
if __name__ == '__main__':
    model = MF_ALS(test_purpose=True)
    model.train()