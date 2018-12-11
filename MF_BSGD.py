import numpy as np
import scipy.sparse as sp
from data import Data
from data_processing import sp_to_df

class MF_BSGD:
    """
    Implementation of a simple matrix factorization model trained
    using Biased Stochastic Gradient Descent (BSGD)
    """

    def __init__(self, data=None, test_purpose=False):
        """
        Initializes inner data structures and hyperparameters.
        Args:
            test_purpose: True for testing, False for creating submission
        """
        if data is not None:
            self.data = data
        else:
            self.data = Data(test_purpose=test_purpose)
        self.test_purpose = test_purpose
        self.num_features = 20
        self.init_matrices()
        self.init_biases()
        self.train_rmses = [0, 0]
        self.init_hyperparams()

    def train(self): # BSGD
        """
        Optimizes Mean Squared Error loss function using Biased Stochastic Gradient
        Descent (BSGD) to learn two feature matrices that factorizes the given 
        training data.
        Returns:
            predictions_df: The predictions of the model on the test data as a Pandas 
                Data Frame.
        """
        print('Learning the matrix factorization using BSGD ...')
        for i in range(self.num_epochs):
            np.random.shuffle(self.data.observed_train)
            self.gamma /= 1.2
            for row, col in self.data.observed_train:
                user_vector, user_bias = self.user_features[row], self.user_biases[row] 
                item_vector, item_bias = self.item_features[col], self.item_biases[col]
                error = self.data.get_rating(row, col) - self.predict(row, col)
                # Updates using gradients
                self.user_biases[row] += self.gamma * (error - self.lambda_u_bias * user_bias)
                self.item_biases[col] += self.gamma * (error- self.lambda_i_bias * item_bias)
                self.user_features[row] += self.gamma * (error * item_vector -
                    self.lambda_user * user_vector)
                self.item_features[col] += self.gamma * (error * user_vector -
                    self.lambda_item * item_vector)
            self.train_rmses.append(self.compute_rmse())
            print('Iteration: {}, RMSE on training set: {}'.format(i + 1, self.train_rmses[-1]))
            if self.is_converged():
                print('The training process converged to a threshold.'); break
        print('... Final RMSE on training set: {}'.format(self.train_rmses[-1]))
        if self.test_purpose: 
            print('Test RMSE: {}'.format(self.compute_rmse(is_train=False)))
        predictions_df = self.get_predictions()
        return predictions_df

    def is_converged(self):
        """
        Determines whether the training process converged to a threshold
        or not.
        Returns:
            True if the training process converged to a threshold, False otherwise
        """
        return np.fabs(self.train_rmses[-1] - self.train_rmses[-2]) < self.threshold

    def compute_rmse(self, is_train=True):
        """
        Computes the Root Mean Squared Error (RMSE) on the training set or 
        the test set depending on the is_train flag.
        Args:
            is_train: A flag indicating whether to compute the RMSE on the 
                training set or the test set
        Returns:
            rmse: The Root Mean Squared Error value
        """
        mse = 0
        observed = self.data.observed_train if is_train else self.data.observed_test
        def get_rating(user, item):
            if is_train:
                return self.data.get_rating(user, item)
            return self.data.get_rating(user, item, from_train=False)
        for user, item in observed:
            error = get_rating(user, item) - self.predict(user, item)
            mse += (error ** 2) 
        mse /= len(observed)
        return np.sqrt(mse) # rmse

    def get_predictions(self):
        """
        Computes and returns the predictions based on the two feature matrices resulted 
        from the matrix factorization process and the baselines.
        Returns:
            predictions: The predictions of the model on the test data as a Pandas
                Data Frame.
        """
        predictions = self.user_features.dot(self.item_features.T)
        predictions_sp = sp.lil_matrix.copy(self.data.test_sp)
        for row, col in self.data.observed_test:
            prediction = predictions[row, col]
            prediction += self.global_bias + self.user_biases[row] + self.item_biases[col] 
            predictions_sp[row, col] = prediction
        predictions_df = sp_to_df(predictions_sp)
        return predictions_df

    def predict(self, user, item):
        """
        Predicts a rating for the specified user, item pair.
        Args:
            user: The specified user
            item: The specified item
        Returns:
            The predicted rating for the specified user, item pair
        """
        baseline = self.global_bias + self.user_biases[user] + self.item_biases[item]  
        return baseline + self.user_features[user].dot(self.item_features[item])

    def init_matrices(self):
        """
        Initializes the user and item feature matrices in random.
        """
        num_users, num_items = self.data.num_users(), self.data.num_items()
        self.user_features = np.random.rand(num_users, self.num_features)
        self.item_features = np.random.rand(num_items, self.num_features)

    def init_biases(self):
        """
        Initializes global, user and movie biases.
        """
        num_users, num_items = self.data.num_users(), self.data.num_items()
        self.global_bias = self.data.global_mean
        self.user_biases = np.zeros(num_users, dtype=float)
        self.item_biases = np.zeros(num_items, dtype=float)

    def init_hyperparams(self):
        """
        Initializes the hyperparameters used in BSGD.
        """
        self.gamma = 0.025
        self.lambda_user = 0.1
        self.lambda_item = 0.1
        self.num_epochs = 20
        self.lambda_u_bias = 0.001
        self.lambda_i_bias = 0.001
        self.threshold = 1e-4
    
if __name__ == '__main__':
    model = MF_BSGD(test_purpose=True)
    model.train()