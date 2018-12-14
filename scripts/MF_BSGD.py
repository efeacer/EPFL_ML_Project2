import numpy as np
from MF import MF

class MF_BSGD(MF):
    """
    Implementation of a simple matrix factorization model trained
    using Biased Stochastic Gradient Descent (BSGD)
    """

    def __init__(self, data=None, test_purpose=False):
        """
        Initializes internal data structures and hyperparameters.
        Args:
            test_purpose: True for testing, False for creating submission
        """
        super().__init__(data=data, test_purpose=test_purpose)
        self.init_hyperparams()
        self.init_biases()

    def train(self): # BSGD
        """
        Optimizes Mean Squared Error loss function using Biased Stochastic Gradient
        Descent (BSGD) to learn two feature matrices that factorizes the given 
        training data together with biases.
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

    def predict(self, user, item):
        """
        Predicts a rating for the specified user, item pair. Includes the baseline
        effect.
        Args:
            user: The specified user
            item: The specified item
        Returns:
            The predicted rating for the specified user, item pair
        """
        prediction = super().predict(user, item)
        return prediction + self.global_bias + self.user_biases[user] + self.item_biases[item]

    def init_biases(self):
        """
        Initializes global, user and movie biases.
        """
        self.global_bias = self.data.global_mean
        self.user_biases = np.zeros(self.data.num_users, dtype=float)
        self.item_biases = np.zeros(self.data.num_items, dtype=float)

    def init_hyperparams(self):
        """
        Initializes the hyperparameters used in BSGD.
        """
        self.gamma = 0.02
        self.lambda_user = 0.07
        self.lambda_item = 0.07
        self.num_epochs = 20
        self.lambda_u_bias = 0.001
        self.lambda_i_bias = 0.001
    
# Testing
if __name__ == '__main__':
    model = MF_BSGD(test_purpose=True)
    model.train()
    model.plot()