import numpy as np
from MF import MF

class MF_SGD(MF):
    """
    Implementation of a simple matrix factorization model trained
    using Stochastic Gradient Descent (SGD)
    """

    def __init__(self, data=None, test_purpose=False):
        """
        Initializes internal data structures and hyperparameters.
        Args:
            test_purpose: True for testing, False for creating submission
        """
        super().__init__(data=data, test_purpose=test_purpose)
        self.init_hyperparams()

    def train(self): # SGD
        """
        Optimizes Mean Squared Error loss function using Stochastic Gradient Descent
        (SGD) to learn two feature matrices that factorizes the given training data.
        Returns:
            predictions_df: The predictions of the model on the test data as a Pandas 
                Data Frame.
        """
        print('Learning the matrix factorization using SGD ...')
        for i in range(self.num_epochs):
            np.random.shuffle(self.data.observed_train)
            self.gamma /= 1.2
            for row, col in self.data.observed_train:
                user_vector = self.user_features[row]
                item_vector = self.item_features[col]
                error = self.data.get_rating(row, col) - self.predict(row, col)
                # Updates using gradients
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
        Initializes the hyperparameters used in SGD.
        """
        self.gamma = 0.02
        self.lambda_user = 0.07
        self.lambda_item = 0.07
        self.num_epochs = 20
        self.threshold = 1e-4
    
# Testing
if __name__ == '__main__':
    model = MF_SGD(test_purpose=True)
    model.train()
    model.plot()