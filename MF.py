from abc import ABC, abstractmethod
from data import Data
import numpy as np
import matplotlib.pylab as plt

class MF(ABC):
    """
    An abstract base class for Matrix Factorization inspired models.
    """

    def __init__(self, data=None, test_purpose=False, num_features=None):
        """
        Initializer for the abstract Matrix Factorization class. Initializes
        the internal data structures and the test flag.
        Args:
            test_purpose: True for testing, False for creating submission
        """
        super().__init__()
        if data is not None:
            self.data = data
        else:
            self.data = Data(test_purpose=test_purpose)
        self.test_purpose = test_purpose
        self.num_features = 20 if num_features is None else num_features
        self.init_matrices()
        self.train_rmses = [0.0, 0.0]
        self.threshold = 1e-4 # the change in error for which training converges

    @abstractmethod
    def train(self):
        """
        Fits the model. Subclasses should implement their own way of training.
        """
        pass

    def is_converged(self):
        """
        Determines whether the training process converged to a threshold
        or not.
        Returns:
            True if the training process converged to a threshold, False otherwise
        """
        return np.fabs(self.train_rmses[-1] - self.train_rmses[-2]) < self.threshold

    def get_predictions(self):
        """
        Computes and returns the predictions based on the two feature matrices resulted 
        from the matrix factorization process.
        Returns:
            predictions: The predictions of the model on the test data as a Pandas
                Data Frame.
        """
        predictions_df = self.data.test_df.copy()
        for i, row in predictions_df.iterrows():
            user = int(row['User'] - 1)
            item = int(row['Item'] - 1)
            predictions_df.at[i, 'Rating'] = self.predict(user, item)
        return predictions_df

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
        
    @abstractmethod
    def predict(self, user, item):
        """
        Predicts a rating for the specified user, item pair. Subclasses
        should implement their own way of predicting. However, the method
        implements the simplest way to predict ratings, which is to compute
        the inner product of latent features.
        Args:
            user: The specified user
            item: The specified item
        Returns:
            The predicted rating for the specified user, item pair
        """
        return self.user_features[user].dot(self.item_features[item])

    def init_matrices(self):
        """
        Subclasses can override this method to initialize their user and
        item feature matrices in any particular way. A random initialization 
        is provided here.
        """
        self.user_features = np.random.rand(self.data.num_users, self.num_features)
        self.item_features = np.random.rand(self.data.num_items, self.num_features)

    def plot(self):
        '''
        Plots the training Root Mean Squared Errors (rmses) as a function 
        of the training epoch (iteration).
        '''
        rmses = self.train_rmses[2:]
        nums = range(len(rmses))
        plt.plot(nums, rmses, label='RMSE')
        plt.xlabel('Epoch')
        plt.ylabel('RMSE')
        plt.title(self.__class__.__name__)
        plt.legend()
        plt.show()