import pandas as pd
from data import Data
from loss_functions import compute_rmse

class Baselines:
    """
    A class that encapsulates the simple user and movie based mean 
    and median models.
    """

    def __init__(self, data=None, test_purpose=False):
        """
        Constructor for the Baselines class. Initializes the inner data 
        structures.
        Args:
            test_purpose: True for testing, False for creating submission
        """
        if data is not None:
            self.data = data
        else:
            self.data = Data(test_purpose=test_purpose)
        self.test_purpose = test_purpose

    def baseline_global_mean(self):
        """
        Makes predictions based on the global mean of the observed
        ratings in the training set.
        Returns:
            predictions: The predictions of the model on the test data
            rmse: The Root Mean Squared Error of the global mean model
                on the training data
        """
        predictions = pd.DataFrame.copy(self.data.test_df)
        predictions['Rating'] = self.data.global_mean
        if self.test_purpose: 
            self.evalueate_model(self.data.global_mean, 'baseline_global_mean')
        return predictions

    def baseline_user_mean(self):
        """
        Makes predictions based on the user mean of the observed
        ratings in the training set. 
        Returns:
            predictions: The predictions of the model on the test data
        """
        predictions = pd.DataFrame.copy(self.data.test_df)
        def predict_group(group):
            group['Rating'] = self.data.user_means[group['User']]
            return group
        predictions = predictions.groupby('User').apply(predict_group)
        if self.test_purpose: 
            self.evalueate_model(predictions['Rating'], 'baseline_user_mean')
        return predictions

    def baseline_movie_mean(self):
        """
        Makes predictions based on the movie mean of the observed
        ratings in the training set. 
        Returns:
            predictions: The predictions of the model on the test data
        """
        predictions = pd.DataFrame.copy(self.data.test_df)
        def predict_group(group):
            group['Rating'] = self.data.item_means[group['Item']]
            return group
        predictions = predictions.groupby('Item').apply(predict_group)
        if self.test_purpose: 
            self.evalueate_model(predictions['Rating'], 'baseline_movie_mean')
        return predictions

    def evalueate_model(self, model_ratings, model_name):
        """
        Evaluates a model on the test set by computing the Root Mean Square 
        Error(RMSE). Prints the value of this test RMSE.
        Args:
            model_ratings: The ratings predicted by the model
            model_name: The name of the model used to make predictions
        """
        rmse = compute_rmse(self.data.test_df['Rating'], model_ratings)
        print('Test RMSE using {}: {}'.format(model_name, rmse))