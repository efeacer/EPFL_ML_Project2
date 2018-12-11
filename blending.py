from scipy.optimize import minimize
from loss_functions import compute_rmse

class Blending:
    """
    A class that encapsulates algorithms to mix different models together.
    """

    def __init__(self, models, test_ratings, weights=None):
        """
        Constructor for the Blending class. Initializes the internal list
        of models (prediction results for each model) whose items will be 
        mixed together.
        Args:
            models: List of models (prediction results for each model) whose 
                items will be mixed together.
            test_ratings: Test set of ratings that will be used as 
                the set of true values during optimization
        """
        self.__models = models
        if weights is not None:
            self.__weights = weights
        else:
            self.__weights = [1.0 / len(models) for x in models]
        self.__test_ratings = test_ratings

    def optimize_weighted_average(self):
        """
        Minimizes the objective function to find the optimal weight vector.
        Returns:
            optimal_weights: The optimal weight vector
        """
        result = minimize(fun=self.__objective_function, x0=self.__weights, method='SLSQP')
        print(result)
        self.__weights = result.x
        return self.__weights

    def get_weighted_average(self):
        """
        Returns the weighted combination of prediction results of each model.
        Returns:
            mixed_models: The weighted combination of prediction results of each model.
        """
        mixed_models = 0
        for i, model in enumerate(self.__models):
            mixed_models += self.__weights[i] * model
        return mixed_models

    def __objective_function(self, weights):
        """
        Objective function to be optimized. The function is indeed the Root Mean 
        Squared Error of the weighted combination of prediction results of each model.
        Args:
            weights: A list containing the individual weights of each model
        """
        mixed_models = 0
        for i, model in enumerate(self.__models):
            mixed_models += weights[i] * model
        rmse = compute_rmse(self.__test_ratings, mixed_models)
        return rmse