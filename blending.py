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
            self.__weights = {'baseline_global_mean': None,
                              'baseline_user_mean': None,
                              'baseline_item_mean': None,
                              'mf_sgd': None,
                              'mf_bsgd': None, 
                              'mf_als': None,
                              'surprise_kNN_baseline_user': None,
                              'surprise_kNN_baseline_item': None,                           
#                              'surprise_SVD': None,
#                              'surprise_SVDpp': None,
                              'surprise_slope_one': None,
                              'surprise_co_clustering': None}
            for key in self.__weights:
                self.__weights[key] = 1.0 / len(models)
        self.__test_ratings = test_ratings

    def optimize_weighted_average(self):
        """
        Minimizes the objective function to find the optimal weight vector.
        Returns:
            optimal_weights: The optimal weight vector
        """
        result = minimize(fun=self.__objective_function, x0=self.__weights.values(),
                          method='SLSQP')
        print(result)
        for i, key in enumerate(self.__weights):
            self.__weights[key] = result.x[i]
        return self.__weights

    def get_weighted_average(self):
        """
        Returns the weighted combination of prediction results of each model.
        Returns:
            mixed_models: The weighted combination of prediction results of each model.
        """
        mixed_models = 0
        for key, value in self.__models.items():
            mixed_models += self.__weights[key] * value
        return mixed_models

    def __objective_function(self, weights):
        """
        Objective function to be optimized. The function is indeed the Root Mean 
        Squared Error of the weighted combination of prediction results of each model.
        Args:
            weights: A list containing the individual weights of each model
        """
        mixed_models = 0
        for key, value in self.__models.items():
            mixed_models += self.__weights[key] * value
        rmse = compute_rmse(self.__test_ratings, mixed_models)
        return rmse