import numpy as np

def compute_mse(real_ratings, predictions):
    """
    Computes the Mean Squared Error given the real and the predicted
    ratings.
    Args:
        real_ratings: The actual ratings
        predictions: Ratings predicted by some model
    Returns:
        mse: The Mean Squared Error value
    """
    error_vector = real_ratings - predictions
    return np.mean(error_vector ** 2)

def compute_rmse(real_ratings, predictions):
    """
    Computes the Root Mean Squared Error given the real and the
    predicted ratings.
    Args:
        real_ratings: The actual ratings
        predictions: Ratings predicted by some model
    Returns:
        rmse: The Root Mean Squared Error value
    """
    rmse = np.sqrt(compute_mse(real_ratings, predictions))
    return rmse