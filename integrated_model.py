import numpy as np
from data_reader import DataReader
from statistics import mean
from collections import defaultdict
from heapq import nlargest

class IntegratedModel:
    """
    Implementation of a matrix factorization model trained
    using Yehuda Koren's integrated model.
    """

    def __init__(self, data=None):
        """
        Initializes inner data structures and hyperparameters.
        """
        if data is not None:
            self.data = data
        else:
            print('Preparing data ...')
            self.data = DataReader()
            print('... data prepared.')
        self.num_features = 20
        self.num_neighbors = 50
        self.init_matrices()
        self.init_biases()
        self.neighborhood = defaultdict(dict)
        self.train_rmses = [0, 0]
        self.init_hyperparams()
        self.test_purpose = False

    def train(self): # SGD
        """
        Optimizes Mean Squared Error loss function using Stochastic Gradient Descent
        (SGD) to learn two feature matrices that factorizes the given training data.
        Returns:
            predictions: The predictions of the model on the test data
        """
        print('Learning the integrated model using SGD ...')
        for i in range(self.num_epochs):
            for row, col in self.data.observed_train:
                user_vector, user_bias = self.user_features[row], self.user_biases[row] 
                item_vector, item_bias = self.item_features[col], self.item_biases[col]
                prediction, neighbor_items = self.predict(row, col)
                error = self.data.get_rating_train(row, col) - prediction
                # Updates using gradients
                self.user_biases[row] += self.gamma * (error - self.lambda_u_bias * user_bias)
                self.item_biases[col] += self.gamma * (error- self.lambda_i_bias * item_bias)
                self.user_features[row] += self.gamma * (error * item_vector -
                    self.lambda_user * user_vector)
                self.item_features[col] += self.gamma * (error * user_vector -
                    self.lambda_item * item_vector)
                neighbor_normalizer = np.sqrt(len(neighbor_items))
                for neighbor in neighbor_items:
                    rating = self.data.get_rating_train(row, neighbor)
                    baseline = self.global_bias + self.user_biases[row] + self.item_biases[neighbor]
                    neighborhood_weight = self.neighbor_weights[row, neighbor]
                    self.neighbor_weights[row, neighbor] += self.gamma * (error * (rating - 
                        baseline) / neighbor_normalizer - self.lambda_neighbor * neighborhood_weight)
            self.train_rmses.append(self.compute_rmse())
            print('Iteration: {}, RMSE on training set: {}'.format(i + 1, self.train_rmses[-1]))
            if self.is_converged():
                print('The training process converged to a threshold.'); break
        print('... Final RMSE on training set: {}'.format(self.train_rmses[-1]))
        if self.test_purpose: 
            print('Test RMSE: {}'.format(self.compute_rmse(is_train=False)))

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
        print('n') #test purpose
        mse = 0
        observed = self.data.observed_train if is_train else self.data.observed_test
        def get_rating(user, item):
            if is_train:
                return self.data.get_rating_train(user, item)
            return self.data.get_rating_test(user, item)
        for user, item in observed:
            error = get_rating(user, item) - self.predict(user, item)
            mse += (error ** 2) 
        mse /= len(observed)
        return np.sqrt(mse) # rmse

    def predict(self, user, item):
        """
        Predicts a rating for the specified user, item pair.
        Args:
            user: The specified user
            item: The specified item
        Returns:
            prediction: The predicted rating for the specified user, item pair
            neighbor_items: Returns the neighbor set in order to help the caller
                function
        """
        prediction = self.global_bias + self.user_biases[user] + self.item_biases[item]
        prediction += self.user_features[user].dot(self.item_features[item]) 
        neighborhood_effect, neighbor_items = self.compute_neighborhood_effect(user, item)
        prediction += neighborhood_effect
        return prediction, neighbor_items

    def compute_neighborhood_effect(self, user, item):
        """
        Computes and returns the term in the loss function which comes from the
        neighbor items of a specified user, item pair. Also returns the neighbor
        items set.
        Args:
            user: The specified user
            item: The specified item
        Returns:
            neighborhood_effect: The term in the loss function which comes from the
                neighbor items of a specified user, item pair
            neighbor_items: Set of neighbor items for a specified user, item pair
        """
        neighborhood_effect = 0.0
        neighbor_items = self.get_neighbor_items(user, item)
        neighbor_normalizer = np.sqrt(len(neighbor_items))
        for neighbor in neighbor_items:
            rating = self.data.get_rating_train(user, neighbor)
            baseline = self.global_bias + self.user_biases[user] + self.item_biases[neighbor]
            neighborhood_effect += self.neighbor_weights[item, neighbor] * (
                rating - baseline)
        if neighbor_normalizer != 0:
            neighborhood_effect /= neighbor_normalizer
        return neighborhood_effect, neighbor_items

    def get_neighbor_items(self, user, item):
        """
        Computes and returns the nearest neighbors of a specified item 
        that are all rated by a specified user. The nearest (similarity)
        metric is Pearson Correlation 
        Args:
            user: The specified user
            item: The specified item
        Returns:
            nearest_neighbors: Nearest neighbors of the specified item 
                that are all rated by the specified user
        """
        if user in self.neighborhood and item in self.neighborhood[user]:
            return self.neighborhood[user][item]
        items = self.data.get_items_rated_by(user)
        item_sim = {}
        for neighbor_item in items:
            if neighbor_item != item:
                sim = self.pearson_correlation(item, neighbor_item)
                item_sim[neighbor_item] = sim
        nearest_neighbors = nlargest(self.num_neighbors, item_sim.items(), key=lambda x: x[1])
        nearest_neighbors = list(zip(*nearest_neighbors))[0]
        if not nearest_neighbors:
            return []
        self.neighborhood[user][item] = nearest_neighbors
        return nearest_neighbors

    def pearson_correlation(self, item1, item2):
        """
        Computes and return the Pearson Correlation Coefficient between two
        specified items.
        Args:
            item1: The first specified item
            item2: The second specified item
        Returns:
            Pearson Correlation Coefficient between item1 and item2
        """
        item1_dict, item2_dict = self.data.get_item(item1), self.data.get_item(item2)
        common = set(item1_dict.keys()).intersection(set(item2_dict.keys()))
        if not common: return 0.0
        ratings1 = ratings2 = []
        for user in common:
            ratings1.append(item1_dict[user])
            ratings2.append(item2_dict[user])
        item1_avg, item2_avg = mean(ratings1), mean(ratings2)
        numerator = denominator1 = denominator2 = 0.0
        for i in range(len(common)):
            numerator += (ratings1[i] - item1_avg) * (ratings2[i] - item2_avg)
            denominator1 += (ratings1[i] - item1_avg) ** 2
            denominator1 += (ratings2[i] - item2_avg) ** 2
        if denominator1 == 0 or denominator2 == 0: # avoid division by zero
            return 0 
        return numerator / (np.sqrt(denominator1) * np.sqrt(denominator2))

    def init_matrices(self):
        """
        Initializes the user and item feature matrices in random together
        with the neighbor weight matrix.
        """
        num_users, num_items = self.data.num_users(), self.data.num_items()
        self.user_features = np.random.rand(num_users, self.num_features)
        self.item_features = np.random.rand(num_items, self.num_features)
        self.neighbor_weights = np.random.rand(num_items, num_items)

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
        self.gamma = 0.01
        self.lambda_user = 0.007
        self.lambda_item = 0.001 
        self.lambda_u_bias = 0.001
        self.lambda_i_bias = 0.001
        self.lambda_neighbor = 0.01
        self.num_epochs = 10
        self.threshold = 1e-4

if __name__ == '__main__':
    model = IntegratedModel()
    model.train()