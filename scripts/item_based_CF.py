import numpy as np
from similarity_matrix import SimilarityMatrix
from data import Data
from heapq import nlargest
from statistics import mean

class ItemBased:
    """
    Implementation of item based Collaborative Filtering using kNN.
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
        self.num_neighbors = 50

    def init_sim_matrix(self):
        print('Building item similarity matrix ...')
        self.item_sim_matrix = SimilarityMatrix()
        for item_test in self.data.test_item_key:
            for item_train in self.data.train_item_key:
                if item_test != item_train:
                    if not self.item_sim_matrix.contains(item_test, item_train):
                        sim = self.pearson_correlation(item_test, item_train)
                        self.item_sim_matrix.set(item_test, item_train, sim)
        print('... item similarity matrix is ready.')

    def predict(self, user, item):
        nearest_neighbors = nlargest(self.num_neighbors, 
            self.item_sim_matrix.get(item).items, key=lambda x: x[1])
        numerator = denominator = 0.0
        for neighbor, sim in nearest_neighbors:
            if self.data.contains_rating(user, neighbor):
                rating = self.data.get_rating(user, neighbor)
                neighbor_avg = self.data.item_means[neighbor]
                numerator += sim * (rating - neighbor_avg)
                denominator += np.fabs(sim)
        prediction = self.data.item_means[item]
        if denominator == 0:
            return prediction
        prediction += numerator / denominator
        return prediction

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