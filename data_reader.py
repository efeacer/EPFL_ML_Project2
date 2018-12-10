import numpy as np
import scipy.sparse as sp
from collections import defaultdict
from statistics import mean
from itertools import groupby

DATA_TRAIN_PATH = 'Datasets/data_train.csv'
DATA_SUBMISSION_PATH = 'Datasets/sample_submission.csv'

class DataReader():
    """
    Class to load the csv files into the data structures
    used throughout the project.
    """

    def __init__(self):
        """
        Initializes the internal data structures and statistics
        """
        self.data_train_path = DATA_TRAIN_PATH
        self.data_test_path = DATA_SUBMISSION_PATH
        self.train_user_key = defaultdict(dict)
        self.train_item_key = defaultdict(dict)
        self.load_data()
        self.test_user_key = defaultdict(dict)
        self.test_item_key = defaultdict(dict)
        self.load_data(is_train=False)
        self.user_means = {}
        self.item_means = {}
        self.global_mean = 0.0 
        self.compute_statistics()
        self.observed_train = self.get_observed_entries(self.train_matrix)
        self.observed_test = self.get_observed_entries(self.test_matrix)
        _, self.observed_by_row_train, self.observed_by_col_train = self.build_index_groups(
            self.train_matrix)

    def load_data(self, is_train=True):
        """
        Loads the data into the inner data structures
        Args:
            is_train: A flag indicating whether to populate the training
                dictionaries or the test dictionaries
        """
        data = self.read_txt(is_train=is_train)[1:]
        self.preprocess_data(data, is_train=is_train)

    def read_txt(self, is_train=True):
        """
        Reads and returns the contents of a file from its path.
        Args:
            is_train: A flag indicating whether to populate the training
                dictionaries or the test dictionaries
        Returns:
            Content of the file
        """
        path = self.data_train_path if is_train else self.data_test_path
        with open(path, 'r') as f:
            return f.read().splitlines()

    def preprocess_data(self, data, is_train=True):
        """
        Preprocessing the data from a file, and populating two dictionaries 
        of dictionaries that will be used as the storage structures. If 
        is_train flag is True, populates the training structures, otherwise
        populates the test structures.
        Args:
            data: The contents of the file
            is_train: A flag indicating whether to populate the training
                dictionaries or the test dictionaries
        """
        def deal_line(line):
            pos, rating = line.split(',')
            row, col = pos.split('_')
            row = row.replace('r', '')
            col = col.replace('c', '')
            return int(row), int(col), float(rating)
        def statistics(data):
            row = set([line[0] for line in data])
            col = set([line[1] for line in data])
            return max(row), max(col)
        # parse each line
        data = [deal_line(line) for line in data]
        max_row, max_col = statistics(data)
        if is_train:
            self.train_matrix = sp.lil_matrix((max_row, max_col))
            for user, item, rating in data:
                self.train_user_key[user - 1][item - 1] = rating
                self.train_item_key[item - 1][user - 1] = rating
                self.train_matrix[user - 1, item - 1] = rating
        else:
            self.test_matrix = sp.lil_matrix((max_row, max_col))
            for user, item, rating in data:
                self.test_user_key[user - 1][item - 1] = rating
                self.test_item_key[item - 1][user - 1] = rating
                self.test_matrix[user - 1, item - 1] = rating

    def compute_statistics(self):
        """
        Computes the global mean, user means and item means using the
        training data.
        """
        rating_sum = rating_num = 0
        for user in self.train_user_key:
            rating_sum_user = sum(self.train_user_key[user].values())
            rating_num_user = len(self.train_user_key[user])
            rating_sum += rating_sum_user
            rating_num += rating_num_user
            self.user_means[user] = rating_sum_user / float(rating_num_user)
        for item in self.train_item_key:
            self.item_means[item] = mean(self.train_item_key[item].values())
        self.global_mean = rating_sum / float(rating_num)

    def get_user(self, user):
        """
        Retrieves the {item : rating} dictionary for a specified user.
        Args:
            user: The specified user
        Returns:
            The {item : rating} dictionary for user 
        """
        return self.train_user_key[user]

    def get_item(self, item):
        """
        Retrieves the {user : rating} dictionary for a specified item.
        Args:
            item: The specified item
        Returns:
            The {user : rating} dictionary for item
        """
        return self.train_item_key[item]

    def get_rating_train(self, user, item):
        """
        Retrieves the rating for a specified user, item pair from the
        training set.
        Args:
            user: The specified user
            item: The specified item
        Returns:
            The rating for the specified user, item pair.
        """
        return self.train_matrix[user, item]

    def get_rating_test(self, user, item):
        """
        Retrieves the rating for a specified user, item pair from the
        test set.
        Args:
            user: The specified user
            item: The specified item
        Returns:
            The rating for the specified user, item pair.
        """
        return self.test_matrix[user, item]

    def contains_user(self, user):
        """ 
        Checks whether a specified user is in the training data or not.
        Args:
            user: The specified user
        Returns:
            True if the user is in the training data, False otherwise
        """
        return user in self.train_user_key.values()

    def contains_item(self, item):
        """ 
        Checks whether a specified item is in the training data or not.
        Args:
            item: The specified item
        Returns:
            True if the item is in the training data, False otherwise
        """
        return item in self.train_item_key.values()

    def contains_rating(self, user, item):
        """ 
        Checks whether a rating exists is in the training data for a 
        specified user item pair or not.
        Args:
            user: The specified user
            item: The specified item
        Returns:
            True if the rating exists in the training data, False otherwise
        """
        return user in self.train_user_key and item in self.train_user_key[user]

    def num_users(self):
        """ 
        Returns the number of users in the training set.
        Returns:
            Number of users in the training set
        """
        return self.train_matrix.shape[0]

    def num_items(self):
        """ 
        Returns the number of items in the training set.
        Returns:
            Number of items in the training set
        """
        return self.train_matrix.shape[1]

    def get_observed_entries(self, matrix: sp.lil_matrix):
        """
        Finds and returns a list containing positions (row, column) of the observed 
        entries in a matrix, i.e. nonzero entries.
        Args:
            matrix: The given matrix containing the zero and nonzero entries.
        Returns:
            observed_entries: The list containing the positions (row, column)
                of the nonzero entries.
        """
        nonzero_rows, nonzero_columns = matrix.nonzero()
        observed_entries = zip(nonzero_rows, nonzero_columns)
        return list(observed_entries)

    def group_by(self, data, index):
        """
        Seperates a list to groups by a specified index. Returns an iterator to
        the resulting groups.
        Args:
            data: The list to be grouped
            index: The specified index thatdetermines the groups
        Returns: 
            grouped_data: The iterator to the resulting groups
        """
        sorted_data = sorted(data, key=lambda x: x[index])
        groupby_data = groupby(sorted_data, lambda x: x[index])
        return groupby_data

    def build_index_groups(self, data):
        """
        Builds and returns two groups from the given data. One group is for rows
        and the indices of nonzero items in them, the other group is for 
        columns and the indices of nonzero in them.
        Args:
            data: Data to be grouped
        Returns:
            observed_entries: Positions (row, column) of observed entries in the data
            observed_entries_by_row: Indices of nonzero entries in each row
            observed_entries_by_column: Indices of nonzero entries in each column
        """
        nonzero_rows, nonzero_columns = data.nonzero()
        observed_entries = list(zip(nonzero_rows, nonzero_columns))
        groups_by_rows = self.group_by(observed_entries, index=0)
        observed_entries_by_row = [(group_name, np.array([x[1] for x in value])) 
            for group_name, value in groups_by_rows]
        groups_by_columns= self.group_by(observed_entries, index=1)
        observed_entries_by_column = [(group_name, np.array([x[0] for x in value])) 
            for group_name, value in groups_by_columns]
        return observed_entries, observed_entries_by_row, observed_entries_by_column

    def get_items_rated_by(self, user):
        """
        Returns the list of items rated by a specified user.
        Returns:
            The list of items rated by a specified user
        """
        return self.train_user_key[user].keys()

if __name__ == '__main__':
    test_reader = DataReader()
    # testing 
    # print(test_reader.train_matrix)
    print(test_reader.observed_train)