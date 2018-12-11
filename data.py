import numpy as np
import scipy.sparse as sp
import data_processing as dp
from collections import defaultdict
from statistics import mean
from itertools import groupby
from sklearn.model_selection import train_test_split

DATA_TRAIN_PATH = 'Datasets/data_train.csv'
SUBMISSION_PATH = 'Datasets/sample_submission.csv'
TEST_SET_SIZE = 0.1

class Data():
    """
    Class represent the dataset by loading the csv files into various
    data structures used throughout the project.
    """

    def __init__(self, data_train_path=None, data_test_path=None, test_purpose=False):
        """
        Initializes the internal data structures and statistics
        Args:
            data_train_path: The specified path for the csv file
                containing the training dataset
            data_test_path: The specified path for the csv file
                containing the test dataset
            test_purpose: True for testing, False for creating submission
        """
        print('Preparing data ...')
        if data_train_path is None:
            data_train_path = DATA_TRAIN_PATH
        if data_test_path is None:
            data_test_path = SUBMISSION_PATH
        if test_purpose:
            print('Splitting data to train and test data ...')
            data_df = dp.load_csv_df(data_train_path)
            self.train_df, self.test_df = train_test_split(data_df, test_size=0.1)
            self.train_sp = dp.df_to_sp(self.train_df)
            self.test_sp = dp.df_to_sp(self.test_df)
            self.train_user_key, self.train_item_key = dp.df_to_dict(self.train_df)
            self.test_user_key, self.test_item_key = dp.df_to_dict(self.test_df)
            print('... data is splitted.')
        else:
            self.train_df = dp.load_csv_df(data_train_path)
            self.test_df = dp.load_csv_df(data_test_path)
            self.train_sp = dp.load_csv_sp(data_train_path)
            self.test_sp = dp.load_csv_sp(data_test_path)
            self.train_user_key, self.train_item_key = dp.load_csv_dict(data_train_path)
            self.test_user_key, self.test_item_key = dp.load_csv_dict(data_test_path)
        self.init_statistics()
        self.observed_train = self.get_observed_entries(self.train_sp)
        self.observed_test = self.get_observed_entries(self.test_sp)
        _, self.observed_by_row_train, self.observed_by_col_train = self.build_index_groups(self.train_sp)
        print('... data is prepared.')

    def init_statistics(self):
        """
        Computes and initializes the global mean, user means and item means 
        using the training data.
        """
        self.global_mean = self.train_df['Rating'].mean()
        self.user_means = {}
        self.item_means = {}
        for user in self.train_user_key:
            self.user_means[user] = mean(self.train_user_key[user].values())
        for item in self.train_item_key:
            self.item_means[item] = mean(self.train_item_key[item].values())

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

    def get_user(self, user, from_train=True):
        """
        Retrieves the {item : rating} dictionary for a specified user
        from the training or test set.
        Args:
            user: The specified user
            from_train: True for training set retrieval, False for test
                set retrieval
        Returns:
            The {item : rating} dictionary for user 
        """
        if from_train:
            return self.train_user_key[user]
        return self.test_user_key[user]

    def get_item(self, item, from_train=True):
        """
        Retrieves the {user : rating} dictionary for a specified item 
        from the training or test set.
        Args:
            item: The specified item
            from_train: True for training set retrieval, False for test
                set retrieval
        Returns:
            The {user : rating} dictionary for item
        """
        if from_train:
            return self.train_item_key[item]
        return self.test_item_key[item]

    def get_rating(self, user, item, from_train=True):
        """
        Retrieves the rating for a specified user, item pair from the
        training or test set.
        Args:
            user: The specified user
            item: The specified item
            from_train: True for training set retrieval, False for test
                set retrieval
        Returns:
            The rating for the specified user, item pair.
        """
        if from_train:
            return self.train_sp[user, item]
        return self.test_sp[user, item]

    def contains_user(self, user, in_train=True):
        """ 
        Checks whether a specified user is in the training/test data or not.
        Args:
            user: The specified user
            in_train: True to check training set, False to check test set
        Returns:
            True if the user is in the data, False otherwise
        """
        if in_train:
            return user in self.train_user_key.values()
        return user in self.test_user_key.values()

    def contains_item(self, item, in_train=True):
        """ 
        Checks whether a specified item is in the training/test data or not.
        Args:
            item: The specified item
            in_train: True to check training set, False to check test set
        Returns:
            True if the item is in the data, False otherwise
        """
        if in_train:
            return item in self.train_item_key.values()
        return item in self.test_item_key.values()

    def contains_rating(self, user, item, in_train=True):
        """ 
        Checks whether a rating exists is in the training/test data for a 
        specified user item pair or not.
        Args:
            user: The specified user
            item: The specified item
            in_train: True to check training set, False to check test set
        Returns:
            True if the rating exists in the data, False otherwise
        """
        return user in self.train_user_key and item in self.train_user_key[user]

    def num_users(self, in_train=True):
        """ 
        Returns the number of users in the training/test set.
        Args:
            in_train: True to get number of users from the training set, 
                False to get the number of users from test set
        Returns:
            Number of users in the training/test set
        """
        return self.train_sp.shape[0]

    def num_items(self, in_train=True):
        """ 
        Returns the number of items in the training/test set.
        Args:
            in_train: True to get number of items from the training set, 
                False to get the number of items from test set
        Returns:
            Number of items in the training/test set
        """
        return self.train_sp.shape[1]

    def get_items_rated_by(self, user, from_train=True):
        """
        Returns the list of items rated by a specified user from the
        training or the test data.
        Args:
            from_train: True for training set retrieval, False for test
                set retrieval
        Returns:
            The list of items rated by a specified user
        """
        return self.train_user_key[user].keys()

if __name__ == '__main__':
    data = Data()
    # testing 
    #print(test_reader.train_matrix)
    #print(data.observed_by_col_train)