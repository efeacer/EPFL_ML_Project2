import numpy as np
from MF import MF

class MF_implicit(MF):
    """
    Implementation of the SVD++ algorithm, which takes 
    implicit feedback into account.
    """

    def __init__(self, data=None, test_purpose=False):
        """
        Initializes internal data structures and hyperparameters.
        Args:
            test_purpose: True for testing, False for creating submission
        """
        super().__init__(data=data, test_purpose=test_purpose)
        self.init_hyperparams()
        self.init_biases()
        self.init_implicit_feedbacks()

    def train(self): # SGD
        """
        Optimizes Mean Squared Error loss function using Stochastic Gradient
        Descent (SGD) to learn two feature matrices that factorizes the given 
        training data together with biases and implicit feedback.
        Returns:
            predictions_df: The predictions of the model on the test data as a Pandas 
                Data Frame.
        """
        print('Learning the implicit matrix factorization using SGD ...')
        for i in range(self.num_epochs):
            np.random.shuffle(self.data.observed_train)
            self.gamma /= 1.2
            for row, col in self.data.observed_train:
                user_vector, user_bias = self.user_features[row], self.user_biases[row] 
                item_vector, item_bias = self.item_features[col], self.item_biases[col]
                prediction, implicit_effect, normalizer = self.predict(row, col)
                error = self.data.get_rating(row, col) - prediction
                # Updates using gradients
                self.user_biases[row] += self.gamma * (error - self.lambda_u_bias * user_bias)
                self.item_biases[col] += self.gamma * (error - self.lambda_i_bias * item_bias)
                self.user_features[row] += self.gamma * (error * item_vector -
                                                         self.lambda_user * user_vector)
                self.item_features[col] += self.gamma * (error * (user_vector + implicit_effect) -
                                                         self.lambda_item * item_vector)
                items = self.data.get_items_rated_by(row)
                if normalizer == 0: continue
                for item in items:
                    implicit_feedback = self.implicit_feedbacks[int(item)]
                    self.implicit_feedbacks[int(item)] += self.gamma * (error / normalizer * item_vector
                                                                        - self.lambda_implicit *
                                                                        implicit_feedback)
            self.train_rmses.append(self.compute_rmse())
            print('Iteration: {}, RMSE on training set: {}'.format(i + 1, self.train_rmses[-1]))
            if self.is_converged():
                print('The training process converged to a threshold.'); break
        print('... Final RMSE on training set: {}'.format(self.train_rmses[-1]))
        if self.test_purpose: 
            print('Test RMSE: {}'.format(self.compute_rmse(is_train=False)))
        predictions_df = self.get_predictions()
        return predictions_df

    # Override, since predict methods return format is different for this class
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
            predictions_df.at[i, 'Rating'] = self.predict(user, item)[0]
        return predictions_df

    # Override, since predict methods return format is different for this class
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
            error = get_rating(user, item) - self.predict(user, item)[0]
            mse += (error ** 2) 
        mse /= len(observed)
        return np.sqrt(mse) # rmse

    # Returns implicit effect and normalizer to make the train method faster
    def predict(self, user, item):
        """
        Predicts a rating for the specified user, item pair. Includes the baseline
        effect and the implicit effect.
        Args:
            user: The specified user
            item: The specified item
        Returns:
            prediction: The predicted rating for the specified user, item pair
            implicit_effect: The combined effect of implicit feedbacks of the 
                items that the specified user rated.
            normalizer: The normalizing coefficient used in the implicit_effect
                computation.
        """
        implicit_effect, normalizer = self.compute_implicit_effect(user)
        prediction = self.item_features[item].dot(self.user_features[user] + implicit_effect)
        prediction += self.global_bias + self.user_biases[user] + self.item_biases[item] 
        return prediction, implicit_effect, normalizer

    def compute_implicit_effect(self, user):
        """
        Computes the effect of implicit feedbacks of the items that a
        specified user rated.
        Args:
            user: The specified user
        Returns:
            mplicit_effect: The combined effect of implicit feedbacks of the 
                items that the specified user rated.
            normalizer: The normalizing coefficient used in the implicit_effect
                computation.
        """
        items = self.data.get_items_rated_by(user)
        normalizer = np.sqrt(len(items))
        implicit_effect = np.zeros(self.num_features)
        for item in items:
            implicit_effect += self.implicit_feedbacks[int(item)]
        if normalizer != 0:
            implicit_effect /= normalizer
        return implicit_effect, normalizer

    def init_implicit_feedbacks(self):
        """
        Initializes the item implicit feedback matrix.
        """
        self.implicit_feedbacks = np.random.normal(size=(self.data.num_items, self.num_features))

    def init_biases(self):
        """
        Initializes global, user and movie biases.
        """
        self.global_bias = self.data.global_mean
        self.user_biases = np.zeros(self.data.num_users, dtype=float) 
        self.item_biases = np.zeros(self.data.num_items, dtype=float) 

    def init_hyperparams(self):
        """
        Initializes the hyperparameters used in BSGD.
        """
        self.gamma = 0.005
        self.lambda_user = 0.01
        self.lambda_item = 0.01
        self.num_epochs = 10
        self.lambda_u_bias = 0.001
        self.lambda_i_bias = 0.001
        self.lambda_implicit = 0.005

# Testing
if __name__ == '__main__':
    model = MF_implicit(test_purpose=True)
    model.train()
    model.plot()