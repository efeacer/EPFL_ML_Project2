from surprise import Reader, Dataset, KNNBaseline, SlopeOne, SVD, SVDpp
from surprise import CoClustering, KNNWithMeans
from loss_functions import compute_rmse
from data import Data

class SurpriseModels:
    """
    A class that encapsulates the Surprise Library Models and adapts them
    to the interface expected by the callers in the project. Big thanks
    to the author of the Surprise Library, Nicolas Hug.
    """

    def __init__(self, data=None, test_purpose=False):
        """
        Constructor for the SurpriseModels class. Initializes the internal
        training and test matrices in the Surprise Library's own format.
        Args:
            test_purpose: True for testing, False for creating submission
        """
        """
        Constructor for the SurpriseModels class. Initializes the internal
        training and test matrices in the Surprise Library's own format.
        Args:
            train_df: Training matrix in the Pandas DataFrame format
            test_df: Test matrix in the Pandas DataFrame format
            test_purpose: A flag to indicate that the algorithm is running
                on a test set with given ratings, not a test set with unknown
                ratings
        """
        if data is not None:
            self.data = data
        else:
            self.data = Data(test_purpose=test_purpose)
        self.test_purpose = test_purpose
        self.train_data = Dataset.load_from_df(self.data.train_df[['User', 'Item', 'Rating']], 
            rating_scale=(1, 5)).build_full_trainset()
        self.test_data = [(x['User'], x['Item'], x['Rating']) 
            for _, x in self.data.test_df.iterrows()]
 
    def kNN_baseline(self, k, sim_options):
        """
        k Nearest Negihbors collaborative filtering algorithm taking into
        account a baseline rating. 
        Args:
            k: Number of Nearest Neighbors for kNN
            sim_options: A dictionary of options for the similarity measure
        Returns:
            predictions_df: The predictions of the model on the test data in
                Pandas Data Frame format
        """
        algorithm = KNNWithMeans(k=k, sim_options=sim_options)
        predictions = algorithm.fit(self.train_data).test(self.test_data)
        predictions_df = self.data.test_df.copy()
        predictions_df['Rating'] = [x.est for x in predictions]
        if self.test_purpose: 
            self.evalueate_model(predictions_df['Rating'], 'Surprise kNN_baseline')
        return predictions_df

    def slope_one(self):
        """
        SlopeOne to reflect how much one item is liked over than another.
        Returns:
            predictions_df: The predictions of the model on the test data in
                Pandas Data Frame format
        """
        algorithm = SlopeOne()
        predictions = algorithm.fit(self.train_data).test(self.test_data)
        predictions_df = self.data.test_df.copy()
        predictions_df['Rating'] = [x.est for x in predictions]
        if self.test_purpose: 
            self.evalueate_model(predictions_df['Rating'], 'Surprise slope_one')
        return predictions_df

    def SVD(self, n_factors=20, n_epochs=30, lr_all=0.001, reg_all=0.001):
        """
        Singular Value Decomposition approach to Matrix Factorization.
        Args:
            n_factors: Number of latent features, factors
            n_epochs: Number of iterations of the optimization loop
            lr_all: The learning rate for all parameters
            reg_all: The regularization term for all parameters
        Returns:
            predictions_df: The predictions of the model on the test data in
                Pandas Data Frame format
        """
        algorithm = SVD(n_factors=n_factors, n_epochs=n_epochs, lr_all=lr_all, 
                        reg_all=reg_all)
        predictions = algorithm.fit(self.train_data).test(self.test_data)
        predictions_df = self.data.test_df.copy()
        predictions_df['Rating'] = [x.est for x in predictions]
        if self.test_purpose: 
            self.evalueate_model(predictions_df['Rating'], 'Surprise SVD')
        return predictions_df

    def SVDpp(self, n_factors=20, n_epochs=20, lr_all=0.005, reg_all=0.02):
        """
        An extension of Singular Value Decomposition algorithm that takes
        implicit ratings into account.
        Args:
            n_factors: Number of latent features, factors
            n_epochs: Number of iterations of the optimization loop
            lr_all: The learning rate for all parameters
            reg_all: The regularization term for all parameters
        Returns:
            predictions_df: The predictions of the model on the test data in
                Pandas Data Frame format
        """
        algorithm = SVDpp(n_factors=n_factors, n_epochs=n_epochs, lr_all=lr_all, 
                          reg_all=reg_all)
        predictions = algorithm.fit(self.train_data).test(self.test_data)
        predictions_df = self.data.test_df.copy()
        predictions_df['Rating'] = [x.est for x in predictions]
        if self.test_purpose: 
            self.evalueate_model(predictions_df['Rating'], 'Surprise SVDpp')
        return predictions_df

    def co_clustering(self, n_cltr_u=3, n_cltr_i=3, n_epochs=20):
        """
        k Nearest Negihbors collaborative filtering algorithm taking into
        account a baseline rating. 
        Args:
            n_cltr_u: Number of user clusters
            n_cltr_i: Number of item clusters
            n_epochs: Number of iteration of the optimization loop
        Returns:
            predictions_df: The predictions of the model on the test data in
                Pandas Data Frame format
        """
        algorithm = CoClustering(n_cltr_u=n_cltr_u, n_cltr_i=n_cltr_i,
             n_epochs=n_epochs)
        predictions = algorithm.fit(self.train_data).test(self.test_data)
        predictions_df = self.data.test_df.copy()
        predictions_df['Rating'] = [x.est for x in predictions]
        if self.test_purpose: 
            self.evalueate_model(predictions_df['Rating'], 'Surprise co_clustering')
        return predictions_df

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