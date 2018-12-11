import numpy as np
from baselines import Baselines
from MF_SGD import MF_SGD
from MF_BSGD import MF_BSGD
from MF_ALS import MF_ALS
from surprise_models import SurpriseModels
from blending import Blending
from data import Data
from data_processing import create_submission

PREDICTIONS_FILENAME = 'Datasets/mixed_model.csv'

OPTIMAL_WEIGHTS = [0.05448098, -0.12085548, 0.00152701, 0.05366179, 0.01066892, 
    -0.01574904, -0.09340066, 0.04176627, 0.07477889, 0.8849526,
    0.05250151, 0.06863501, -0.01624362] # RMSE = 1.0438135875533157 from last run_blending

def main():
    np.random.seed(98) # to be able to reproduce the results

    data = Data()

    models = {'baseline_global_mean': None,
        'baseline_user_mean': None,
        'baseline_item_mean': None,
        'mf_sgd': None,
        'mf_bsgd': None, 
        'mf_als': None,
        'surprise_kNN_baseline_user': None,
        'surprise_kNN_baseline_item': None,
        'surprise_slope_one': None,
        'surprise_co_clustering': None}

    baselines = Baselines(data=data)
    
    print('\nModelling using baseline_global_mean:')
    models['baseline_global_mean'] = baselines.baseline_global_mean()['Rating']
    
    print('\nModelling using baseline_user_mean:')
    models['baseline_user_mean'] = baselines.baseline_user_mean()['Rating']

    print('\nModelling using baseline_movie_mean:')
    models['baseline_item_mean'] = baselines.baseline_item_mean()['Rating']
    
    mf_sgd = MF_SGD(data=data)

    print('\nModelling using MF_SGD:')
    models['mf_sgd'] = mf_sgd.train()['Rating']

    mf_bsgd = MF_BSGD(data=data)

    print('\nModelling using MF_BSGD:')
    models['mf_bsgd'] = mf_bsgd.train()['Rating']
    
    mf_als = MF_ALS(data=data)

    print('\nModelling using MF_ALS:')
    models['mf_als'] = mf_als.train()['Rating']
    
    surprise_models = SurpriseModels(data=data)
    
    print('\nModelling using user based Surprise kNN Baseline:')
    models['surprise_kNN_baseline_user'] = surprise_models.kNN_baseline(k=50, 
        sim_options={'name': 'cosine', 'user_based': True})['Rating']

    print('\nModelling using item based Surprise kNN Baseline:')
    models['surprise_kNN_baseline_item'] = surprise_models.kNN_baseline(k=100, 
        sim_options={'name': 'pearson_baseline', 'user_based': False})['Rating']

    print('\nModelling using Surprise SlopeOne:')
    models['surprise_slope_one'] = surprise_models.slope_one()['Rating']
    
    #print('\nModelling using Surprise SVD:')
    #models.append(surprise_models.SVD()['Rating'])
    
    #print('\nModelling using Surprise SVD++:')
    #models.append(surprise_models.SVDpp()['Rating'])
    
    print('\nModelling using Surprise Co-Clustering:')
    models['surprise_co_clustering'] = surprise_models.co_clustering()['Rating']
    
    blending = Blending(models, data.test_df['Rating'], OPTIMAL_WEIGHTS)

    print('\nModelling using weighted averaging of the previous models.')
    mixed_model = blending.get_weighted_average()

    data.test_df['Rating'] = mixed_model
    print('\nCreating mixed_model.csv ...')
    create_submission(data.test_df, PREDICTIONS_FILENAME)
    print('... mixed_model.csv created.')
    
if __name__ == '__main__':
    main()