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

OPTIMAL_WEIGHTS = {'baseline_global_mean': 0.40687204452635317,
                   'baseline_user_mean': -0.42577449917982385,
                   'baseline_item_mean': -0.3165107607916656, 
#                   'baseline_global_median': None,
#                   'baseline_user_median': None,
#                   'baseline_item_median': None,
                   'mf_sgd': -0.1708316741679142,
                   'mf_bsgd': 0.3462436054039932, 
                   'mf_als': 0.7375480510708262, 
                   'surprise_kNN_baseline_user': 0.25714575898297704,
                   'surprise_kNN_baseline_item': 0.34780294190565625,
#                   'surprise_SVD': None,
#                   'surprise_SVDpp': None,
                   'surprise_slope_one': -0.1823488497149787, 
                   'surprise_co_clustering': 0.007572666695397467}

def main():
    np.random.seed(98) # to be able to reproduce the results

    data = Data()

    models = {}

    baselines = Baselines(data=data)

    print('\nModelling using baseline_global_mean ...')
    models['baseline_global_mean'] = baselines.baseline_global_mean()['Rating']
    print('... done')

    print('\nModelling using baseline_user_mean ...')
    models['baseline_user_mean'] = baselines.baseline_user_mean()['Rating']
    print('... done')

    print('\nModelling using baseline_movie_mean ...')
    models['baseline_item_mean'] = baselines.baseline_item_mean()['Rating']
    print('... done')

    print('\nModelling using baseline_global_median ...')
    models['baseline_global_median'] = baselines.baseline_global_median()['Rating']
    print('... done')

    print('\nModelling using baseline_user_median ...')
    models['baseline_user_median'] = baselines.baseline_user_median()['Rating']
    print('... done')

    print('\nModelling using baseline_movie_median ...')
    models['baseline_item_median'] = baselines.baseline_item_median()['Rating']
    print('... done')
    
    mf_sgd = MF_SGD(data=data)

    print('\nModelling using MF_SGD ...')
    models['mf_sgd'] = mf_sgd.train()['Rating']
    print('... done')

    mf_bsgd = MF_BSGD(data=data)

    print('\nModelling using MF_BSGD ...')
    models['mf_bsgd'] = mf_bsgd.train()['Rating']
    print('... done')
    
    mf_als = MF_ALS(data=data)

    print('\nModelling using MF_ALS ...')
    models['mf_als'] = mf_als.train()['Rating']
    print('... done')
    
    surprise_models = SurpriseModels(data=data)
    
    print('\nModelling using user based Surprise kNN Baseline ...')
    models['surprise_kNN_baseline_user'] = surprise_models.kNN_baseline(k=100, 
                                                                        sim_options={'name': 'pearson_baseline',
                                                                                     'user_based': True})['Rating']
    print('... done')

    print('\nModelling using item based Surprise kNN Baseline ...')
    models['surprise_kNN_baseline_item'] = surprise_models.kNN_baseline(k=200, 
                                                                        sim_options={'name': 'pearson_baseline',
                                                                                     'user_based': False})['Rating']
    print('... done')

    print('\nModelling using Surprise SlopeOne ...')
    models['surprise_slope_one'] = surprise_models.slope_one()['Rating']
    print('... done')

    #print('\nModelling using Surprise SVD ...')
    #models['surprise_SVD'] = surprise_models.SVD()['Rating']
    #print('... done')

    #print('\nModelling using Surprise SVD++ ...')
    #models['surprise_SVDpp'] = surprise_models.SVDpp()['Rating']
    #print('... done')

    print('\nModelling using Surprise Co-Clustering ...')
    models['surprise_co_clustering'] = surprise_models.co_clustering()['Rating']
    print('... done')

    blending = Blending(models, data.test_df['Rating'], OPTIMAL_WEIGHTS)

    print('\nModelling using weighted averaging of the previous models ...')
    mixed_model = blending.get_weighted_average()
    print('... done')

    data.test_df['Rating'] = mixed_model
    print('\nCreating mixed_model.csv ...')
    create_submission(data.test_df, PREDICTIONS_FILENAME)
    print('... mixed_model.csv created.')
    
if __name__ == '__main__':
    main()