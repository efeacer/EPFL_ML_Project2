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

OPTIMAL_WEIGHTS = {'baseline_global_mean': 0.28820931,
                   'baseline_user_mean': -0.31216191,
                   'baseline_item_mean': -0.16738859,
                   'mf_sgd': -0.37163926,
                   'mf_bsgd': 0.53760117, 
                   'mf_als': 0.69963825,
                   'surprise_kNN_baseline_user': 0.0310014,
                   'surprise_kNN_baseline_item': 0.40249769,
#                   'surprise_SVD': None,
#                   'surprise_SVDpp': None,
                   'surprise_slope_one': -0.14927565,
                   'surprise_co_clustering': 0.04481663}

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
#              'surprise_SVD': None,
#              'surprise_SVDpp': None,
              'surprise_slope_one': None,
              'surprise_co_clustering': None}

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
    models['surprise_kNN_baseline_user'] = surprise_models.kNN_baseline(k=150, 
                                                                        sim_options={'name': 'cosine',
                                                                                     'user_based': True})['Rating']
    print('... done')

    print('\nModelling using item based Surprise kNN Baseline ...')
    models['surprise_kNN_baseline_item'] = surprise_models.kNN_baseline(k=150, 
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