import numpy as np
from models.baselines import Baselines
from models.MF_SGD import MF_SGD
from models.MF_BSGD import MF_BSGD
from models.MF_ALS import MF_ALS
from models.surprise_models import SurpriseModels
from blending import Blending
from data_related.data import Data
from data_related.data_processing import create_submission

PREDICTIONS_FILENAME = 'datasets/final_submission.csv'

OPTIMAL_WEIGHTS = {'baseline_global_mean': 0.4042972983445468, 
                   'baseline_user_mean': -0.42286968808430825,
                   'baseline_item_mean': -0.31372914406183444,
                   'mf_sgd': -0.1746699117090964, 
                   'mf_bsgd': 0.3454789282745774,
                   'mf_als': 0.7377406202924518,
                   'surprise_kNN_means_user': 0.25695648098796964,
                   'surprise_kNN_means_item': 0.3478280128242141, 
                   'surprise_slope_one': -0.17666897047712135, 
                   'surprise_co_clustering': 0.003294142281703342}

def main():
    np.random.seed(98) # to be able to reproduce the results

    data = Data()

    models = {}

    baselines = Baselines(data=data)

    print('\nModelling using baseline_global_mean ...')
    models['baseline_global_mean'] = baselines.baseline_global_mean()['Rating']
    print('... done.')

    print('\nModelling using baseline_user_mean ...')
    models['baseline_user_mean'] = baselines.baseline_user_mean()['Rating']
    print('... done.')

    print('\nModelling using baseline_movie_mean ...')
    models['baseline_item_mean'] = baselines.baseline_item_mean()['Rating']
    print('... done.')
    
    mf_sgd = MF_SGD(data=data)

    print('\nModelling using MF_SGD ...')
    models['mf_sgd'] = mf_sgd.train()['Rating']
    print('... done.')

    mf_bsgd = MF_BSGD(data=data)

    print('\nModelling using MF_BSGD ...')
    models['mf_bsgd'] = mf_bsgd.train()['Rating']
    print('... done.')

    mf_als = MF_ALS(data=data)

    print('\nModelling using MF_ALS ...')
    models['mf_als'] = mf_als.train()['Rating']
    print('... done.')
    
    surprise_models = SurpriseModels(data=data)
    
    print('\nModelling using user based Surprise kNN Means ...')
    models['surprise_kNN_means_user'] = surprise_models.kNN_means(k=100,
                                                                  sim_options={'name': 'pearson_baseline',
                                                                               'user_based': True})['Rating']
    print('... done.')

    print('\nModelling using item based Surprise kNN Means ...')
    models['surprise_kNN_means_item'] = surprise_models.kNN_means(k=300,
                                                                  sim_options={'name': 'pearson_baseline',
                                                                               'user_based': False})['Rating']
    print('... done.')

    print('\nModelling using Surprise SlopeOne ...')
    models['surprise_slope_one'] = surprise_models.slope_one()['Rating']
    print('... done.')

    print('\nModelling using Surprise CoClustering ...')
    models['surprise_co_clustering'] = surprise_models.co_clustering()['Rating']
    print('... done.')

    blending = Blending(models, data.test_df['Rating'], OPTIMAL_WEIGHTS)

    print('\nModelling using weighted averaging of the previous models ...')
    mixed_model = blending.get_weighted_average()
    print('... done.')

    data.test_df['Rating'] = mixed_model
    print('\nCreating final_submission.csv ...')
    create_submission(data.test_df, PREDICTIONS_FILENAME)
    print('... final_submission.csv created.')
    
if __name__ == '__main__':
    main()