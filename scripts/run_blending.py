import numpy as np
from models.baselines import Baselines
from models.MF_SGD import MF_SGD
from models.MF_BSGD import MF_BSGD
from models.MF_ALS import MF_ALS
from models.surprise_models import SurpriseModels
from blending import Blending
from data_related.data import Data

def main():
    np.random.seed(98) # to be able to reproduce the results

    data = Data(test_purpose=True)

    models = {}

    baselines = Baselines(data=data, test_purpose=True)
    
    print('\nModelling using baseline_global_mean:')
    models['baseline_global_mean'] = baselines.baseline_global_mean()['Rating']
    
    print('\nModelling using baseline_user_mean:')
    models['baseline_user_mean'] = baselines.baseline_user_mean()['Rating']

    print('\nModelling using baseline_movie_mean:')
    models['baseline_item_mean'] = baselines.baseline_item_mean()['Rating']
    
    mf_sgd = MF_SGD(data=data, test_purpose=True)

    print('\nModelling using MF_SGD:')
    models['mf_sgd'] = mf_sgd.train()['Rating']

    mf_bsgd = MF_BSGD(data=data, test_purpose=True)

    print('\nModelling using MF_BSGD:')
    models['mf_bsgd'] = mf_bsgd.train()['Rating']
    
    mf_als = MF_ALS(data=data, test_purpose=True)

    print('\nModelling using MF_ALS:')
    models['mf_als'] = mf_als.train()['Rating']
    
    surprise_models = SurpriseModels(data=data, test_purpose=True)

    print('\nModelling using user based Surprise kNN Means:')
    models['surprise_kNN_means_user'] = surprise_models.kNN_means(k=100,
                                                                  sim_options={'name': 'pearson_baseline',
                                                                               'user_based': True})['Rating']

    print('\nModelling using item based Surprise kNN Means:')
    models['surprise_kNN_means_item'] = surprise_models.kNN_means(k=300,
                                                                  sim_options={'name': 'pearson_baseline',
                                                                               'user_based': False})['Rating']
                                                                                     
    print('\nModelling using Surprise SlopeOne:')
    models['surprise_slope_one'] = surprise_models.slope_one()['Rating']
    
    print('\nModelling using Surprise Co-Clustering:')
    models['surprise_co_clustering'] = surprise_models.co_clustering()['Rating']
    
    blending = Blending(models, data.test_df['Rating'])

    print('\nModelling using weighted averaging of the previous models.')
    optimal_weights = blending.optimize_weighted_average()
    print('\nOptimal weights: ', optimal_weights)
    
if __name__ == '__main__':
    main()