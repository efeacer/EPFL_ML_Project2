from baselines import Baselines
from MF_SGD import MF_SGD
from MF_BSGD import MF_BSGD
from MF_ALS import MF_ALS
from surprise_models import SurpriseModels
from blending import Blending
from data import Data

def main():
    data = Data(test_purpose=True)
    
    models = [] 

    baselines = Baselines(data=data, test_purpose=True)
    
    print('\nModelling using baseline_global_mean:')
    models.append(baselines.baseline_global_mean()['Rating'])
    
    print('\nModelling using baseline_user_mean:')
    models.append(baselines.baseline_user_mean()['Rating'])

    print('\nModelling using baseline_movie_mean:')
    models.append(baselines.baseline_movie_mean()['Rating'])
    
    mf_sgd = MF_SGD(data=data, test_purpose=True)

    print('\nModelling using MF_SGD:')
    models.append(mf_sgd.train()['Rating'])

    mf_bsgd = MF_BSGD(data=data, test_purpose=True)

    print('\nModelling using MF_BSGD:')
    models.append(mf_bsgd.train()['Rating'])
    
    mf_als = MF_ALS(data=data, test_purpose=True)

    print('\nModelling using MF_ALS:')
    models.append(mf_als.train()['Rating'])
    
    surprise_models = SurpriseModels(data=data, test_purpose=True)
    
    print('\nModelling using user based Surprise kNN Baseline:')
    models.append(surprise_models.kNN_baseline(k=50, 
        sim_options={'name': 'cosine', 'user_based': True})['Rating'])

    print('\nModelling using item based Surprise kNN Baseline:')
    models.append(surprise_models.kNN_baseline(k=100, 
        sim_options={'name': 'pearson_baseline', 'user_based': False})['Rating'])

    print('\nModelling using Surprise SlopeOne:')
    models.append(surprise_models.slope_one()['Rating'])
    
    #print('\nModelling using Surprise SVD:')
    #models.append(surprise_models.SVD()['Rating'])
    
    #print('\nModelling using Surprise SVD++:')
    #models.append(surprise_models.SVDpp()['Rating'])
    
    print('\nModelling using Surprise Co-Clustering:')
    models.append(surprise_models.co_clustering()['Rating'])
    
    blending = Blending(models, data.test_df['Rating'])

    print('\nModelling using weighted averaging of the previous models.')
    optimal_weights = blending.optimize_weighted_average()
    print('\nOptimal weights: ', optimal_weights)
    
if __name__ == '__main__':
    main()