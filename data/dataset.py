import torch
from torchvision import transforms

from data.income import Income
from data.cancellation import Cancellation
from data.nps import Nps
from data.drybean import Drybean
from data.covtype import Covtype
from data.wine import Wine

def get_meta_dataset(P, dataset, only_test=False):

    if dataset == 'income':
        meta_train_dataset = Income(tabular_size = 115,
                                    seed=P.seed,
                                    source='train',
                                    shot=P.num_shots,
                                    tasks_per_batch=P.batch_size,
                                    test_num_way = P.num_ways,
                                    query = P.num_shots_test)

        meta_val_dataset = Income(tabular_size = 115,
                                    seed=P.seed,
                                    source='val',
                                    shot=1, 
                                    tasks_per_batch=P.test_batch_size,
                                    test_num_way = 2,
                                    query = 30)
        
    elif dataset == 'cancellation':
        meta_train_dataset = Cancellation(tabular_size = 23,
                                    seed=P.seed,
                                    source='train',
                                    shot=P.num_shots,
                                    tasks_per_batch=P.batch_size,
                                    test_num_way = P.num_ways,
                                    query = P.num_shots_test)

        meta_val_dataset = Cancellation(tabular_size = 23,
                                    seed=P.seed,
                                    source='val',
                                    shot=1, 
                                    tasks_per_batch=P.test_batch_size,
                                    test_num_way = 2,
                                    query = 30)
        
    elif dataset == 'nps':
        meta_train_dataset = Nps(tabular_size = 24,
                                    seed=P.seed,
                                    source='train',
                                    shot=P.num_shots,
                                    tasks_per_batch=P.batch_size,
                                    test_num_way = P.num_ways,
                                    query = P.num_shots_test)

        meta_val_dataset = Nps(tabular_size = 24,
                                    seed=P.seed,
                                    source='val',
                                    shot=1, 
                                    tasks_per_batch=P.test_batch_size,
                                    test_num_way = 3,
                                    query = 30)
        
    elif dataset == 'drybean':
        meta_train_dataset = Drybean(tabular_size = 22,
                                    seed=P.seed,
                                    source='train',
                                    shot=P.num_shots,
                                    tasks_per_batch=P.batch_size,
                                    test_num_way = P.num_ways,
                                    query = P.num_shots_test)

        meta_val_dataset = Drybean(tabular_size = 22,
                                    seed=P.seed,
                                    source='val',
                                    shot=1, 
                                    tasks_per_batch=P.test_batch_size,
                                    test_num_way = 7,
                                    query = 30)
        
    elif dataset == 'covtype':
        meta_train_dataset = Covtype(tabular_size = 62,
                                    seed=P.seed,
                                    source='train',
                                    shot=P.num_shots,
                                    tasks_per_batch=P.batch_size,
                                    test_num_way = P.num_ways,
                                    query = P.num_shots_test)

        meta_val_dataset = Covtype(tabular_size = 62,
                                    seed=P.seed,
                                    source='val',
                                    shot=1, 
                                    tasks_per_batch=P.test_batch_size,
                                    test_num_way = 7,
                                    query = 30)
        
    elif dataset == 'wine':
        meta_train_dataset = Wine(tabular_size = 17,
                                    seed=P.seed,
                                    source='train',
                                    shot=P.num_shots,
                                    tasks_per_batch=P.batch_size,
                                    test_num_way = P.num_ways,
                                    query = P.num_shots_test)

        meta_val_dataset = Wine(tabular_size = 17,
                                    seed=P.seed,
                                    source='val',
                                    shot=1, 
                                    tasks_per_batch=P.test_batch_size,
                                    test_num_way = 3,
                                    query = 6)

    else:
        raise NotImplementedError()

    return meta_train_dataset, meta_val_dataset
