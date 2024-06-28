# test_latent_opt/train_latent_opt_cv.py
'''
* Author: Evan Komp
* Created: 6/27/2024
* Company: Bottle Institute @ National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT

Train a small regressor on the embeddings from ESM3 and see if it can predict P740.
Do so in CV
'''

import torch
from datasets import DatasetDict
import tools.torch_modules  # Assuming MyModel is the regressor you want to use
from tools.data_loader import LatentOptData, LatentOptDataLoader
from tools.training cross_validation
import numpy as np

import dvc

dvc_PARAMS = dvc.api.get_params()

PARAMS = {
    'model': dvc_PARAMS['latent_model']['module'],
    'batch_size': dvc_PARAMS['latent_model']['batch_size'],
    'lr': dvc_PARAMS['latent_model']['lr'],
    'epochs': dvc_PARAMS['latent_model']['epochs'],
    'early_stopping_patience': dvc_PARAMS['latent_model']['early_stopping_patience'],
    'early_stopping_threshold': dvc_PARAMS['latent_model']['early_stopping_threshold'],
    'dropout': dvc_PARAMS['latent_model']['dropout'],
    'optimizer': dvc_PARAMS['latent_model']['optimizer'],

}

if __name__ == '__main__':
    model_module = getattr(tools.torch_modules, PARAMS['model'])
    
    cv_score, fig = cross_validation(
        data_path=os.path.join('data', 'embedded_data'),
        model=model_module,
        splits=5,
        epochs=PARAMS['epochs'],
        batch_size=PARAMS['batch_size'],
        lr=PARAMS['lr'],
        early_stopping_patience=PARAMS['early_stopping_patience'],
        early_stopping_threshold=PARAMS['early_stopping_threshold'],
        criterion_name='mse',
        optimizer_name=PARAMS['optimizer'],
        model_kwargs={'dropout': PARAMS['dropout']},
    )
