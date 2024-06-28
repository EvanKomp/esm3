# test_latent_opt/tools/training.py
'''
* Author: Evan Komp
* Created: 6/27/2024
* Company: Bottle Institute @ National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT

Functions for training the latent space regressor
'''
import torch
from datasets import DatasetDict
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm
sns.set_style('whitegrid')
sns.set_context('talk')

from tools.data_loader import LatentOptData, LatentOptDataLoader

import logging
logger = logging.getLogger(__name__)


class TrainingCallback:
    def __init__(self, early_stopping: bool=False, patience=0, threshold=0.0):
        self.early_stopping = early_stopping
        self.patience = patience
        self.threshold = threshold
        self.counter = 0
        self.best_score = None
        self.stop = False
        self.train_history = []
        self.val_history = []

    def __call__(self, train_score, val_score=None):
        self.train_history.append(train_score)
        self.val_history.append(val_score)
        if self.early_stopping:
            if self.best_score is None:
                self.best_score = val_score
            elif val_score < self.best_score - self.threshold:
                self.counter += 1
                if self.counter >= self.patience:
                    self.stop = True
            else:
                self.best_score = val_score
                self.counter = 0
        
def train_model(
    train_loader,
    model,
    validation_loader=None,
    optimizer_name: str='adam',
    lr: float=1e-3,
    criterion_name: str='mse',
    epochs: int=100,
    early_stopping: bool=True,
    early_stopping_patience: int=5,
    early_stopping_threshold: float=0.0,        
):
    model = model
    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        raise ValueError(f'Optimizer {optimizer_name} not supported')
    
    if criterion_name == 'mse':
        criterion = torch.nn.MSELoss()
    else:
        raise ValueError(f'Criterion {criterion_name} not supported')
    
    callback = TrainingCallback(
        early_stopping=early_stopping,
        patience=early_stopping_patience,
        threshold=early_stopping_threshold
    )
    step = 0
    for epoch in range(epochs):
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            model.train()
            # note that X has zero padding which is labeled with mask.
            # this needs to be passed to the model to handle it properly
            X = batch['X']
            mask = batch['mask']
            y = batch['y']

            optimizer.zero_grad()

            outputs = model(X, mask)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            step += 1

            if validation_loader is not None:
                val_loss = test_model(validation_loader, model, criterion)
            callback(train_score=loss.item(), val_score=val_loss)
            if callback.stop:
                logger.info(f'Early stopping at step {step}')
                return model, callback
    return model, callback

def test_model(test_loader, model, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing'):
            X = batch['X']
            mask = batch['mask']
            y = batch['y']

            outputs = model(X, mask)
            loss = criterion(outputs, y)
            total_loss += loss.item()
    return total_loss / len(test_loader)

def cross_validation(
    data_path,
    model,
    splits=5,
    epochs=10,
    batch_size=32,
    lr=1e-3,
    early_stopping_patience=5,
    early_stopping_threshold=0.0,
    criterion_name='mse',
    optimizer_name='adam',
    model_kwargs={},
):
    # load huggingface dataset dict from disk
    hf_dataset = DatasetDict.load_from_disk(data_path)
    results = []
    
    for split_idx in range(splits):
        # concatenate the other splits
        other_splits = [str(i) for i in range(splits) if i != split_idx]
        train_loader = LatentOptDataLoader(
            LatentOptData(hf_dataset, other_splits),
            batch_size=batch_size,
            shuffle=True
        )
        test_loader = LatentOptDataLoader(
            LatentOptData(hf_dataset, [str(split_idx)]),
            batch_size=batch_size,
            shuffle=False
        )
        model = model(**model_kwargs)
        _, callback = train_model(
            train_loader,
            model,
            validation_loader=test_loader,
            lr=lr,
            epochs=epochs,
            early_stopping_patience=early_stopping_patience,
            early_stopping_threshold=early_stopping_threshold,
            criterion_name=criterion_name,
            optimizer_name=optimizer_name,
        )
        results.append(callback)
    
    # compute the mean metric over CVs and also create a training plot for each
    cv_score = sum([r.val_history[-1] for r in results]) / len(results)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    df_data = []
    for i, r in enumerate(results):
        for j, score in enumerate(r.val_history):
            df_data.append({'split': i, 'score': score, 'type': 'val'})
        for j, score in enumerate(r.train_history):
            df_data.append({'split': i, 'score': score, 'type': 'train', 'step': j})
    df = pd.DataFrame(df_data)
    sns.lineplot(data=df, x='step', y='score', hue='split', style='type', ax=ax)
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')

    return cv_score, fig




