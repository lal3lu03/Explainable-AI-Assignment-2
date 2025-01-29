import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from typing import Tuple, Dict
import numpy as np

class RainPredictor:
    def __init__(self, model_params):
        self.model = TabNetClassifier(
            n_d=model_params['n_d'],
            n_a=model_params['n_a'],
            n_steps=model_params['n_steps'],
            gamma=model_params['gamma'],
            n_independent=model_params['n_independent'],
            n_shared=model_params['n_shared'],
            lambda_sparse=model_params['lambda_sparse'],
            momentum=model_params['momentum'],
            clip_value=model_params['clip_value'],
            optimizer_fn=torch.optim.Adam,
            optimizer_params={'lr': model_params['learning_rate']},
            scheduler_params={
                'mode': "min",
                'patience': 5,
                'min_lr': 1e-5,
                'factor': 0.5
            },
            scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
            epsilon=model_params['epsilon']
        )
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            X_valid: np.ndarray, y_valid: np.ndarray, 
            num_epochs: int, batch_size: int, virtual_batch_size: int, augmentations=None) -> None:
        """Train the model"""
        self.model.fit(
            X_train=X_train, y_train=y_train,
            eval_set=[(X_valid, y_valid)],
            eval_name=['valid'],
            max_epochs=num_epochs,
            patience=20,
            batch_size=batch_size,
            virtual_batch_size=virtual_batch_size,
            num_workers=0,
            weights=1,
            drop_last=False,
            augmentations=augmentations
        )
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probability predictions"""
        return self.model.predict_proba(X)
    
    def save_model(self, model_path: str) -> None:
        """Save the trained model"""
        self.model.save_model(model_path)
    
    def load_model(self, model_path: str) -> None:
        """Load a trained model"""
        self.model.load_model(model_path)
