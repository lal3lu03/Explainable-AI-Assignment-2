import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple

class DataLoader:
    def __init__(self, data_path: str):
        self.data_path = data_path
        
    def load_data(self) -> pd.DataFrame:
        """Load the weather dataset"""
        return pd.read_csv(self.data_path)
    
    def split_data(self, data: pd.DataFrame, test_size: float = 0.1, val_size: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train, validation and test sets"""
        
        # First split: separate test set
        train_val, test = train_test_split(data, test_size=test_size, random_state=42)
        
        # Second split: separate validation set from training set
        val_ratio = val_size / (1 - test_size)
        train, val = train_test_split(train_val, test_size=val_ratio, random_state=42)
        
        return train, val, test
