import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Dict
import pickle

class Preprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.categorical_features = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday']
        self.numerical_features = ['MinTemp', 'MaxTemp', 'Rainfall', 'WindGustSpeed', 'WindSpeed9am', 
                                   'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 
                                   'Pressure3pm', 'Temp9am', 'Temp3pm']
        
    def preprocess(self, data: pd.DataFrame, is_training: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess the data for TabNet"""
        df = data.copy()
            
        # Drop Date column
        df = df.drop('Date', axis=1)
        # Drop columns with too many NaN values > 30%
        df = df.drop(["Evaporation", "Sunshine", "Cloud9am", "Cloud3pm"], axis=1)

        # Drop the rest of the rows that contain NaN values
        df = df.dropna(how='any')
            
        # Encode target variable
        if 'RainTomorrow' in df.columns:
            y = df['RainTomorrow'].map({'No': 0, 'Yes': 1}).values
            df = df.drop('RainTomorrow', axis=1)
        else:
            y = None

        # Encode categorical features labels
        for feature in self.categorical_features:
            if feature in df.columns:
                if is_training:
                    # Encode labels as integers
                    self.label_encoders[feature] = LabelEncoder()
                    df[feature] = self.label_encoders[feature].fit_transform(df[feature].astype(str))
                else:
                    df[feature] = self.label_encoders[feature].transform(df[feature].astype(str))

        
        # return dataframe, datamatrix, target
        return df, df.to_numpy(), y
    
    def save(self, file_path: str) -> None:
        """Save the preprocessor state"""
        state = {
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'feature_means': getattr(self, 'feature_means', {}),
            'categorical_modes': getattr(self, 'categorical_modes', {})
        }
        with open(file_path, 'wb') as f:
            pickle.dump(state, f)
    
    @staticmethod
    def load(file_path: str):
        """Load a preprocessor state"""
        with open(file_path, 'rb') as f:
            state = pickle.load(f)
        
        preprocessor = Preprocessor()
        preprocessor.label_encoders = state['label_encoders']
        preprocessor.scaler = state['scaler']
        preprocessor.feature_means = state['feature_means']
        preprocessor.categorical_modes = state['categorical_modes']
        return preprocessor

