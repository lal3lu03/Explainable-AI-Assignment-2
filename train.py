from src.data.loader import DataLoader
from src.data.preprocessing import Preprocessor
from src.models.tabnet_model import RainPredictor
from pytorch_tabnet.augmentations import ClassificationSMOTE
import json
import logging
import pickle  # For saving the preprocessor
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

categorical_features = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday']
numerical_features = ['MinTemp', 'MaxTemp', 'Rainfall', 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 
                      'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Temp9am', 'Temp3pm']

def calculate_metrics(y_true, y_pred, y_prob):
    """Calculate classification metrics"""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'auc_roc': roc_auc_score(y_true, y_prob[:, 1]),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred)
    }

def train():
    # Load configuration
    with open('config.json', 'r') as f:
        config = json.load(f)

    # Initialize components with config parameters
    data_loader = DataLoader(config['data_path'])
    preprocessor = Preprocessor()
    
    # Load and split data
    logger.info("Loading data...")
    data = data_loader.load_data()
    train_df, val_df, test_df = data_loader.split_data(data)
    
    # Preprocess data
    logger.info("Preprocessing data...")
    df_train, X_train, y_train = preprocessor.preprocess(train_df, is_training=True)  # Use instance method
    df_val, X_val, y_val = preprocessor.preprocess(val_df, is_training=False)  # Use instance method
    df_test, X_test, y_test = preprocessor.preprocess(test_df, is_training=False)  # Use instance method

    cat_idxs = [f for f, feature in enumerate(df_train.columns) if feature in categorical_features]
    cat_dims = [len(preprocessor.label_encoders[feature].classes_) for feature in df_train.columns if feature in categorical_features]

    config['model_params']["cat_idxs"] = cat_idxs
    config['model_params']["cat_dims"] = cat_dims
    
    # Train model with config parameters
    logger.info("Training model...")
    model = RainPredictor(config['model_params'])
    aug = ClassificationSMOTE(p=0.2)
    model.fit(
        X_train, y_train,
        X_val, y_val,
        num_epochs=config['num_epochs'],
        batch_size=config['batch_size'],
        virtual_batch_size=config['virtual_batch_size'],
        augmentations=aug
    )
    
    # Make predictions
    logger.info("Making predictions...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred, y_prob)
    
    # Log metrics
    logger.info("Model performance:")
    for metric_name, value in metrics.items():
        logger.info(f"{metric_name}: {value:.4f}")
        
    # Save metrics
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)

    # Save model using the path from config
    logger.info("Saving model...")
    model.save_model(config['model_path'])
    
    # Save preprocessor using the path from config
    preprocessor.save(config['preprocessor_path'])

if __name__ == "__main__":
    train()
