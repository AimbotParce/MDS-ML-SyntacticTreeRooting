
import pandas as pd
from lib.gridsearch import GridSearch, Dimension
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import json
import time
from typing import Dict, Any, List
import joblib
import logging
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score 
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="(%(asctime)s) %(levelname)s # %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def one_hot_encode(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode the 'language' column in the DataFrame.
    """
    return pd.get_dummies(df, columns=["language"], prefix="", prefix_sep="", drop_first=False)

cross_validation_folds = 5


def run_config_fold(
    configuration: Dict[str, Any],
    fold: int,
    fold_path: Path,
    train_fold_data: pd.DataFrame,
    validation_fold_data: pd.DataFrame,
) -> Dict[str, float]:

    # One-hot encode the training and validation data
    X_train = one_hot_encode(train_fold_data.drop(columns=["row_index", "node", "is_root"]))
    y_train = train_fold_data["is_root"]
    X_val = one_hot_encode(validation_fold_data.drop(columns=["row_index", "node", "is_root"]))
    y_val = validation_fold_data["is_root"]

    model = XGBClassifier(**configuration, random_state=42)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    with open(fold_path / "metrics.json", "w") as f:
        json.dump({"accuracy": accuracy}, f, indent=4)
    logger.info(f"Fold {fold + 1}/{cross_validation_folds} - Accuracy: {accuracy:.4f}")

    del model

    return accuracy

