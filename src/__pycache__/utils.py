import mlflow
from pyspark.sql import SparkSession
from pyspark.ml.classification import (
    GBTClassificationModel, RandomForestClassificationModel,
    DecisionTreeClassificationModel, LogisticRegressionModel
)
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

def get_spark_session(app_name, memory="4g", cores="4"):
    """
    Initializes and returns a Spark session with a near-optimal and configurable configuration.
    This version accepts memory and cores as arguments.
    """
    return SparkSession.builder \
        .appName(app_name) \
        .config("spark.executor.memory", memory) \
        .config("spark.executor.cores", cores) \
        .config("spark.hadoop.io.native.lib.available", "false") \
        .config("spark.hadoop.fs.mlflow-artifacts.impl", "org.mlflow.spark.MlflowArtifactsFileSystem") \
        .getOrCreate()

def log_evaluation_artifacts(predictions, bestModel, algo_name):
    """Logs confusion matrix and feature importance plots as MLflow artifacts."""
    y_true = [int(row['Survived']) for row in predictions.select("Survived").collect()]
    y_pred = [int(row['prediction']) for row in predictions.select('prediction').collect()]
    
    # Create and log confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Died", "Survived"], yticklabels=["Died", "Survived"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    plt.close()

    # Create and log feature importance plot
    if hasattr(bestModel, "coefficients"):
        importances = np.abs(bestModel.coefficients.toArray())
        title = 'Logistic Regression Feature Importance'
        x_label = 'Coefficient'
    elif hasattr(bestModel, "featureImportances"):
        importances = bestModel.featureImportances.toArray()
        title = f"{algo_name} Feature Importance"
        x_label = 'Importance'
    else:
        return # Skip if model type has no importance attribute

    feature_count = len(importances)
    feature_names = [f"feature_{i}" for i in range(feature_count)]
    feature_df = pd.DataFrame({'Feature': feature_names, 'importance': importances})
    feature_df_sorted = feature_df.sort_values(by='importance', ascending=False).head(10)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x="importance", y="Feature", data=feature_df_sorted)
    plt.title(title)
    plt.xlabel(x_label)
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    mlflow.log_artifact("feature_importance.png")
    plt.close()

def load_best_model(model_path):
    """
    Loads the best saved Spark ML model from a given path.
    Tries different model types to handle any saved format.
    """
    model_classes = [
        RandomForestClassificationModel,
        DecisionTreeClassificationModel,
        LogisticRegressionModel,
        GBTClassificationModel 
    ]
    for model_class in model_classes:
        try:
            return model_class.load(model_path)
        except Exception:
            continue
    raise IOError(f"Could not load any of the supported model types from path: {model_path}")

