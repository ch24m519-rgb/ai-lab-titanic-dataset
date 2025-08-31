import mlflow
import mlflow.spark
from mlflow.tracking import MlflowClient
from pyspark.sql import SparkSession
from pyspark.ml.classification import (
    LogisticRegression, RandomForestClassifier, DecisionTreeClassifier
)
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# --- Helper Functions (to make this script standalone and robust) ---

def get_spark_session(app_name):
    """Initializes and returns a Spark session."""
    return SparkSession.builder \
        .appName(app_name) \
        .config("spark.executor.memory", "4g") \
        .config("spark.executor.cores", "4") \
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
        title, x_label = 'Logistic Regression Feature Importance', 'Coefficient'
    elif hasattr(bestModel, "featureImportances"):
        importances = bestModel.featureImportances.toArray()
        title, x_label = f"{algo_name} Feature Importance", 'Importance'
    else:
        return

    feature_df = pd.DataFrame({
        'Feature': [f"feature_{i}" for i in range(len(importances))],
        'importance': importances
    }).sort_values(by='importance', ascending=False).head(10)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x="importance", y="Feature", data=feature_df)
    plt.title(title)
    plt.xlabel(x_label)
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    mlflow.log_artifact("feature_importance.png")
    plt.close()

# --- Main Training Function ---

def run_training():
    """
    Runs the full model training pipeline, reading hyperparameters from params.yaml.
    """
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Titanic-Project")
    
    spark = get_spark_session("Titanic Training")

    train_df = spark.read.parquet("data/processed/train_features.parquet")
    val_df = spark.read.parquet("data/processed/val_features.parquet")

    with open("params.yaml", 'r') as f:
        params_config = yaml.safe_load(f)

    # Evaluator for AUC (binary classification)
    binary_evaluator = BinaryClassificationEvaluator(
        labelCol="Survived", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
    )
    
    # Evaluator for other metrics (accuracy, precision, etc.)
    multi_evaluator = MulticlassClassificationEvaluator(
        labelCol="Survived", predictionCol="prediction"
    )
    
    all_results = []
    
    # Define models to train based on the params.yaml file
    model_configs = [
        ("LogisticRegression", LogisticRegression, params_config["LogisticRegression"]),
        ("RandomForest", RandomForestClassifier, params_config["RandomForestClassifier"]),
        ("DecisionTree", DecisionTreeClassifier, params_config["DecisionTreeClassifier"])
    ]

    for algo_name, model_class, params in model_configs:
        with mlflow.start_run(run_name=algo_name) as run:
            model = model_class(featuresCol="features", labelCol="Survived")
            
            # Dynamically build the parameter grid from the YAML file
            param_grid = ParamGridBuilder()
            for param_name, values in params.items():
                param_grid.addGrid(getattr(model, param_name), values)
            
            cv = CrossValidator(estimator=model, estimatorParamMaps=param_grid.build(), evaluator=binary_evaluator, numFolds=3, parallelism=2)
            
            print(f"--- Training {algo_name} ---")
            cv_model = cv.fit(train_df)
            best_model = cv_model.bestModel
            
            mlflow.log_params({p.name: v for p, v in best_model.extractParamMap().items() if p.name in params})
            
            predictions = best_model.transform(val_df)
            
            # --- Log All Evaluation Metrics ---
            auc = binary_evaluator.evaluate(predictions)
            accuracy = multi_evaluator.setMetricName("accuracy").evaluate(predictions)
            precision = multi_evaluator.setMetricName("weightedPrecision").evaluate(predictions)
            recall = multi_evaluator.setMetricName("weightedRecall").evaluate(predictions)
            f1 = multi_evaluator.setMetricName("f1").evaluate(predictions)

            mlflow.log_metric("Test_AUC", auc)
            mlflow.log_metric("Accuracy", accuracy)
            mlflow.log_metric("Precision", precision)
            mlflow.log_metric("Recall", recall)
            mlflow.log_metric("F1_Score", f1)

            print(f"Test AUC for {algo_name}: {auc:.4f}")
            print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

            log_evaluation_artifacts(predictions, best_model, algo_name)

            registered_model_name = f"Titanic{algo_name}"
            logged_model_info = mlflow.spark.log_model(best_model, artifact_path="model", registered_model_name=registered_model_name)
            
            all_results.append({
                "model_name": registered_model_name, 
                "auc": auc, 
                "model": best_model,
                "version": logged_model_info.registered_model_version
            })

    # Find and promote the best model
    best_entry = max(all_results, key=lambda x: x["auc"])
    best_model, best_model_name, best_auc, best_version = best_entry["model"], best_entry["model_name"], best_entry["auc"], best_entry["version"]
    
    best_model.write().overwrite().save("models/Classifier")
    print(f"\nBest model ({best_model.__class__.__name__}) saved to models/Classifier")

    client = MlflowClient()
    client.transition_model_version_stage(name=best_model_name, version=best_version, stage="Staging", archive_existing_versions=True)

    prod_versions = client.get_latest_versions(name=best_model_name, stages=["Production"])
    if not prod_versions or best_auc > client.get_run(prod_versions[0].run_id).data.metrics.get("Test_AUC", 0):
        print(f"Promoting model to Production (AUC: {best_auc:.4f}).")
        client.transition_model_version_stage(name=best_model_name, version=best_version, stage="Production", archive_existing_versions=True)
    else:
        prod_auc = client.get_run(prod_versions[0].run_id).data.metrics.get("Test_AUC", 0)
        print(f"New model (AUC: {best_auc:.4f}) does not outperform Production model (AUC: {prod_auc:.4f}). Keeping in Staging.")

    spark.stop()

if __name__ == "__main__":
    run_training()

