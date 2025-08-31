from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession
from pyspark.ml.classification import (
    RandomForestClassificationModel, DecisionTreeClassificationModel, 
    LogisticRegressionModel, GBTClassificationModel
)
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, DoubleType
)
from typing import Optional
import uvicorn

# --- Configuration values are now defined directly in the script ---
PREPROCESSING_PIPELINE_PATH = "models/preprocessing_pipeline"
BEST_MODEL_PATH = "models/Classifier"
API_HOST = "0.0.0.0"
API_PORT = 5050
# --- End of configuration ---

# --- Helper Functions (moved from utils.py to make this standalone) ---
def get_spark_session(app_name):
    return SparkSession.builder \
        .appName(app_name) \
        .config("spark.executor.memory", "4g") \
        .config("spark.executor.cores", "4") \
        .config("spark.hadoop.io.native.lib.available", "false") \
        .getOrCreate()

def load_best_model(model_path):
    model_classes = [
        RandomForestClassificationModel, DecisionTreeClassificationModel,
        LogisticRegressionModel, GBTClassificationModel
    ]
    for model_class in model_classes:
        try:
            return model_class.load(model_path)
        except Exception:
            continue
    raise IOError(f"Could not load any supported model from path: {model_path}")

# --- Initialize App and Load Models ---
app = FastAPI()
spark = get_spark_session("Titanic Inference API")

try:
    preprocessing_pipeline = PipelineModel.load(PREPROCESSING_PIPELINE_PATH)
    model = load_best_model(BEST_MODEL_PATH)
    print("Models loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")
    preprocessing_pipeline = None
    model = None

# --- Define the explicit schema for incoming data ---
input_schema = StructType([
    StructField("PassengerId", IntegerType(), True),
    StructField("Pclass", IntegerType(), True),
    StructField("Name", StringType(), True),
    StructField("Sex", StringType(), True),
    StructField("Age", DoubleType(), True),
    StructField("SibSp", IntegerType(), True),
    StructField("Parch", IntegerType(), True),
    StructField("Ticket", StringType(), True),
    StructField("Fare", DoubleType(), True),
    StructField("Cabin", StringType(), True),
    StructField("Embarked", StringType(), True)
])

# --- Define API Data Models ---
class Passenger(BaseModel):
    PassengerId: int
    Pclass: int
    Name: Optional[str] = None
    Sex: Optional[str] = None
    Age: Optional[float] = None
    SibSp: Optional[int] = None
    Parch: Optional[int] = None
    Ticket: Optional[str] = None
    Fare: Optional[float] = None
    Cabin: Optional[str] = None
    Embarked: Optional[str] = None

# --- Define Prediction Endpoint ---
@app.post("/predict")
async def predict(passenger: Passenger):
    if not preprocessing_pipeline or not model:
        raise HTTPException(status_code=503, detail="Models are not available.")
    try:
        # Create Spark DataFrame using the explicit schema
        raw_df = spark.createDataFrame([passenger.dict()], schema=input_schema)
        
        processed_df = preprocessing_pipeline.transform(raw_df)
        prediction = model.transform(processed_df).select("prediction").first()["prediction"]
        
        status = "Survived" if prediction == 1 else "Not-Survived"
        return {"prediction": int(prediction), "status": status}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host=API_HOST, port=API_PORT)

