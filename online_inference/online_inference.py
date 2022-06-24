import time

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel
import pandas as pd

from ml_project_package.models.model_functions import (
    load_model, 
    predict_unprocessed_df,
)
from ml_project_package.features.build_features import load_encoder
from ml_project_package.entities.train_pipeline_params import read_training_pipeline_params

CONFIG_PATH = "configs/train_config.yaml"

class TestSample(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

    def to_pandas_df(self):
        data = {
            "age": [self.age], 
            "sex": [self.sex], 
            "cp": [self.cp], 
            "trestbps": [self.trestbps], 
            "chol": [self.chol], 
            "fbs": [self.fbs], 
            "restecg": [self.restecg], 
            "thalach": [self.thalach], 
            "exang": [self.exang], 
            "oldpeak": [self.oldpeak], 
            "slope": [self.slope], 
            "ca": [self.ca], 
            "thal": [self.thal], 
        }
        return pd.DataFrame(data=data)

    def validate_params(self):
        # Data types checking
        if not isinstance(self.age, int):
            return "age must be integer"
        if not isinstance(self.sex, int):
            return "sex must be integer"
        if not isinstance(self.cp, int):
            return "cp must be integer"
        if not isinstance(self.trestbps, int):
            return "trestbps must be integer"
        if not isinstance(self.chol, int):
            return "chol must be integer"
        if not isinstance(self.fbs, int):
            return "fbs must be integer"
        if not isinstance(self.restecg, int):
            return "restecg must be integer"
        if not isinstance(self.thalach, int):
            return "thalach must be integer"
        if not isinstance(self.exang, int):
            return "exang must be integer"
        if not isinstance(self.oldpeak, float):
            return "oldpeak must be float"
        if not isinstance(self.slope, int):
            return "slope must be integer"
        if not isinstance(self.ca, int):
            return "ca must be integer"
        if not isinstance(self.thal, int):
            return "thal must be integer"
        # Bounds checking
        if self.age < 0:
            return "age cannot be less than 0"
        if self.sex not in [0, 1]:
            return "sex must be 0(female) or 1(male)"
        if self.cp not in [0, 1, 2, 3]:
            return "cp must be integer between 0 and 3"
        if self.trestbps < 0:
            return "trestbps cannot be less than 0"
        if self.chol < 0:
            return "chol cannot be less than 0"
        if self.fbs not in [0, 1]:
            return "fbs must be 0(if fasting blood sugar <= 120 mg/dl) or 1(otherwise)"
        if self.restecg not in [0, 1, 2]:
            return "restecg must be 0, 1 or 2"
        if self.thalach < 0:
            return "thalach cannot be less than 0"
        if self.exang not in [0, 1]:
            return "exang(exercise induced angina) must be 0(if no) or 1(if yes)"
        if self.slope not in [0, 1, 2]:
            return "slope(he slope of the peak exercise ST segment) must be 0(upsloping), 1(flat) or 2(downsloping)"
        if self.ca not in [0, 1, 2, 3]:
            return "ca: number of major vessels (0-3) colored by flourosopy"
        if self.thal not in [0, 1, 2]:
            return "thal must be 0(normal), 1(fixed defect) or 2(reversable defect and the label)"
        return "OK"

def initialize_service_features():
    params = read_training_pipeline_params(CONFIG_PATH)
    model = load_model(params.output_model_path)
    enc = load_encoder(params.output_encoder_path)
    return params, model, enc

PARAMS, MODEL, ENC = initialize_service_features()

app = FastAPI()

time.sleep(20)
start_time = time.time()

@app.get("/health/", status_code=200)
async def check_service_ready():
    if time.time() - start_time > 60:
        raise RuntimeError("App stopped working")
    return "App is ready for working"

@app.post("/predict/")
async def predict_item(item: TestSample):
    validation = item.validate_params()
    if validation != "OK":
        raise HTTPException(status_code=400, detail=validation)
    pred = predict_unprocessed_df(df=item.to_pandas_df(), model=MODEL, enc=ENC, params=PARAMS.feature_params)[0]
    return "Artificial doctor's verdict is: {}".format(bool(pred))
