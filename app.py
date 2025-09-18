from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import json
import io
import os
from datetime import datetime

app = FastAPI(
    title="Boston House Price Prediction API",
    description="API to predict Boston house prices using ML model 🚀",
    version="2.0.0"
)

# --- Enable CORS so React can call this backend ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict to ["http://localhost:3000"] in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load model and metadata once at startup ---
MODEL_PATH = "models/boston_pipeline.joblib"
META_PATH = "models/model_metadata.json"

pipeline = joblib.load(MODEL_PATH)

metadata = {}
if os.path.exists(META_PATH):
    with open(META_PATH, "r") as f:
        metadata = json.load(f)

# --- Expected features ---
EXPECTED_COLUMNS = [
    "CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS",
    "RAD","TAX","PTRATIO","B","LSTAT"
]

# --- Folder for saving uploaded files ---
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --- Root Route ---
@app.get("/")
def root():
    return {
        "message": "Boston House Price Prediction API is running 🚀",
        "endpoints": ["/predict", "/predict_csv", "/feature_importance", "/docs"]
    }

# --- Define single-row schema ---
class HouseFeatures(BaseModel):
    CRIM: float
    ZN: float
    INDUS: float
    CHAS: float
    NOX: float
    RM: float
    AGE: float
    DIS: float
    RAD: float
    TAX: float
    PTRATIO: float
    B: float
    LSTAT: float

@app.post("/predict")
def predict_price(features: HouseFeatures):
    """
    Predict for a single row of features (JSON body).
    """
    try:
        df = pd.DataFrame([features.dict()])
        preds = pipeline.predict(df)
        return {"predictions": preds.tolist()}
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

@app.post("/predict_csv")
async def predict_from_csv(file: UploadFile = File(...)):
    """
    Predict for multiple rows from a CSV file.
    """
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))

        # Validate columns
        if list(df.columns) != EXPECTED_COLUMNS:
            return {"error": f"Invalid CSV format. Expected columns: {EXPECTED_COLUMNS}"}

        preds = pipeline.predict(df)

        # Save uploaded file + predictions
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(UPLOAD_DIR, f"{timestamp}_{file.filename}")
        df["PredictedPrice"] = preds
        df.to_csv(save_path, index=False)

        return {"predictions": preds.tolist(), "saved_file": save_path}

    except Exception as e:
        return {"error": f"Failed to process CSV: {str(e)}"}

@app.get("/feature_importance")
def feature_importance():
    """
    Return feature importances either from metadata.json
    or directly from the model if available.
    """
    if "feature_importances" in metadata:
        transformed = metadata.get("transformed_feature_names", metadata.get("input_features", []))
        importances = metadata["feature_importances"]
    else:
        model = None
        try:
            model = pipeline.named_steps.get("model", None)
        except Exception:
            pass

        if model is not None and hasattr(model, "feature_importances_"):
            importances = model.feature_importances_.tolist()
            if "preprocessor" in pipeline.named_steps:
                transformed = pipeline.named_steps["preprocessor"].get_feature_names_out().tolist()
            else:
                transformed = [f"feature_{i}" for i in range(len(importances))]
        else:
            return {"error": "This model does not provide feature importances"}

    # 🧹 Clean feature names
    clean_features = []
    for f in transformed:
        if "__" in f:
            f = f.split("__")[-1]
        clean_features.append(f)

    feature_importance_list = [
        {"feature": f, "importance": float(i)}
        for f, i in zip(clean_features, importances)
    ]
    feature_importance_list.sort(key=lambda x: x["importance"], reverse=True)
    return {"features": feature_importance_list}
