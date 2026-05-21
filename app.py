from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

app = FastAPI()

model = joblib.load("finalModel.joblib")
le = joblib.load("label_encoder.joblib")
model_embed = SentenceTransformer('all-MiniLM-L6-v2')
df = joblib.load("history_df.joblib")
planet_texts = joblib.load("planet_texts.joblib")
faiss_index = faiss.read_index("faiss_index.bin")


def engine(prob):
    confidence = round(prob * 100, 2)
    if confidence >= 70:
        priority = "Critical"
        review_status = "Immediate Review"
    elif confidence >= 50:
        priority = "High"
        review_status = "Scientist Validation"
    elif confidence >= 30:
        priority = "Medium"
        review_status = "Review Queue"
    else:
        priority = "Low"
        review_status = "Auto-Filtered"
    return {
        "Confidence Score": confidence,
        "Priority": priority,
        "Review Status": review_status
    }

def create_input_signal(period,depth,prad,steff):
  text = f"""
    Orbital Period: {'koi_period'},
    Transit Depth: {'koi_depth'},
    Planet Radius: {'koi_prad'},
    Stellar Temperature: {'koi_steff'}
    """
  return text

def retrieve_similar_planets(input_text,k=3):

    query_embedding = model_embed.encode([input_text])

    distances,indices = faiss_index.search(np.array(query_embedding).astype('float32'),k)

    results = []

    for idx in indices[0]:

        idx = int(idx)
        results.append({
            "Planet Name": str(df.iloc[idx]['kepoi_name']),

            "Planet Summary": str(planet_texts[idx]),

            "Disposition": le.inverse_transform([int(df.iloc[idx]['koi_disposition'])])[0]

        })

    return results

class ValidationRequest(BaseModel):
    koi_period: float
    koi_time0bk: float
    koi_duration: float
    koi_depth: float
    koi_impact: float
    koi_model_snr: float
    koi_prad: float
    koi_teq: float
    koi_insol: float
    koi_steff: float

class HistoryRequest(BaseModel):
    
        koi_period: float

        koi_depth: float

        koi_prad: float

        koi_steff: float



@app.get("/")
def home():
    return {"message": "Welcome to Astronomy with FastAPI!"}

@app.get("/example")
def example():

    return {

        "predict_example": {

            "url": "/predict",

            "method": "POST",

            "sample_input": {

                "koi_period": 0.5,
                "koi_time0bk": 100.0,
                "koi_duration": 2.5,
                "koi_depth": 120.0,
                "koi_impact": 0.3,
                "koi_model_snr": 15.0,
                "koi_prad": 1.1,
                "koi_teq": 500.0,
                "koi_insol": 1.2,
                "koi_steff": 5500.0

            }

        },

        "history_example": {

            "url": "/history",

            "method": "POST",

            "sample_input": {

                "koi_period": 0.5,
                "koi_depth": 120.0,
                "koi_prad": 1.1,
                "koi_steff": 5500.0

            }

        }

    }

@app.post("/predict")
def predict(data: ValidationRequest):
    input_df = pd.DataFrame([{
        "koi_period": data.koi_period,
        "koi_time0bk": data.koi_time0bk,
        "koi_duration": data.koi_duration,
        "koi_depth": data.koi_depth,
        "koi_impact": data.koi_impact,
        "koi_model_snr": data.koi_model_snr,
        "koi_prad": data.koi_prad,
        "koi_teq": data.koi_teq,
        "koi_insol": data.koi_insol,
        "koi_steff": data.koi_steff
    }])
    
    prediction = model.predict(input_df)[0]

    prediction_label = le.inverse_transform([prediction])[0]

    probabilities = model.predict_proba(input_df)[0]

    confirmed_idx = list(le.classes_).index("CONFIRMED")

    confirmed_prob = float(probabilities[confirmed_idx])

    review_data = engine(confirmed_prob)

    return {

        "predicted_class": prediction_label,

        "Confidence Score": review_data["Confidence Score"],

        "Priority": review_data["Priority"],

        "Review Recommendation": review_data["Review Status"]

    }

@app.post("/history")
def history(data : HistoryRequest):
    query_text = create_input_signal(data.koi_period,data.koi_depth,data.koi_prad,data.koi_steff)

    similar_planets = retrieve_similar_planets(query_text,k=3)

    return {

        "Historical Similar Planets": similar_planets

    }


        