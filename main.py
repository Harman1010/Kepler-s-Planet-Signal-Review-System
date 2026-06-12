from fastapi import FastAPI, HTTPException,Depends
from pydantic import BaseModel
import joblib
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import sqlalchemy
from sqlalchemy import create_engine,String,Integer,Float
from sqlalchemy.orm import declarative_base,sessionmaker,Mapped,mapped_column,Session

Base = declarative_base()
eng = create_engine("sqlite:///database.db",connect_args={"check_same_thread" : False})
sessionLocal = sessionmaker(autocommit=False,autoflush=False,bind=eng)

class Model(Base):
    __tablename__ = "predictions"
    id: Mapped[int] = mapped_column(Integer,primary_key=True,index=True)
    confidence: Mapped[float] = mapped_column(Float)
    prediction: Mapped[str] = mapped_column(String)
    priority: Mapped[str] = mapped_column(String)
    review_status: Mapped[str] = mapped_column(String)

Base.metadata.create_all(bind=eng)

app = FastAPI()

model = joblib.load("finalModel.joblib")
model_embed = SentenceTransformer('all-MiniLM-L6-v2')
df = pd.read_csv("history_df.csv")
planet_texts = joblib.load("planet_texts.joblib")
faiss_index = faiss.read_index("faiss_index.bin")
le = joblib.load("label_encoder.joblib")

def get_db():
    try:
        db = sessionLocal()
        yield db
    finally:
        db.close()

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
    Orbital Period: {period},
    Transit Depth: {depth},
    Planet Radius: {prad},
    Stellar Temperature: {steff}
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

@app.get("/predictions")
def get_predictions(db: Session = Depends(get_db)):
    return db.query(Model).all()

@app.post("/predict")
def predict(data: ValidationRequest,db : Session = Depends(get_db)):
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

    record = Model(
    confidence=review_data["Confidence Score"],
    prediction=prediction_label,
    priority=review_data["Priority"],
    review_status=review_data["Review Status"])

    db.add(record)
    db.commit()
    db.refresh(record)

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

class Update(BaseModel):
    review_status : str | None = None

@app.patch("/predictions/{prediction_id}")
def patch(data : Update,prediction_id,db : Session = Depends(get_db)):
    stmt = db.query(Model).filter(prediction_id == Model.id).first()
    if stmt is None:
        raise HTTPException(status_code=404,detail="Prediction not found")
    
    else:
        stmt.review_status = data.review_status

    db.commit()
    db.refresh(stmt)
    return stmt

@app.delete("/predictions/{prediction_id}")
def delete_record(prediction_id,db : Session = Depends(get_db)):
    stmt = db.query(Model).filter(prediction_id == Model.id).first()
    if stmt is None:
        raise HTTPException(status_code=404,detail="Record not found")
    
    db.delete(stmt)
    db.commit()
    
    return {
        "Message" : "Record Deleted",
        "ID" : prediction_id
    }





        