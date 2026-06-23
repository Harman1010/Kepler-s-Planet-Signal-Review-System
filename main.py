from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from database import Base, eng

from routers.auth_router import router as auth_router

from routers.prediction_router import router as prediction_router

from routers.history_router import router as history_router

Base.metadata.create_all(bind=eng)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(auth_router)
app.include_router(prediction_router)
app.include_router(history_router)


@app.get("/")
def home():
    return {
        "message":
        "Welcome to Astronomy with FastAPI!"
    }


@app.get("/example")
def example():

    return {
        "predict_example": {
            "url": "/predict"
        },

        "history_example": {
            "url": "/history"
        }
    }