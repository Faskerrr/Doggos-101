from dogs_prediction.DL_logic.predict import load_latest_model()
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
