from dogs_prediction.DL_logic import predict
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

#iniate api
app = FastAPI()

# Optional, good practice for dev purposes. Allow all middlewares
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# preload the model
app.state.model = predict.load_latest_model()

# add predict endpoint
@app.get("/predict")
def prediction(url: str):
    model = app.state.model
    assert model is not None

    return predict.predict_labels(url, model)


#root endpoint
@app.get("/")
def root():
    return dict(greeting="Bark!")
