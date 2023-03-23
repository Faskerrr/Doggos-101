from dogs_prediction.DL_logic import predict, registry
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
app.state.model = registry.load_latest_model(loading_method = 'local')

# add predict endpoint
@app.get("/predict")
def prediction(url: str, model_type = 'inception_v3'):
    model = app.state.model
    assert model is not None
    prediction = predict.predict_labels(model, model_type, url = url)
    return prediction


#root endpoint
@app.get("/")
def root():
    return dict(greeting="Bark!")
