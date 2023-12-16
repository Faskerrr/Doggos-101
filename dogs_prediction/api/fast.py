from dogs_prediction.DL_logic import predict, registry
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# iniate api
app = FastAPI()

# optional, good practice for dev purposes. Allow all middlewares
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],  # Allows all origins
    allow_credentials=True,
    allow_methods=['*'],  # Allows all methods
    allow_headers=['*'],  # Allows all headers
)

# preload the model
app.state.model = registry.load_convnext_model()

# predicts from url provided by user
@app.get('/predict_url')
def prediction(url_with_pic):
    model = app.state.model
    assert model is not None
    prediction = predict.predict_labels(model, url_with_pic=url_with_pic)
    return prediction

# predicts from file provided by user
@app.post('/predict_file')
def prediction(file: UploadFile):
    model = app.state.model
    assert model is not None
    prediction = predict.predict_labels(model, img=file.file)
    return prediction

# root endpoint
@app.get('/')
def root():
    return dict(greeting='Bark!')
