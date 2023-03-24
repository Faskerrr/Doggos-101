from dogs_prediction.DL_logic import predict, registry
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

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
app.state.model = registry.load_latest_model()

# add predict endpoint
# @app.get("/predict")
# def prediction(img='', url='', model_type='inception_v3'):
#     model = app.state.model
#     assert model is not None
#     if url:
#         prediction = predict.predict_labels(model, model_type, url = url)
#     else:
#         prediction = predict.predict_labels(b64_2_img(img), model, model_type)
#     return prediction

# works with urls and files. if both provided will predict the url
@app.post("/predict")
def prediction(file: UploadFile | None = None, url='', model_type='inception_v3'):
    model = app.state.model
    assert model is not None
    if url:
        prediction = predict.predict_labels(model, model_type, url = url)
    else:
        prediction = predict.predict_labels(model, model_type, img=Image.open(file.file))
    return prediction


#root endpoint
@app.get("/")
def root():
    return dict(greeting="Bark!")
