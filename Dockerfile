FROM python:3.8.12-buster
WORKDIR /prod

# Copy and install requirements
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
# Copy dog_prediction logic
COPY dogs_prediction dogs_prediction
COPY setup.py setup.py
RUN pip install .
# Copy the model to container
COPY models models

CMD uvicorn dogs_prediction.api.fast:app --host 0.0.0.0 --port $PORT
