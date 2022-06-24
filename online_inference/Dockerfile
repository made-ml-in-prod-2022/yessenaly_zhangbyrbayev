FROM python:3.7

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .
RUN pip install -e .
RUN python train_pipeline.py

CMD ["uvicorn", "online_inference:app", "--host", "0.0.0.0", "--port", "80"]