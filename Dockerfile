FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY expected_features.pkl .
COPY gender_expected_features.pkl .
COPY mlruns ./mlruns

EXPOSE 5000

CMD ["python", "app.py"]
