FROM python:3.9

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ ./api
COPY model/ ./model

WORKDIR /app/api

CMD ["python", "main.py"]
