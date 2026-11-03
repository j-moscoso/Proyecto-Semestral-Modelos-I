FROM python:3.11.9-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY tests/ tests/
COPY train.py predict.py pytest.ini ./

RUN mkdir -p /app/data /app/models /app/outputs

CMD ["python", "train.py", "--help"]
