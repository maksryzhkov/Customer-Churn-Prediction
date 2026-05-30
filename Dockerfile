FROM python:3.10-slim

WORKDIR /app

RUN useradd -m appuser

COPY requirements.txt .

RUN pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chown -R appuser:appuser /app

USER appuser

CMD ["python", "pipeline.py"]
