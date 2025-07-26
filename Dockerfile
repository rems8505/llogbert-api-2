FROM python:3.10-slim

WORKDIR /app
COPY . .
RUN pip install --upgrade pip && pip install -e .

ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
