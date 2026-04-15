FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Ensure /app is always on Python path
ENV PYTHONPATH=/app
RUN cp sitecustomize.py /usr/local/lib/python3.11/site-packages/sitecustomize.py

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
