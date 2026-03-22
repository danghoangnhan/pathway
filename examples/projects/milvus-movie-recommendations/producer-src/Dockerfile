FROM python:3.10-slim
RUN pip install --no-cache-dir kafka-python
WORKDIR /app
COPY producer-src/create_stream.py .
CMD ["python", "-u", "create_stream.py"]
