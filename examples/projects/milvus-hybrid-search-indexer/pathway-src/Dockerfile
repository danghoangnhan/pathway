FROM pathwaycom/pathway:latest
RUN pip install --no-cache-dir "pymilvus>=2.5.0" "sentence-transformers"
WORKDIR /app
COPY pathway-src/app.py pathway-src/hybrid_milvus_writer.py ./
CMD ["python", "-u", "app.py"]
