FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY api.py app.py build_index.py ./

RUN mkdir -p embeddings .cache && useradd --create-home app && chown -R app /app
USER app

EXPOSE 8000 8501
# Default: REST API. Override for the UI:  streamlit run app.py --server.port 8501 --server.address 0.0.0.0
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
