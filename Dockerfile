FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    RAG_DATA_DIR=/data \
    OLLAMA_BASE_URL=http://host.docker.internal:11434

RUN apt-get update \
    && apt-get install --yes --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY app.py cli.py config.py engineering_agent.py physics_cli.py rag.py ./
COPY engineering ./engineering
COPY experiments ./experiments
COPY fine_tuning ./fine_tuning
COPY docs ./docs
RUN pip install --no-cache-dir .

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
