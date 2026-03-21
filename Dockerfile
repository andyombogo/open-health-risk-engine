FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PROJECT_REPO_URL=https://github.com/andyombogo/open-health-risk-engine
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

COPY runtime_requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r runtime_requirements.txt

COPY . .
RUN python src/verify_runtime.py

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--server.enableXsrfProtection=false"]
