# ─────────────────────────────────────────────────────────────
# Dashboard Dockerfile (Multi-stage)
# ─────────────────────────────────────────────────────────────
# Stage 1: Build dependencies
FROM python:3.11-slim AS builder

WORKDIR /app

COPY dashboard/requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy dashboard code
COPY dashboard/ ./dashboard/

# Create non-root user
RUN useradd --create-home appuser
USER appuser

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

# Run Streamlit
CMD ["streamlit", "run", "dashboard/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
