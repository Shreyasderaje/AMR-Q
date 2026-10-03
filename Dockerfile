FROM python:3.12-slim

WORKDIR /app

# CPU-only torch keeps the image small
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY amrq ./amrq
COPY data ./data
COPY frontend ./frontend
COPY scripts ./scripts
COPY tests ./tests

EXPOSE 8000
CMD ["uvicorn", "amrq.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
