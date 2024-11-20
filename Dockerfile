FROM python:3.12.7

WORKDIR /app

COPY . /app

RUN pip install /app/install_files/*

ENV POPPLER_PATH /usr/local/bin

# Expose port for FastAPI
EXPOSE 8000

# Set the default command to run the FastAPI app
CMD ["uvicorn", "doc-n-process-ai_app:app", "--host", "0.0.0.0", "--port", "8000"]
