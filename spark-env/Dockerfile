
FROM python:3.10-slim

USER root
# Install Java required for PySpark
RUN apt-get update && \
    apt-get install -y default-jre-headless procps curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Expose Jupyter port
EXPOSE 8888

# Ensure container stays running (or we could start Jupyter by default)
# Command to start jupyter: jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
CMD ["tail", "-f", "/dev/null"]
