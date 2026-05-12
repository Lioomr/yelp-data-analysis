(
echo FROM jupyter/pyspark-notebook:spark-3.3.2
echo USER root
echo WORKDIR /opt/workspace
echo COPY requirements.txt .
echo RUN pip install -r requirements.txt
echo USER $NB_USER
) > jupyter\Dockerfile