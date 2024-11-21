from python:3.11

WORKDIR /app



RUN apt update
RUN apt install -y poppler-utils
RUN apt install -y tesseract-ocr
RUN apt install -y libtesseract-dev
RUN export TESSERACT_PATH=/usr/bin/tesseract
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

COPY ./api/ /app/
COPY poetry.lock pyproject.toml /app/
COPY ./api/swe.traineddata /usr/share/tesseract-ocr/5/tessdata/

RUN pip install poetry
RUN poetry config virtualenvs.create false
RUN poetry install 

RUN pip install pytesseract
RUN pip install gunicorn
RUN pip install opencv-python
RUN pip install pip-licenses

#CMD uvicorn api:app --port=80 --host="0.0.0.0" --log-level=debug --lifespan=on
#CMD gunicorn api:app --timeout 300 -w 1 --threads 80 --worker-class uvicorn.workers.UvicornWorker
ENTRYPOINT ["./entrypoint.sh"] 

