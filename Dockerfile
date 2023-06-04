FROM python:3.10.11

WORKDIR /app

ADD https://storage.googleapis.com/somethingssss/CNNmodif.h5 /app

RUN pip install --upgrade pip

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY . .

EXPOSE 8080

ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "--host", "0.0.0.0", "--port", "8080", "main:app"]
