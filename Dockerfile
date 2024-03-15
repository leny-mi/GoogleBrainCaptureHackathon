FROM python:3.10-slim

EXPOSE 8080
WORKDIR /app

RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt /app

RUN pip install -r requirements.txt

COPY . /app

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]