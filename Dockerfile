FROM tensorflow/tensorflow:2.12.0-gpu

COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && pip install -r /tmp/requirements.txt
