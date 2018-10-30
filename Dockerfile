FROM python:3.6

COPY requirements.txt .

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

ENV APP_DIR=/ml_recsys_tools
ENV PYTHONPATH "${PYTHONPATH}:${APP_DIR}"

ADD . ${APP_DIR}

WORKDIR ${APP_DIR}

RUN pip install .

CMD ["python", "-m", "unittest"]

# docker build -t artdgn/ml_recsys_tools:latest .
# docker run --rm artdgn/ml_recsys_tools:latest python -m unittest
# docker push artdgn/ml_recsys_tools:latest
