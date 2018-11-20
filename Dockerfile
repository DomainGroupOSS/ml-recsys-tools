FROM python:3.6

COPY requirements.txt .

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

ENV APP_DIR=/ml_recsys_tools

ADD . ${APP_DIR}

WORKDIR ${APP_DIR}

RUN pip install -i file://$(realpath .) .

CMD ["python", "-m", "unittest"]

# docker build --pull -t domaingroupossml/ml_recsys_tools:latest .
# docker run --rm domaingroupossml/ml_recsys_tools:latest python -m unittest
# docker push domaingroupossml/ml_recsys_tools:latest
