FROM python

RUN set -x \
    && apt-get update \
    && apt-get install -y build-essential \
    && apt-get install -y python-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install ml_recsys_tools
