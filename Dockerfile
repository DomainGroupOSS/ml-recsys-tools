FROM python:3.6

#RUN pip install ml_recsys_tools
RUN git clone "https://github.com/DomainGroupOSS/ml-recsys-tools.git"
RUN pip install -e ml-recsys-tools

# docker build -t artdgn/ml_recsys_tools:latest .
# docker push artdgn/ml_recsys_tools:latest
