FROM python:3.7-slim-buster

COPY requirements.txt ./

RUN apt-get update \ 
    && apt-get install -y python3-pip libpangocairo-1.0-0 libpcre3 libpcre3-dev \
    && pip3 install --no-cache-dir -r requirements.txt \
    && apt-get remove --auto-remove -y libpcre3-dev \ 
    && rm -rf /var/cache/apt/archives

WORKDIR /app

COPY . .

# ENTRYPOINT ["/bin/bash"]
ENTRYPOINT ["uwsgi"]
CMD ["--ini=/etc/uwsgi.ini"]