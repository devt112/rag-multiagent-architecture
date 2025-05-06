FROM marketplace.gcr.io/google/debian11:latest

USER root

# Install dependencies needed for building Python and other tools
RUN apt-get update && \
    apt-get install -y \
    wget unzip libpq-dev build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev libbz2-dev \
    # Add these for SSL support which is often needed by pip
    libssl-dev openssl

WORKDIR /tmp

WORKDIR /tmp/Python-3.10.11

RUN ./configure --enable-optimizations --with-ensurepip
RUN make -j$(nproc)
RUN make altinstall

RUN update-alternatives --install /usr/bin/python python /usr/local/bin/python3.10 100 && \
    update-alternatives --install /usr/bin/pip pip /usr/local/bin/pip3.10 100
	
RUN python --version
RUN pip --version

WORKDIR /app

COPY requirements.txt .
COPY app.py .

RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["python", "app.py"]