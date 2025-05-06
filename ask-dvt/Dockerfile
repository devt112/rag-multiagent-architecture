FROM gcr.io/iaas-gcr-reg-prd-ad3d/golden/os/debian:11.0-slim

USER root

# Install dependencies needed for building Python and other tools
RUN apt-get update && \
    apt-get install -y \
    curl \
    software-properties-common \
    wget unzip libpq-dev build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev libbz2-dev \
    # Add these for SSL support which is often needed by pip
    libssl-dev openssl
    # && rm -rf /var/lib/apt/lists/*    

WORKDIR /home/efx_container_user/

# Download and extract Python.  Using specific version is good.
RUN wget https://nrm.us..com/repository/iaas-container-imaging-aritifacts/gcp_container_artifacts/Python/3.10.11/Python-3.10.11.tar.xz && \
    tar -xf Python-3.10.11.tar.xz && \
    rm Python-3.10.11.tar.xz

WORKDIR /home/efx_container_user/Python-3.10.11


RUN ./configure --enable-optimizations --with-ensurepip
RUN make -j$(nproc)
RUN make altinstall

RUN update-alternatives --install /usr/bin/python python /usr/local/bin/python3.10 100 && \
    update-alternatives --install /usr/bin/pip pip /usr/local/bin/pip3.10 100
	
RUN python --version
RUN pip --version

WORKDIR /home/efx_container_user/app

COPY --chown=efx_container_user ./ ./

RUN pip install --index-url https://nrm.us..com/repository/efxpypi/simple/ --extra-index-url=https://nrm.us..com/repository/pypiregistry/ -r requirements.txt

EXPOSE 8501
WORKDIR /home/efx_container_user/app
RUN chmod -R 775 /home/efx_container_user/app && \
    chown -R efx_container_user:efx_container_user /home/efx_container_user
    
USER efx_container_user
ENTRYPOINT ["streamlit", "run", "streamlit/app.py", "--server.port=8501", "--server.enableCORS=false", "--server.enableXsrfProtection=false", "--server.headless=true", "--server.address=0.0.0.0", "--server.baseUrlPath=askgbp-engine"]