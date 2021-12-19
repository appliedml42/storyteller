FROM nvcr.io/nvidia/pytorch:21.11-py3

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y sudo poppler-utils curl git gcc vim wget unzip lsb-core software-properties-common

# Install NCCL2
RUN apt-get -y remove libnccl2 libnccl-dev
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
RUN add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
RUN apt-get update && apt-get install -y libnccl2=2.9.9-1+cuda11.3 libnccl-dev=2.9.9-1+cuda11.3

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
ADD . /app

# Install OpenMPI
RUN mv /usr/local/mpi /usr/local/bk_mpi
RUN wget https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.0.tar.gz
RUN gunzip -c openmpi-4.0.0.tar.gz | tar xf -
WORKDIR openmpi-4.0.0
RUN ./configure --prefix=/usr/local/openmpi
RUN make -j 8 all
RUN make install
ENV LD_LIBRARY_PATH /usr/local/openmpi/lib:$LD_LIBRARY_PATH
ENV PATH /usr/local/openmpi/bin:$PATH
RUN rm -rf openmpi*
WORKDIR /app

# Install Horovod
RUN HOROVOD_NCCL_INCLUDE=/usr/include HOROVOD_NCCL_LIB=/usr/lib/x86_64-linux-gnu HOROVOD_GPU_OPERATIONS=NCCL\
    HOROVOD_WITH_MPI=1 HOROVOD_WITH_PYTORCH=1 HOROVOD_WITHOUT_TENSORFLOW=1 HOROVOD_WITHOUT_MXNET=1\
    pip install --no-cache-dir horovod[pytorch]

# Install packages
RUN pip install --trusted-host pypi.python.org -r ./requirements.txt

RUN useradd -r -u 1000 -p $(openssl passwd -1 password) appuser

RUN usermod -a -G root appuser

RUN usermod -a -G sudo appuser

ENV HOME="/app"

# Install aws cli
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
RUN unzip awscliv2.zip
RUN sudo ./aws/install
RUN rm -rf awscli*

# Set correct permissions
RUN chown -R appuser /app