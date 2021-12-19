FROM nvcr.io/nvidia/pytorch:21.11-py3

# Install ubuntu dependencies
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y sudo poppler-utils curl git gcc vim wget unzip\
                              lsb-core software-properties-common llvm-9-dev

## Install NCCL2
RUN apt-get -y remove libnccl2 libnccl-dev
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
RUN add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
RUN apt-get update && sudo apt-get install -y libnccl2=2.9.9-1+cuda11.3 libnccl-dev=2.9.9-1+cuda11.3

# Setup user
RUN useradd --uid 1000 --shell /bin/bash appliedml85
RUN usermod -aG sudo appliedml85
RUN echo "appliedml85 ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers
ADD . /app
RUN chown -R appliedml85 /app
ENV HOME /app
WORKDIR /app
USER appliedml85
ENV PATH /app/.local/bin:$PATH

# Install other requirements
RUN pip install --trusted-host pypi.python.org -r ./requirements.txt

# Install Deepspeed
RUN git clone https://github.com/microsoft/DeepSpeed.git
WORKDIR DeepSpeed
RUN sed -i 's/triton/triton==0.4.2/g' requirements/requirements-sparse_attn.txt
RUN DS_BUILD_SPARSE_ATTN=1 pip install ./ --no-cache-dir
WORKDIR /app
RUN rm -rf DeepSpeed

# Install Custom Triton
RUN git clone -b v0.4_cuda11.3_ngc https://github.com/appliedml85/triton.git
WORKDIR triton/python
RUN pip install -e .
WORKDIR /app

# Install OpenMPI
RUN sudo mv /usr/local/mpi /usr/local/bk_mpi
RUN wget https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.0.tar.gz
RUN gunzip -c openmpi-4.0.0.tar.gz | tar xf -
WORKDIR openmpi-4.0.0
RUN ./configure --prefix=/usr/local/openmpi
RUN make -j 8 all
RUN sudo make install
ENV LD_LIBRARY_PATH /usr/local/openmpi/lib:$LD_LIBRARY_PATH
ENV PATH /usr/local/openmpi/bin:$PATH
WORKDIR /app
RUN rm -rf openmpi*

# Install Horovod
RUN HOROVOD_NCCL_INCLUDE=/usr/include HOROVOD_NCCL_LIB=/usr/lib/x86_64-linux-gnu HOROVOD_GPU_OPERATIONS=NCCL\
    HOROVOD_WITH_MPI=1 HOROVOD_WITH_PYTORCH=1 HOROVOD_WITHOUT_TENSORFLOW=1 HOROVOD_WITHOUT_MXNET=1\
    pip install --no-cache-dir horovod[pytorch] \
