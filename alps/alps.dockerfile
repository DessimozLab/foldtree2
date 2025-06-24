FROM nvcr.io/nvidia/pytorch:24.01-py3

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV VENV_PATH=/workspace/gemma-venv

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3-pip \
    python3-dev \
    git \
    wget \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create workspace directory
RUN mkdir -p /workspace
WORKDIR /workspace

# Create virtual environment with system site packages
RUN python3.10 -m venv --system-site-packages ${VENV_PATH}
ENV PATH="${VENV_PATH}/bin:$PATH"

# Upgrade pip
RUN ${VENV_PATH}/bin/pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
RUN ${VENV_PATH}/bin/pip install --no-cache-dir torch==2.0.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# Install PyTorch Geometric and its CUDA-dependent packages
RUN ${VENV_PATH}/bin/pip install --no-cache-dir \
    torch-scatter==2.1.1 \
    torch-sparse==0.6.17 \
    torch-cluster==1.6.1 \
    torch-spline-conv==1.2.2 \
    torch-geometric==2.3.1 \
    --extra-index-url https://data.pyg.org/whl/torch-2.0.1+cu118.html

# Install scientific and deep learning packages
RUN ${VENV_PATH}/bin/pip install --no-cache-dir \
    numpy==1.24.3 \
    scipy==1.10.1 \
    pandas==1.5.3 \
    pytorch-lightning==2.0.6 \
    matplotlib==3.7.2 \
    networkx==3.1 \
    biopython==1.81 \
    pydssp==0.9.0 \
    h5py==3.9.0 \
    wget==3.2 \
    tqdm==4.65.0 \
    einops==0.6.1 \
    pebble==5.0.3 \
    datasketch==1.5.3 \
    urllib3==2.0.4

# Create directories for mounting data
RUN mkdir -p /workspace/datasets

# Create a non-root user
RUN useradd -m user && \
    chown -R user:user /workspace && \
    chown -R user:user ${VENV_PATH}
USER user

# Ensure the virtual environment is activated on login
RUN echo "source ${VENV_PATH}/bin/activate" >> ~/.bashrc

# Set entry point that activates virtual environment
ENTRYPOINT ["/bin/bash", "-c", "source ${VENV_PATH}/bin/activate && exec \"$@\"", "--"]
CMD ["/bin/bash"]