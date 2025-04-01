FROM nvcr.io/nvidia/pytorch:24.01-py3

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV VENV_PATH=/capstor/store/cscs/swissai/prep01/venv

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



#mount the capstor
RUN mkdir -p /capstor && mount -t nfs server:/capstor /capstor


# Create workspace directory
RUN mkdir -p /workspace
WORKDIR /workspace

# add the gemma env directory
RUN mkdir -p /capstor/store/cscs/swissai/prep01/venv

# Create virtual environment with system site packages
RUN python3.10 -m venv --system-site-packages ${VENV_PATH}
ENV PATH="${VENV_PATH}/bin:$PATH"

# Upgrade pip
RUN ${VENV_PATH}/bin/pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch Geometric and its CUDA-dependent packages (without version numbers)
RUN ${VENV_PATH}/bin/pip install --no-cache-dir \
    torch-scatter \
    torch-sparse \
    torch-cluster \
    torch-spline-conv \
    torch-geometric \
    --extra-index-url https://data.pyg.org/whl/torch-$(python -c "import torch; print(torch.__version__.split('+')[0])")+cu$(python -c "import torch; print(torch.version.cuda.replace('.',''))")

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

#activate the env
RUN source ${VENV_PATH}/bin/activate

#test python venv
RUN python --version
RUN pip --version

#output the venv path
RUN echo $VENV_PATH

