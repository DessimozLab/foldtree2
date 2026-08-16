FROM nvcr.io/nvidia/pytorch:24.01-py3

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV VENV_PATH=/workspace/foldtree2-venv

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3-pip \
    python3-dev \
    git \
    wget \
    build-essential \
    ca-certificates \
    libstdc++6 \
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

# Install SE3 + Lightning stack and scientific dependencies.
# Keep base-image torch to avoid ABI/CUDA mismatch with NGC builds.
RUN ${VENV_PATH}/bin/pip install --no-cache-dir \
    "numpy>=1.23.5,<1.24" \
    "scipy>=1.10,<1.11" \
    "pandas>=1.5,<2.0" \
    "pytorch-lightning>=2.3,<2.4" \
    "lightning-utilities>=0.11,<0.12" \
    "torchmetrics>=1.4,<1.5" \
    "matplotlib>=3.7,<3.8" \
    "networkx>=3.1,<3.4" \
    biopython==1.79 \
    "pydssp>=0.9,<1.0" \
    "h5py>=3.9,<3.12" \
    wget==3.2 \
    "tqdm>=4.65,<5.0" \
    "einops>=0.6,<0.9" \
    "pebble>=5.0,<6.0" \
    "datasketch>=1.5,<1.7" \
    "urllib3>=2.0,<2.3" \
    "pyyaml>=6.0,<7.0" \
    prody==2.4.1 \
    gotennet-pytorch==0.2.2

# Install PyTorch Geometric + CUDA extension wheels that match the torch/cuda
# versions already bundled in the base image.
RUN set -eux; \
        TORCH_BASE="$(${VENV_PATH}/bin/python -c 'import re, torch; v=torch.__version__.split("+")[0]; m=re.match(r"(\\d+\\.\\d+\\.\\d+)", v); print(m.group(1) if m else ".".join(v.split(".")[:2])+".0")')"; \
    CUDA_TAG="$(${VENV_PATH}/bin/python -c 'import torch; c=torch.version.cuda or "cpu"; print("cpu" if c=="cpu" else "cu"+c.replace(".", ""))')"; \
    PYG_WHL="https://data.pyg.org/whl/torch-${TORCH_BASE}+${CUDA_TAG}.html"; \
    echo "Using PyG wheel index: ${PYG_WHL}"; \
    ${VENV_PATH}/bin/pip install --no-cache-dir \
      pyg_lib \
      torch_scatter \
      torch_sparse \
      torch_cluster \
      torch_spline_conv \
      torch_geometric \
            --no-build-isolation \
      -f "${PYG_WHL}"

# Install Foldcomp static binary (avoid source build issues from Python packaging).
RUN set -eux; \
    wget -O /tmp/foldcomp-linux-x86_64.tar.gz https://mmseqs.com/foldcomp/foldcomp-linux-x86_64.tar.gz; \
    mkdir -p /tmp/foldcomp-extract; \
    tar -xzf /tmp/foldcomp-linux-x86_64.tar.gz -C /tmp/foldcomp-extract; \
    FOLDCOMP_BIN="$(find /tmp/foldcomp-extract -type f -name foldcomp | head -n1)"; \
    test -n "${FOLDCOMP_BIN}"; \
    install -m 0755 "${FOLDCOMP_BIN}" /usr/local/bin/foldcomp; \
    foldcomp --help >/dev/null || true; \
    rm -rf /tmp/foldcomp-linux-x86_64.tar.gz /tmp/foldcomp-extract

# Clone project and install in editable mode so training scripts can import foldtree2.
ARG FOLDTREE2_REPO_URL=https://github.com/DessimozLab/foldtree2.git
ARG FOLDTREE2_REPO_REF=dev
RUN git clone --depth 1 --branch ${FOLDTREE2_REPO_REF} ${FOLDTREE2_REPO_URL} /workspace/foldtree2
WORKDIR /workspace/foldtree2
RUN ${VENV_PATH}/bin/pip install --no-cache-dir --no-deps -e .

# Build-time sanity check for Lightning + SE3 imports.
RUN ${VENV_PATH}/bin/python - <<'PY'
import torch
import pytorch_lightning as pl
import torch_geometric
from foldtree2.src.se3_struct_decoder import se3_denoiser
print('torch', torch.__version__, 'cuda', torch.version.cuda)
print('lightning', pl.__version__)
print('pyg', torch_geometric.__version__)
print('se3_denoiser', se3_denoiser.__name__)
PY

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