ARG CUDA_VERSION=12.1.1
ARG UBUNTU_VERSION=22.04
ARG UV_VERSION=0.11.6

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

########################################################################

FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-runtime-ubuntu${UBUNTU_VERSION}

ARG TARGETARCH
ARG PYTHON_VERSION=3.11

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# build-essential: needed to build the few dependencies that only ship sdists
# libGL1: needed to avoid following error when using cv2
# ImportError: libGL.so.1: cannot open shared object file: No such file or directory
# See https://stackoverflow.com/questions/55313610/importerror-libgl-so-1-cannot-open-shared-object-file-no-such-file-or-directo
RUN apt-get update && \
    apt-get install -y wget=1.* build-essential libgl1 curl git tree && \
    apt-get autoremove && apt-get autoclean && apt-get clean

# needed for jupyter lab extensions
RUN curl -fsSL https://deb.nodesource.com/node_16.x | bash - && \
    apt-get install -y nodejs

# Install pandoc, needed for rendering notebooks
# Get latest release link from here: https://github.com/jgm/pandoc/releases
RUN wget \
    https://github.com/jgm/pandoc/releases/download/3.1.12.2/pandoc-3.1.12.2-1-${TARGETARCH}.deb && \
    dpkg -i pandoc-3.1.12.2-1-${TARGETARCH}.deb && \
    rm pandoc-3.1.12.2-1-${TARGETARCH}.deb

# This gets rid of the following error when importing cv2 on arm64.
# We cannot use the ENV directive since it cannot be used conditionally.
# See https://github.com/opencv/opencv/issues/14884
# ImportError: /lib/aarch64-linux-gnu/libGLdispatch.so.0: cannot allocate memory in static TLS block
RUN if [ "${TARGETARCH}" = "arm64" ]; \
    then echo "export LD_PRELOAD=/lib/aarch64-linux-gnu/libGLdispatch.so.0:$LD_PRELOAD" >> /root/.bashrc; fi

#------------------------------------------------------------------------
# Python and all dependencies are installed by uv from uv.lock.
# GDAL is not needed: the rasterio wheels bundle it.

COPY --from=uv /uv /uvx /usr/local/bin/

ENV UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_PYTHON=${PYTHON_VERSION} \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1
ENV VIRTUAL_ENV=/opt/venv
ENV PATH=/opt/venv/bin:$PATH

RUN uv python install ${PYTHON_VERSION}

WORKDIR /opt/src/

# Install third-party dependencies first, in their own layer, so that it can
# be cached independently of changes to the Raster Vision source code.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=rastervision_pipeline/pyproject.toml,target=rastervision_pipeline/pyproject.toml \
    --mount=type=bind,source=rastervision_aws_s3/pyproject.toml,target=rastervision_aws_s3/pyproject.toml \
    --mount=type=bind,source=rastervision_aws_batch/pyproject.toml,target=rastervision_aws_batch/pyproject.toml \
    --mount=type=bind,source=rastervision_aws_sagemaker/pyproject.toml,target=rastervision_aws_sagemaker/pyproject.toml \
    --mount=type=bind,source=rastervision_core/pyproject.toml,target=rastervision_core/pyproject.toml \
    --mount=type=bind,source=rastervision_pytorch_learner/pyproject.toml,target=rastervision_pytorch_learner/pyproject.toml \
    --mount=type=bind,source=rastervision_pytorch_backend/pyproject.toml,target=rastervision_pytorch_backend/pyproject.toml \
    uv sync --frozen --no-install-workspace --all-packages --all-extras --all-groups

#------------------------------------------------------------------------

COPY scripts /opt/src/scripts/
COPY scripts/rastervision /usr/local/bin/rastervision
COPY tests /opt/src/tests/
COPY integration_tests /opt/src/integration_tests/
COPY .flake8 /opt/src/.flake8
COPY .coveragerc /opt/src/.coveragerc
COPY pyproject.toml uv.lock README.md LICENSE /opt/src/

COPY ./rastervision_pipeline/ /opt/src/rastervision_pipeline/
COPY ./rastervision_aws_s3/ /opt/src/rastervision_aws_s3/
COPY ./rastervision_aws_batch/ /opt/src/rastervision_aws_batch/
COPY ./rastervision_aws_sagemaker/ /opt/src/rastervision_aws_sagemaker/
COPY ./rastervision_core/ /opt/src/rastervision_core/
COPY ./rastervision_pytorch_learner/ /opt/src/rastervision_pytorch_learner/
COPY ./rastervision_pytorch_backend/ /opt/src/rastervision_pytorch_backend/

# Install the Raster Vision packages themselves in editable mode.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --all-packages --all-extras --all-groups

# Needed for this image to be used by the AWS SageMaker PyTorch Estimator.
# This is installed outside of the lockfile because its dependency,
# sagemaker-training, requires protobuf versions that are incompatible with
# the rest of the locked dependencies.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install sagemaker_pytorch_training==2.8.1
ENV SAGEMAKER_TRAINING_MODULE=sagemaker_pytorch_container.training:main

#------------------------------------------------------------------------

ENTRYPOINT ["bash", "-l", "-c"]
CMD ["bash"]
