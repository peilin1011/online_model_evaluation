# 使用NVIDIA提供的基础镜像，它已经包含了CUDA和cuDNN，便于安装PyTorch
FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

# 设置环境变量以确保容器内的编码和路径
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PATH=/opt/conda/bin:$PATH

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    git \
    python3 \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 安装Python依赖
RUN pip3 install --upgrade pip
RUN pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

# 安装hnswlib
RUN pip3 install hnswlib

# 安装Julia
RUN mkdir /julia && \
    curl -sSL https://julialang-s3.julialang.org/bin/linux/x64/1.7/julia-1.7.3-linux-x86_64.tar.gz | tar -xz -C /julia --strip-components=1 && \
    ln -s /julia/bin/julia /usr/local/bin/julia

# 安装PyJulia
RUN pip3 install julia && \
    python3 -c "import julia; julia.install()"

# 将项目代码复制到容器中
WORKDIR /workspace
COPY . /workspace

# 安装项目的Python依赖
RUN pip3 install -r requirements.txt


