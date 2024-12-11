#!/bin/bash

# 设置变量
IMAGE_NAME="vlattack_gpu_image"    # Docker 镜像名称
CONTAINER_NAME="container-1"  # 容器名称
LOCAL_DIR="/home/peiyue/online_model_evaluation"   # 本地目录
CONTAINER_DIR="/app"   # 容器中的目录
PORT_MAPPING="8888:8888"             # 主机端口:容器端口

# 检查容器是否已经存在
if [ "$(docker ps -a -q -f name=${CONTAINER_NAME})" ]; then
    echo "容器 ${CONTAINER_NAME} 已存在。启动中..."
    docker start ${CONTAINER_NAME}
else
    echo "容器 ${CONTAINER_NAME} 不存在。正在创建..."
    docker run --rm -it -d \
        --gpus "device=6" \
        --name ${CONTAINER_NAME} \
        -v ${LOCAL_DIR}:${CONTAINER_DIR} \
        -p ${PORT_MAPPING} \
        ${IMAGE_NAME}
fi

# 确保requirements.txt安装依赖
echo "安装requirements.txt中的依赖..."
docker exec -it ${CONTAINER_NAME} bash -c "cd ${CONTAINER_DIR} && pip install -r requirements.txt"

# 附加到容器
echo "附加到容器 ${CONTAINER_NAME}..."
docker exec -it ${CONTAINER_NAME} bash
