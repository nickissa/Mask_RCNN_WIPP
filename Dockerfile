# Python version 3.9.16
# Tensorflow 2
# FROM scratch
# FROM pengchuanzhang/pengchuanzhang/maskrcnn:latest
FROM python:3.9.16

# Create working directory to be /app/MASK_RCNN
WORKDIR /app
RUN mkdir MASK_RCNN
WORKDIR /app/MASK_RCNN

# Default Environmental Variables
ENV ROOTDIR="/app/MASK_RCNN/"

COPY . /app/MASK_RCNN

# Install the shell binary
# RUN apk add --no-cache bash

# Download anaconda installer script
RUN apt-get update && \
    apt-get install -y wget && \
    wget https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh -O ~/anaconda.sh

# Install Anaconda and set the system path
RUN /bin/bash ~/anaconda.sh -b -p /opt/anaconda && \
    rm ~/anaconda.sh && \
    echo "export PATH=/opt/anaconda/bin:$PATH" >> ~/.bashrc && \
    /opt/anaconda/bin/conda init bash

# Set default shell to Bash and add 'conda' command to the PATH env variable
SHELL ["/bin/bash", "--login", "-c"]
ENV PATH /opt/anaconda/bin/:$PATH

# Create the maskrcnn environment and activate it
RUN conda env create -f environment.yml

RUN conda init bash && \
    source ~/.bashrc

RUN pip install opencv-python

RUN source activate maskrcnn

# Run ARCMaskRCNN.py when container is run
# Curr DIR is /app/MASK_RCNN
CMD ["conda", "run", "-n", "maskrcnn", "python", "/app/MASK_RCNN/samples/ARCMaskRCNN.py"]
