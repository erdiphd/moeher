
FROM nvidia/cuda:11.4.3-base-ubuntu20.04

RUN apt-get update && apt-get install --no-install-recommends -y 
    

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
  locales \
  git \
  libgl1-mesa-dri \
  menu \
  python \
  pip \
  mesa-utils \
  nano \
  net-tools \
  openbox \
  openssh-server \
  sudo \
  supervisor \
  terminator \
  tint2 \
  vim \
  x11-xserver-utils \
  x11vnc \
  xinit \
  xserver-xorg-video-dummy \
  xserver-xorg-input-void \
  websockify && \
  pip install supervisor-stdout && \
  apt-get -y clean

RUN apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    curl \
    git \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    patchelf \
    software-properties-common \
    net-tools \
    unzip \
    vim \
    virtualenv \
    wget \
    xpra \
    mpich \
    xserver-xorg-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


RUN locale-gen en_US.UTF-8 en_GB.UTF-8 de_DE.UTF-8 de_DE@euro
RUN echo 'LANG=en_US.UTF-8' >> /etc/locale.conf
# RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/user

# RUN adduser --disabled-password \
# --gecos '' user
# RUN adduser user sudo

RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> \
/etc/sudoers

# Create user
RUN useradd -s /bin/bash -d /home/user/ -m -G sudo user

RUN mkdir -p /home/user/.mujoco \
    && wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz -O mujoco.tar.gz \
    && tar -zxf mujoco.tar.gz --directory /home/user/.mujoco \
    && rm mujoco.tar.gz


# Mini Anaconda Installation

# # Mini Anaconda Installation

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash Miniconda3-latest-Linux-x86_64.sh -b -u -p /home/user/conda \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

USER user

ENV LD_LIBRARY_PATH /home/user/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH}
ENV LD_PRELOAD /usr/lib/x86_64-linux-gnu/libGLEW.so
ENV MUJOCO_GL "glfw"




# RUN git clone https://github.com/openai/gym /home/user/hgg_test/gym

RUN /bin/bash -c "source /home/user/conda/bin/activate && \ 
                  conda create --name her_ea python=3.8 -y"



RUN /bin/bash -c "source /home/user/conda/bin/activate && \
                  conda activate her_ea && \
                  pip install numpy==1.24.4 torch==2.1.1 nvidia-cublas-cu12==12.1.3.1 nvidia-cuda-cupti-cu12==12.1.105 nvidia-cuda-nvrtc-cu12==12.1.105 nvidia-cuda-runtime-cu12==12.1.105 nvidia-cudnn-cu12==8.9.2.26 nvidia-cufft-cu12==11.0.2.54 nvidia-curand-cu12==10.3.2.106 nvidia-cusolver-cu12==11.4.5.107 nvidia-cusparse-cu12==12.1.0.106 nvidia-nccl-cu12==2.18.1 nvidia-nvjitlink-cu12==12.3.101 nvidia-nvtx-cu12==12.1.105"


RUN /bin/bash -c "source /home/user/conda/bin/activate && \
                  conda activate her_ea && \
                  pip install scipy==1.10.1 six==1.16.0 pyglet==1.3.2 cloudpickle==1.2.2 opencv-python==4.8.1.78 mujoco-py==2.1.2.14 \
                  Cython==0.29.34 termcolor==2.3.0 beautifultable==1.1.0 deap==1.4.1 plotly==4.5.0 pandas==2.0.3 scikit-learn==1.3.2 \ 
                  scipy==1.10.1 matplotlib==3.7.4 tqdm==4.66.1"

RUN /bin/bash -c "source /home/user/conda/bin/activate && \ 
                  conda create --name cher python=3.6 -y"


RUN /bin/bash -c "source /home/user/conda/bin/activate && \
                  conda activate cher && \
                  pip install absl-py==1.4.0 astor==0.8.1 beautifultable==0.7.0 certifi==2021.5.30 cffi==1.15.1 click==8.0.4 cloudpickle==1.2.2 Cython==0.29.34 dataclasses==0.8 fasteners==0.19 future==0.18.3 gast==0.5.4 glfw==2.6.4 grpcio==1.48.2 gym-notices==0.0.8 imageio==2.15.0 importlib-metadata==4.8.3 joblib==1.1.1 lockfile==0.12.2 Markdown==3.3.7 mpi4py==3.1.5 mujoco-py==2.1.2.14 numpy==1.19.5 numpy-quaternion==2022.4.4 opencv-python==3.4.13.47 Pillow==8.4.0 protobuf==3.19.6 pycparser==2.21 pyglet==1.3.2 PyYAML==6.0.1 scikit-learn==0.24.2 scipy==1.5.4 six==1.16.0 tensorboard==1.9.0 tensorflow==1.9.0 termcolor==1.1.0 threadpoolctl==3.1.0 typing_extensions==4.1.1 Werkzeug==2.0.3 zipp==3.6.0"


RUN /bin/bash -c "source /home/user/conda/bin/activate && \ 
                  conda create --name mega python=3.6 -y"

RUN /bin/bash -c "source /home/user/conda/bin/activate && \
                  conda activate mega && \
                  pip install --upgrade pip \
                  pip install absl-py==0.9.0 cachetools==4.1.0 certifi==2020.4.5.1 cffi==1.14.0 chardet==3.0.4 cloudpickle==1.3.0 colour==0.1.5 cycler==0.10.0 Cython==0.29.17 dill==0.2.9 mujoco-py==2.1.2.14 numpy==1.19.5 numpy-quaternion==2022.4.4 opencv-python==3.4.13.47 Pillow==8.4.0 tabuate tensorboard==1.15 matplotlib"
# RUN /bin/bash -c "source /home/user/conda/bin/activate && \
#                   conda activate her_ea && \
#                   pip install numpy-quaternion gym PyYAML "

# RUN /bin/bash -c "source /home/user/conda/bin/activate && \
#                   conda activate her_ea && \
#                   pip install opencv-python==3.4.13.47  scipy==1.5.4 'pyglet<=1.3.2,>=1.2.0' cloudpickle~=1.2.0 mpi4py scikit-learn click "
# ENV DISPLAY :0

VOLUME [ "/home/user/her_ea/" ]

# ENV mujoco_env "FrankaPickAndPlace-v1"
# ENV log_path "log/train1"
# ENV n_epochs "200"



COPY startup.sh /home/user/
# RUN /home/user/.conda/envs/HGG/bin/python /home/user/HGG/train.py
ENV PATH /home/user/.conda/envs/her_ea/bin:$PATH
CMD ["bin/bash", "home/user/startup.sh"]
