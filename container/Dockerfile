## ROS2
FROM tensorflow/tensorflow:2.10.0-gpu
WORKDIR /root
RUN apt update -y
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=America/New_York
RUN apt install -y \
            software-properties-common \
            usbutils \
            tree \
            vim \
            less \
            htop \
            wget \
            curl \
            zsh \
            git
RUN apt update -y
RUN apt upgrade -y

# customizations
RUN wget -nv -O - https://raw.githubusercontent.com/zimfw/install/master/install.zsh | zsh
RUN wget https://gist.githubusercontent.com/saikrn112/b55bf8a8bad6f6538a40561bdafe45bc/raw/8ad92e6c07858266b6740790212c6b518e3a028b/.vimrc
RUN ls /root/.vimrc
RUN curl -fLo ~/.vim/autoload/plug.vim --create-dirs https://raw.githubusercontent.com/junegunn/vim-plug/master/plug.vim
RUN eval `vim -E -s -u "/root/.vimrc" +PlugInstall +qall`
RUN wget -O - https://gist.githubusercontent.com/saikrn112/1e353283a091824bb0074c7aafe203fe/raw/zimrc-components >> .zimrc
RUN /usr/bin/zsh /root/.zim/zimfw.zsh install
RUN wget -O - https://gist.githubusercontent.com/saikrn112/b9d16ad838fc240fe7370697039eb56c/raw/62d0ef0fabce1781c818b26b2a2acfe190ebc8fa/alias.sh >> .zshrc

# useful libraries
RUN python -m pip install --upgrade pip
RUN pip3 install opencv-python matplotlib scipy
RUN pip3 install scikit-image tqdm sklearn scikit-learn tflite-runtime

RUN apt-get update -y
RUN apt-get install libglu1-mesa libxi-dev libxmu-dev libglu1-mesa-dev -y
RUN apt install --allow-change-held-packages libcudnn8=8.1.0.77-1+cuda11.2 -y --allow-downgrades
CMD ["zsh"]
LABEL maintainer="saikrn112@gmail.com"

