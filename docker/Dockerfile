
FROM pytorch/pytorch as base

ENV DEBIAN_FRONTEND noninteractive

ARG USERNAME=dexter
ARG UID=$UID
ARG GID=$GID
# Set environment variables
ENV USERNAME ${USERNAME}
ENV HOME /home/${USERNAME}


RUN apt-get update && \
      apt-get -y install sudo

RUN sudo apt-get install libx11-dev -y 

# Create user and setup permissions on /etc/sudoers
RUN addgroup --gid ${GID} ${USERNAME} \
 && adduser --disabled-password --gecos '' --uid ${GID} --gid ${GID} ${USERNAME} \
 && echo ${USERNAME} ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/${USERNAME} \
 && chown -R ${UID}:${GID} /home/${USERNAME} 

RUN sudo apt install build-essential -y

RUN sudo apt-get install libglfw3 libglew2.1 libgl1-mesa-glx libosmesa6 -y
RUN conda env config vars set MUJOCO_GL=egl PYOPENGL_PLATFORM=egl

RUN pip install torchrl
RUN pip install tqdm
# Visualization app
RUN python -m pip install PyQt5
RUN pip install matplotlib
RUN pip install pybullet


RUN pip install mujoco

RUN pip install roboticstoolbox-python

RUN echo 'source /opt/conda/bin/activate base' >> ~/.bashrc 

USER $USERNAME

# ENTRYPOINT [ "entrypoint.sh" ]
