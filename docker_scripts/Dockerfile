#FROM robotlocomotion/drake
FROM ubuntu:18.04

RUN apt-get -y update
RUN apt-get install -y wget unzip git g++
RUN apt-get -y install libgl1-mesa-dev libgl1-mesa-glx libosmesa6-dev patchelf libopenmpi-dev libglew-dev libglfw3 libglfw3-dev

# for display through vnc
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y xfce4 tigervnc-standalone-server

RUN apt-get update
RUN apt-get -y install curl
RUN apt-get -y install python3-venv
RUN apt-get -y install ipython
RUN apt-get -y firefox

#-----------drake-----------------

# note: drake is incompatable with conda

RUN apt-get update

RUN curl -o drake.tar.gz https://drake-packages.csail.mit.edu/drake/nightly/drake-latest-bionic.tar.gz

RUN mkdir -p /python_venvs/drake_env
RUN tar -xvzf drake.tar.gz -C /python_venvs/drake_env --strip-components=1
RUN yes | /python_venvs/drake_env/share/drake/setup/install_prereqs
WORKDIR /python_venvs
RUN python3 -m venv drake_env --system-site-packages
#---------------------------------

# utilities 
RUN apt-get -y install vim
RUN apt-get -y install tmux
RUN apt-get -y install openssh-server
RUN apt-get install sudo


# this means root can ssh in with a password
#RUN echo "PermitRootLogin yes"  >> /etc/ssh/sshd_config

RUN apt-get install xfce4-terminal
# changing the terminal config
RUN printf "2\n" | update-alternatives --config x-terminal-emulator

ARG password
ARG id
ARG user

ENV PASS=$password
#RUN printf "$password\n$password\n" | passwd
#ENV LANG=en_US.UTF-8 
#ENV LC_ALL=en_US.UTF-8.




RUN groupadd -g 1004 agrobenj
RUN useradd -s /bin/bash -u $id -g $id -m $user && echo "$user:$password" | chpasswd && adduser $user sudo

RUN chown -R agrobenj /python_venvs 
ENV PATH="/python_venvs/drake_env/bin:$PATH"

USER agrobenj

RUN pip install mjcf2urdf
RUN pip install pandas
RUN pip install altair
RUN pip install plotly
RUN pip install pylint
RUN git clone https://github.com/Khodeir/furniture.git ~/furniture

RUN echo "source /python_venvs/drake_env/bin/activate" >> ~/.bashrc
#RUN echo "export LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8." >> ~/.bashrc

WORKDIR /home/$user/workspace

COPY .tmux.conf /home/$user/.tmux.conf
COPY .vimrc /home/$user/.vimrc


COPY entrypoint.sh /entrypoint.sh
ENTRYPOINT ["bash","/entrypoint.sh"]

CMD ["bash"]

