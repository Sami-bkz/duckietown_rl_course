FROM ros:noetic-ros-core-focal

# Installer les outils de base
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    wget \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Installer Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh

# Ajouter conda au PATH
ENV PATH="/opt/conda/bin:${PATH}"

# Configurer le shell pour conda
RUN echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc

# Créer l'environnement Conda à partir du fichier environment.yml
WORKDIR /app
COPY environment.yml .
RUN conda env create -f environment.yml


# Installation de toutes les dépendances graphiques nécessaires
RUN apt-get update && apt-get install -y \
    pkg-config \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libgles2 \
    libglvnd-dev \
    libgl1-mesa-dev \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    libglfw3-dev \
    libglu1-mesa \
    libglu1-mesa-dev \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libosmesa6 \
    libosmesa6-dev \
    mesa-utils \
    mesa-utils-extra \
    xvfb \
    x11-utils \
    patchelf \
    libxrandr-dev \
    libxinerama-dev \
    libxcursor-dev \
    libxi-dev \
    libxext-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Configuration des variables d'environnement
ENV DISPLAY=:1
ENV ROBOT_NAME="paperino"
ENV MESA_GL_VERSION_OVERRIDE=3.3
ENV PYGLET_GRAPHICS_INSTALLATION=headless
ENV PYOPENGL_PLATFORM=egl

# Ajouter la configuration Xvfb au .bashrc pour qu'elle s'exécute à chaque ouverture de console
RUN echo '# Configuration automatique de Xvfb' >> /root/.bashrc && \
    # echo 'pkill Xvfb > /dev/null 2>&1 || true' >> /root/.bashrc && \
    # echo 'Xvfb :1 -screen 0 1920x1080x24 -ac +extension GLX +render -noreset &' >> /root/.bashrc && \
    # echo 'sleep 1' >> /root/.bashrc && \
    echo 'export DISPLAY=localhost:10.0' >> /root/.bashrc && \
    # echo 'export MESA_GL_VERSION_OVERRIDE=3.3' >> /root/.bashrc && \
    # echo 'export PYGLET_GRAPHICS_INSTALLATION=headless' >> /root/.bashrc && \
    # echo 'export PYOPENGL_PLATFORM=egl' >> /root/.bashrc && \
    # echo 'export PYGLET_SHADOW_WINDOW=1' >> /root/.bashrc && \
    # echo 'export PYOPENGL_PLATFORM=osmesa' >> /root/.bashrc && \
    # echo 'export QT_X11_NO_MITSHM=1' >> /root/.bashrc && \
    # echo 'echo "Xvfb et variables OpenGL configurés automatiquement."' >> /root/.bashrc && \
    echo 'export PYTHONUNBUFFERED=1' >> /root/.bashrc && \
    echo 'export MESA_GL_VERSION_OVERRIDE=3.3' >> /root/.bashrc && \
    echo 'export PYGLET_DEBUG_GL=True' >> /root/.bashrc && \
    echo 'export PYGLET_SHADOW_WINDOW=0' >> /root/.bashrc && \
    echo 'conda activate duckietownrl' >> /root/.bashrc


RUN mkdir -p /home/duckietown_rl_course/
WORKDIR /home/duckietown_rl_course
COPY setup.py /home/duckietown_rl_course/
COPY README.md /home/duckietown_rl_course/
RUN /opt/conda/bin/conda run -n duckietownrl pip install -e .
RUN apt-get update && apt-get install -y x11-apps && rm -rf /var/lib/apt/lists/*
RUN /opt/conda/bin/conda run -n duckietownrl pip uninstall -y pyglet
RUN /opt/conda/bin/conda run -n duckietownrl pip install pyglet==1.5.11
RUN /opt/conda/bin/conda run -n duckietownrl pip install pygame



