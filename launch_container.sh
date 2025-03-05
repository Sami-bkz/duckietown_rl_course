#!/bin/bash


CURRENT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Créer le fichier d'autorisation X
XAUTH=/tmp/.docker.xauth
touch $XAUTH
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -

# Vérifie si le conteneur existe déjà et s'il est en cours d'exécution
CONTAINER_RUNNING=$(docker ps -q -f name=duckie-container)
CONTAINER_EXISTS=$(docker ps -aq -f name=duckie-container)

if [ -n "$CONTAINER_RUNNING" ]; then
    # Le conteneur est en cours d'exécution, on s'y connecte simplement
    echo "Connexion au conteneur duckie-container déjà en cours d'exécution..."
    docker exec -it duckie-container bash
elif [ -n "$CONTAINER_EXISTS" ]; then
    # Le conteneur existe mais n'est pas en cours d'exécution, on le démarre
    echo "Démarrage du conteneur duckie-container existant..."
    docker start duckie-container
    docker exec -it duckie-container bash
else
    # Le conteneur n'existe pas, on le crée et on le démarre
    echo "Création et démarrage d'un nouveau conteneur duckie-container..."
    docker run -it --name duckie-container \
      --gpus all \
      -e DISPLAY=$DISPLAY \
      -e XAUTHORITY=$XAUTH \
      -v $XAUTH:$XAUTH \
      -v /tmp/.X11-unix:/tmp/.X11-unix \
      -v $CURRENT_DIR:/home/duckietown_rl_course/\
      --device /dev/input:/dev/input \
      --privileged \
      --network=host \
      --ipc=host \
      duckie-course bash
fi
