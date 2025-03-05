# Docker pour les Noobs - Commandes Essentielles

## Gérer les Conteneurs

### Lister les conteneurs

```bash
# Voir tous les conteneurs en cours d'exécution
docker ps

# Voir tous les conteneurs (même arrêtés)
docker ps -a
```

### Lancer des conteneurs

```bash
# Lancer un conteneur simple
docker run -d --name mon-conteneur image:tag

# Avec mappage de ports (port-hôte:port-conteneur)
docker run -d -p 8080:80 --name mon-site nginx

# Avec montage de volume (dossier-hôte:dossier-conteneur)
docker run -d -v /chemin/local:/chemin/conteneur --name mon-app image:tag
```

### Arrêter et supprimer des conteneurs

```bash
# Arrêter un conteneur
docker stop mon-conteneur

# Supprimer un conteneur
docker rm mon-conteneur

# 🔥 Arrêter TOUS les conteneurs en cours
docker stop $(docker ps -q)

# 🔥 Supprimer TOUS les conteneurs (même arrêtés)
docker rm $(docker ps -aq)
```

## Gérer les Images

```bash
# Lister les images
docker images

# Télécharger une image
docker pull ubuntu:latest

# Supprimer une image
docker rmi image:tag

# 🔥 Supprimer toutes les images inutilisées
docker image prune -a
```

## Logs et Inspection

```bash
# Voir les logs d'un conteneur
docker logs mon-conteneur

# Voir les logs en continu (suivre)
docker logs -f mon-conteneur

# Inspecter un conteneur
docker inspect mon-conteneur
```

## Entrer dans un Conteneur

```bash
# Ouvrir un terminal dans un conteneur en cours d'exécution
docker exec -it mon-conteneur bash
```

## Docker Compose (pour multi-conteneurs)

```bash
# Démarrer les services définis dans docker-compose.yml
docker-compose up -d

# Arrêter les services
docker-compose down
```

## Nettoyage Global

```bash
# 🧹 Nettoyage complet (conteneurs, images, réseaux, volumes non utilisés)
docker system prune -a
```

---
*Note: Remplacez `mon-conteneur` et `image:tag` par vos noms réels de conteneurs et d'images*