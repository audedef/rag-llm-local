# Utilise une image Python légère comme base
FROM python:3.10-slim-bookworm

# Définit le répertoire de travail à l'intérieur du conteneur
WORKDIR /app

# Copie le fichier requirements.txt dans le conteneur
COPY requirements.txt .

# Installe les dépendances Python
# --no-cache-dir pour ne pas stocker le cache pip et réduire ainsi la taille de l'image
RUN pip install --no-cache-dir -r requirements.txt

# Si votre application expose une API web (Flask/FastAPI), je dois décommenter ceci
# EXPOSE 8000

# Commande qui permet de garder le conteneur en vie et de l'exécuter avec docker compose exec
CMD ["tail", "-f", "/dev/null"]