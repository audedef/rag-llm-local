# Utilise une image Python légère comme base
FROM python:3.10-slim-bookworm

# Définit le répertoire de travail à l'intérieur du conteneur
WORKDIR /app

# Copie le fichier requirements.txt dans le conteneur
COPY requirements.txt .

# Installe les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Commande qui permet de garder le conteneur en vie et de l'exécuter avec docker compose exec
CMD ["tail", "-f", "/dev/null"]