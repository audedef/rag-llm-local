# rag-llm-local

Ce projet implémente un système RAG (Retrieval-Augmented Generation) en local en utilisant Ollama pour le LLM (avec Gemma) et les embeddings, et PostgreSQL avec l'extension pgvector comme base de données vectorielle. L'ensemble de l'architecture est orchestré avec Docker Compose.

## Architecture

Le système est composé de trois services principaux :

1.  **ollama**: Héberge le modèle Gemma et fournit l'API pour les inférences LLM et la génération d'embeddings.
2.  **postgres-pgvector**: Une base de données PostgreSQL avec l'extension `pgvector` activée pour stocker et rechercher des embeddings vectoriels.
3.  **rag-app**: L'application Python qui gère :
    * L'ingestion de la base de connaissance personnalisée (fichiers texte, PDF, etc.).
    * La récupération des informations pertinentes depuis la base de données vectorielle.
    * L'interaction avec le LLM Ollama pour générer des réponses enrichies.

## Prérequis

* Docker Desktop (ou Docker Engine et Docker Compose) installé.

## Démarrage du Projet

1.  **Cloner le dépôt :**
    ```bash
    git clone [https://github.com/votre-utilisateur/rag-ollama-local.git](https://github.com/votre-utilisateur/rag-ollama-local.git)
    cd rag-ollama-local
    ```

2.  **Placer votre base de connaissance :**
    Déposez vos fichiers (ex: `.txt`, `.pdf`) dans le répertoire `rag-app/app/knowledge_base/`.

3.  **Démarrer les services Docker :**
    ```bash
    docker compose up --build -d
    ```
    Cela va construire l'image `rag-app`, démarrer Ollama, PostgreSQL et `rag-app`.
    *Note :* La première fois, Ollama téléchargera le modèle Gemma, ce qui peut prendre du temps.

4.  **Vérifier l'état des services :**
    ```bash
    docker compose ps
    ```
    Attendez que tous les services soient "healthy".

5.  **Ingérer la base de connaissance :**
    Exécutez le script d'ingestion depuis le conteneur `rag-app` :
    ```bash
    docker compose exec rag-app python -m app.ingest
    ```
    Cela va traiter vos documents, créer des embeddings et les stocker dans PostgreSQL.

6.  **Interroger le système RAG :**
    Exécutez le script de requête depuis le conteneur `rag-app` :
    ```bash
    docker compose exec rag-app python -m app.query
    ```
    Vous pourrez alors taper vos questions dans le terminal.

## Nettoyage

Pour arrêter et supprimer les conteneurs et les volumes Docker (cela supprimera aussi les données Ollama et PostgreSQL) :
```bash
docker compose down -v