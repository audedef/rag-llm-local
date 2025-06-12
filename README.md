# rag-llm-local

Ce projet implémente un système RAG (Retrieval-Augmented Generation) en local en utilisant Ollama pour le LLM (avec Gemma) et les embeddings, et PostgreSQL avec l'extension pgvector comme base de données vectorielle. L'ensemble de l'architecture est orchestré avec Docker Compose.

## Architecture

Le système est composé de trois services principaux :

1.  **ollama**: Héberge le modèle Gemma et fournit l'API pour les inférences LLM et la génération d'embeddings.
2.  **postgres-pgvector**: Une base de données PostgreSQL avec l'extension `pgvector` activée pour stocker et rechercher des embeddings vectoriels.
3.  **rag_app**: L'application Python qui gère :
    * L'ingestion de la base de connaissance personnalisée (fichiers texte, PDF, etc.).
    * La récupération des informations pertinentes depuis la base de données vectorielle.
    * L'interaction avec le LLM Ollama pour générer des réponses enrichies.

## Prérequis

* Docker Desktop (ou Docker Engine et Docker Compose) installé sur votre machine.

## Démarrage du Projet

1.  **Cloner le dépôt :**
    ```bash
    git clone [https://github.com/audedef/rag-llm-local.git]
    cd rag-llm-local
    ```

2.  **Placer votre base de connaissance :**
    Déposez vos fichiers (ex: `.txt`, `.pdf`) dans le répertoire `rag_app/knowledge_base/`.
    *Note :* Vous pouvez utiliser également Hugging Face et sa lib Dataset pour l'importer directement.

3.  **Démarrer les services Docker :**
    Commencez par ouvrir Docker Desktop puis depuis rag-llm-local lancez la commande :
    ```bash
    docker-compose up --build -d
    ```
    Cela va construire l'image `rag_app`, démarrer Ollama, PostgreSQL et `rag_app`.

4.  **Vérifier l'état des services :**
    ```bash
    docker-compose ps
    ```
    Attendez que tous les services soient "healthy".

5. **Télécharger le modèle Gemma sur Ollama si ce n'est pas déjà fait :**
    ```bash
    docker exec -it ollama_server ollama run gemma3:4b
    ```
    On utilise un petit modèle qui requiert une mémoire de seulement 3.3GiB
    *Note :* La première fois, Ollama téléchargera le modèle Gemma, ce qui peut prendre du temps.
    
    **Puis le modèle d'embedding car Gemma3 ne dispose pas de cette fonctionnalité :**
    ```bash
    docker exec -it ollama_server ollama pull nomic-embed-text
    ```
    *Note 2 :* Vérifiez si les modèles sont toujours dans le conteneur avec :
    ```bash
    docker exec -it ollama_server ollama list
    ```
    Dans ce cas, pas besoin de le re-télécharger.

6. **Ingérer la base de connaissance :**
    Exécutez le script d'ingestion depuis le conteneur `rag-app` :
    ```bash
    docker compose exec rag-app python ingest.py
    ```
    Cela va traiter vos documents, créer des embeddings et les stocker dans PostgreSQL.

7.  **Interroger le système RAG :**
    Exécutez le script de requête depuis le conteneur `rag-app` :
    ```bash
    docker compose exec rag_application python query.py
    ```
    Vous pourrez alors taper vos questions dans le terminal.

## Nettoyage

Pour arrêter et supprimer les conteneurs et les volumes Docker (cela supprimera aussi les données Ollama et PostgreSQL) :
```bash
docker compose down -v
```

## Evaluation

Afin de prouver l'efficacité du système RAG, un script d'évaluation est à votre disposition.
Il comporte déjà un ensemble de questions prédéfinies, basés sur la base de connaissance Hugging Face ingérée.
Si vous souhaitez évaluer votre propre RAG, modifiez les questions dans le script evaluate.py en vous basant sur votre propre base de connaissances. 
Puis exécutez le script avec :
```bash
docker compose exec rag_application python evaluate.py
```
Le résultat sortira sous forme de dataframe dans votre terminal, et sous forme de fichier markdown téléchargé dans votre dossier local.