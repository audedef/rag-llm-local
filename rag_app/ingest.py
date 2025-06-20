import os
# import requests

# Chargeurs de documents et splitters de texte en chunks de LangChain
from langchain_community.document_loaders import HuggingFaceDatasetLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Modèle d'embedding via Ollama
from langchain_community.embeddings import OllamaEmbeddings

# Stockage vectoriel avec PGVector
from langchain_community.vectorstores.pgvector import PGVector

# Ignore les avertissements liés à l'utilisation de pandas avec pyarrow
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pandas")

# 1. Configuration Initiale

# Configuration de la connexion à la base de données PostgreSQL (les mêmes valeurs que dans le docker-compose)
DB_USER = os.getenv("POSTGRES_USER", "rag_user")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "rag_password")
DB_HOST = os.getenv("POSTGRES_HOST", "postgres_db")
DB_PORT = os.getenv("POSTGRES_PORT", "5433")
DB_NAME = os.getenv("POSTGRES_DB", "rag_db")

# URL de connexion
CONNECTION_STRING = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Nom de la table dans la base de données vectorielle
COLLECTION_NAME = "rag_knowledge_base"

# Nom du dataset sur Hugging Face et la colonne contenant le texte à indexer dans la base de connaissances
DATASET_NAME = "neural-bridge/rag-dataset-12000"
PAGE_CONTENT_COLUMN = "context"

# Modèle d'embedding à utiliser avec Ollama
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text:latest")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL").strip(
) if os.getenv("OLLAMA_BASE_URL") else None
if OLLAMA_BASE_URL is None:
    print("Erreur: OLLAMA_BASE_URL n'est pas défini. Veuillez vérifier le docker-compose.yml")
    exit(1)
# "http://ollama_server:11434" si dans le docker-compose
# "http://localhost:11434" si en local

print("La configuration est bien chargée :")
print(f"Dataset: {DATASET_NAME}")
print(f"Modèle d'embedding: {EMBEDDING_MODEL}")
print(f"Collection PGVector: {COLLECTION_NAME}")

# 2. Chargement des données
print("\nÉtape 1: Chargement des documents depuis Hugging Face")

# Utilise le chargeur de LangChain pour charger un jeu de données depuis le Hub.
# On ne va charger qu'un échantillon (100 premières lignes) pour l'exercice.
loader = HuggingFaceDatasetLoader(
    DATASET_NAME, PAGE_CONTENT_COLUMN)
documents = list(loader.load())[:100]

print(f"{len(documents)} documents ont été chargés.")

# 3. Découpage des documents : le Chunking
print("\nÉtape 2: Découpage des documents en chunks")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # Taille maximale d'un chunk (en nombre de caractères)
    chunk_overlap=50
)
chunks = text_splitter.split_documents(documents)

print(f"Les documents ont été découpés en {len(chunks)} chunks.")

# 4. Création des embeddings et stockage vectoriel
print("\nÉtape 3: Génération des embeddings et stockage dans PGVector")
print("Cette étape peut prendre plusieurs minutes, en fonction du volume de données et de la puissance de la machine.")


# BLOC DE DEBUG
# affiche un échantillon de chunk et teste une requête d'embedding
""" if chunks:
    sample_chunk_content = chunks[0].page_content
    print(f"\n--- Diagnostic: Tentative de requête d'embedding pour un échantillon ---")
    print(f"Échantillon de texte: '{sample_chunk_content[:100]}...'")
    try:
        ollama_embeddings_url = f"{OLLAMA_BASE_URL}/api/embeddings"
        payload = {
            "model": EMBEDDING_MODEL,
            "prompt": sample_chunk_content
        }
        print(f"Envoi de la requête à: {ollama_embeddings_url}")
        print(
            f"Payload (partiel): {{'model': '{EMBEDDING_MODEL}', 'prompt': '{sample_chunk_content[:50]}...'}}")

        # Timeout augmenté à 120s
        response = requests.post(
            ollama_embeddings_url, json=payload, timeout=120)

        print(f"Code de statut HTTP: {response.status_code}")
        print(
            f"Type de contenu de la réponse: {response.headers.get('Content-Type')}")
        print(f"Réponse brute (tronquée à 500 caractères):")
        print(response.text[:500])

        # Lève une exception pour les codes d'erreur HTTP (4xx ou 5xx)
        response.raise_for_status()

        # Tente de parser la réponse JSON
        json_response = response.json()
        if "embedding" in json_response:
            print(
                f"Embedding pour l'échantillon généré avec succès. Longueur: {len(json_response['embedding'])}")
        else:
            print(
                f"Réponse JSON valide, mais pas de clé 'embedding'. Contenu: {json_response}")

    except requests.exceptions.Timeout:
        print("Erreur de timeout lors de la requête d'embedding. Ollama n'a pas répondu à temps.")
        print("Vérifiez la charge d'Ollama et l'allocation des ressources Docker.")
        raise  # Rélève l'exception pour arrêter le script
    except requests.exceptions.RequestException as req_err:
        print(
            f"Erreur de requête HTTP lors de l'appel à l'API d'embeddings: {req_err}")
        print("Ceci peut indiquer un problème réseau ou une réponse invalide.")
        raise  # Rélève l'exception pour arrêter le script
    except ValueError as json_err:
        print(f"Erreur de parsing JSON de la réponse d'Ollama: {json_err}")
        print("La réponse n'était pas un JSON valide.")
        print(f"Réponse reçue (brute): {response.text[:500]}...")
        raise  # Rélève l'exception pour arrêter le script
    print("--- FIN DU BLOC DE DIAGNOSTIC ---")
else:
    print("Aucun chunk généré pour le diagnostic.") """


# Initialisation du modèle d'embedding via le service Ollama
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)

# Utilisation de la méthode `from_documents` de PGVector
try:
    PGVector.from_documents(
        embedding=embeddings,
        documents=chunks,
        collection_name=COLLECTION_NAME,
        connection_string=CONNECTION_STRING,
        pre_delete_collection=True  # supprime la collection existante pour repartir de zéro
    )
    print("\n✅ Succès!")
    print("La base de connaissance a été créée et peuplée avec succès.")
    print(
        f"On peut maintenant interroger la collection '{COLLECTION_NAME}' dans la base de données '{DB_NAME}'.")

except Exception as e:
    print(f"\n❌ Erreur")
    print("Une erreur est survenue lors de la création des embeddings ou du stockage.")
    print(f"Détails de l'erreur: {e}")
