# Ce script a pour objectif de charger des documents, de les traiter
# et de les stocker dans une base de données vectorielle PostgreSQL (avec pgvector).
# Ce processus constituera la base de connaissance pour notre système RAG.

import os
from dotenv import load_dotenv

# Chargeurs de documents et splitters de texte de LangChain
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
# Charge les variables d'environnement depuis un fichier .env
load_dotenv()

# Configuration de la connexion à la base de données PostgreSQL
# Les valeurs sont récupérées depuis les variables d'environnement,
# qui doivent correspondre à celles définies dans votre docker-compose.yml.
DB_USER = os.getenv("POSTGRES_USER", "rag_user")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "strong_password")
# Utiliser 'postgres_db_rag' si le script tourne dans un conteneur Docker du même réseau
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = os.getenv("POSTGRES_PORT", "5433")
DB_NAME = os.getenv("POSTGRES_DB", "rag_database")

# Construction de l'URL de connexion
# Format: postgresql+psycopg2://user:password@host:port/dbname
CONNECTION_STRING = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Nom de la collection (table) dans la base de données vectorielle
COLLECTION_NAME = "rag_knowledge_base"

# Nom du dataset sur Hugging Face et la colonne contenant le texte
DATASET_NAME = "neural-bridge/rag-dataset-12000"
PAGE_CONTENT_COLUMN = "text"  # La colonne qui contient le texte à indexer

# Modèle d'embedding à utiliser avec Ollama
EMBEDDING_MODEL = "gemma"
# URL du service Ollama (doit correspondre au service dans docker-compose)
OLLAMA_BASE_URL = "http://localhost:11434"

print("------ Configuration chargée ------")
print(f"Dataset: {DATASET_NAME}")
print(f"Modèle d'embedding: {EMBEDDING_MODEL}")
print(f"Collection PGVector: {COLLECTION_NAME}")

# 2. Chargement des Données
print("\nÉtape 1: Chargement des documents depuis Hugging Face")

# Utilise le chargeur de LangChain pour charger un jeu de données depuis le Hub.
# On ne charge qu'un échantillon (100 premières lignes) pour l'exemple.
# Pour un cas réel, vous pourriez vouloir charger l'ensemble des données.
loader = HuggingFaceDatasetLoader(
    DATASET_NAME, PAGE_CONTENT_COLUMN, streaming=True)
# Le streaming est utilisé pour ne pas tout charger en mémoire d'un coup.
# On prend un nombre limité de documents pour la démo.
documents = list(loader.load())[:100]

print(f"{len(documents)} documents ont été chargés.")

# 3. Découpage des Documents (Chunking)
print("\nÉtape 2: Découpage des documents en segments (chunks)")

# Le découpage est essentiel pour que la recherche sémantique soit efficace.
# On divise les documents en plus petits morceaux.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Taille maximale d'un chunk (en caractères)
    chunk_overlap=100  # Nombre de caractères de chevauchement entre les chunks
)
chunks = text_splitter.split_documents(documents)

print(f"Les documents ont été découpés en {len(chunks)} chunks.")

# 4. Création des Embeddings et Stockage Vectoriel
print("\nÉtape 3: Génération des embeddings et stockage dans PGVector")
# Cette étape peut prendre plusieurs minutes, en fonction du volume de données et de la puissance de votre machine..."

# Initialisation du modèle d'embedding via le service Ollama
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)

# Utilisation de la méthode `from_documents` de PGVector.
# C'est une méthode très pratique qui gère pour vous :
# 1. La connexion à la base de données.
# 2. La création (si nécessaire) de la table pour la collection.
# 3. La génération des embeddings pour chaque chunk en appelant le modèle Ollama.
# 4. L'insertion des chunks et de leurs embeddings dans la base.
try:
    PGVector.from_documents(
        embedding=embeddings,
        documents=chunks,
        collection_name=COLLECTION_NAME,
        connection_string=CONNECTION_STRING,
        # Optionnel: supprime la collection existante pour repartir de zéro
        pre_delete_collection=True
    )
    print("\n✅ Succès!")
    print("La base de connaissance a été créée et peuplée avec succès.")
    print(
        f"Vous pouvez maintenant interroger la collection '{COLLECTION_NAME}' dans votre base de données '{DB_NAME}'.")

except Exception as e:
    print(f"\n--- ❌ Erreur ---")
    print("Une erreur est survenue lors de la création des embeddings ou du stockage.")
    print(f"Détails de l'erreur: {e}")
    print("\nVérifications possibles:")
    print("1. Le service Ollama est-il bien démarré ? (`docker-compose ps`)")
    print("2. Le service PostgreSQL est-il bien démarré et accessible sur le port 5433 ?")
    print("3. Le modèle 'gemma' a-t-il été téléchargé sur Ollama ? (Sinon, lancez: `docker exec -it ollama_server ollama run gemma`)")
