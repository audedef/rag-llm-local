# Ce script permet d'interroger le RAG.
# Il récupère des infos pertinentes depuis la db vectorielle pour contextualiser la ? de l'utilisateur avant de la soumettre au LLM.

import os
from dotenv import load_dotenv

# Composants LangChain pour construire la chaîne RAG
from langchain_community.vectorstores.pgvector import PGVector
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# 1. Configuration Initiale

# Charge les variables d'environnement depuis un fichier .env
load_dotenv()

# Configuration de la connexion à la base de données PostgreSQL
DB_USER = os.getenv("POSTGRES_USER", "rag_user")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "rag_password")
DB_HOST = os.getenv("POSTGRES_HOST", "postgres_db")
DB_PORT = os.getenv("POSTGRES_PORT", "5433")
DB_NAME = os.getenv("POSTGRES_DB", "rag_db")

# Construction de l'URL de connexion
CONNECTION_STRING = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Nom de la collection (table) dans laquelle chercher les informations
COLLECTION_NAME = "rag_knowledge_base"

# Modèle d'embedding et modèle de chat à utiliser avec Ollama
EMBEDDING_MODEL = "nomic-embed-text:latest"
LLM_MODEL = "gemma3:4b"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL").strip(
) if os.getenv("OLLAMA_BASE_URL") else None
if OLLAMA_BASE_URL is None:
    print("Erreur: OLLAMA_BASE_URL n'est pas défini. Veuillez vérifier votre docker-compose.yml")
    exit(1)

print("--- Configuration du client RAG chargée ---")
print(f"Modèle LLM: {LLM_MODEL}")
print(f"Modèle d'embedding: {EMBEDDING_MODEL}")
print(f"Collection PGVector: {COLLECTION_NAME}")


def main():
    try:
        # 2. Initialisation des composants LangChain
        print("\nInitialisation des composants LangChain")

        # Initialisation du modèle d'embedding (doit être le même que pour l'ingestion)
        embeddings = OllamaEmbeddings(
            model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)

        # Connexion à la base de données vectorielle existante
        vector_store = PGVector(
            connection_string=CONNECTION_STRING,
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
        )

        # Création d'un "retriever" à partir du vector store.
        # pour récupérer les documents pertinents.
        # k=3 signifie qu'on récupérera les 3 chunks les plus pertinents.
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        print("Retriever configuré pour chercher les 3 documents les plus pertinents.")

        # Initialisation du modèle de langage (LLM) via Ollama
        llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL)

        # 3. Définition du Prompt Template

        prompt_template = """
        SYSTEM: Vous êtes un assistant expert et concis. Utilisez UNIQUEMENT les informations 
        suivantes pour répondre à la question. Si vous ne connaissez pas la réponse, 
        dites simplement que vous ne savez pas. Ne tentez pas d'inventer une réponse.

        CONTEXTE:
        {context}

        QUESTION:
        {question}

        RÉPONSE:
        """
        prompt = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        # 4. Création de la RAG chain
        # C'est ici qu'on orchestre le pipeline avec LangChain Expression Language (LCEL).
        # Le pipeline se lit comme suit :
        # 1. On prend la question de l'utilisateur.
        # 2. Le retriever cherche le contexte pertinent.
        # 3. On passe la question et le contexte récupéré au prompt.
        # 4. Le prompt formaté est envoyé au LLM.
        # 5. La sortie du LLM est parsée pour obtenir une chaîne de caractères.
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        print("\n✅ Le système RAG est prêt. Posez vos questions.")
        print("Tapez 'exit' ou 'quit' pour quitter.")

        # 5. Boucle d'Interaction
        while True:
            question = input("\nVotre question: ")
            if question.lower() in ["exit", "quit"]:
                break

            print("\nRéponse de l'assistant:")
            # On invoque la chaîne avec la question et on stream la réponse
            for chunk in rag_chain.stream(question):
                print(chunk, end="", flush=True)
            print("\n" + "="*50)

    except Exception as e:
        print(f"\n--- ❌ Erreur critique ---")
        print(f"Détails de l'erreur: {e}")
        print("\nVérifications possibles:")
        print("1. Avez-vous bien lancé le script d'ingestion `ingest.py` au préalable ?")
        print("2. Les services Docker (Ollama, PostgreSQL) sont-ils en cours d'exécution ? (`docker-compose ps`)")
        print(
            "3. Les noms de collection et les configurations de la DB sont-ils corrects ?")


if __name__ == "__main__":
    main()
