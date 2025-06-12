import os

# Composants LangChain pour construire le RAG
from langchain_community.vectorstores.pgvector import PGVector
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# 1. Configuration Initiale

# Configuration de la connexion à la base de données PostgreSQL
DB_USER = os.getenv("POSTGRES_USER", "rag_user")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "rag_password")
DB_HOST = os.getenv("POSTGRES_HOST", "postgres_db")
DB_PORT = os.getenv("POSTGRES_PORT", "5433")
DB_NAME = os.getenv("POSTGRES_DB", "rag_db")

CONNECTION_STRING = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

COLLECTION_NAME = "rag_knowledge_base"

# Modèle d'embedding et LLM à utiliser avec Ollama
EMBEDDING_MODEL = "nomic-embed-text:latest"
LLM_MODEL = "gemma3:4b"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL").strip(
) if os.getenv("OLLAMA_BASE_URL") else None
if OLLAMA_BASE_URL is None:
    print("Erreur: OLLAMA_BASE_URL n'est pas défini. Veuillez vérifier votre docker-compose.yml")
    exit(1)

print("Configuration du RAG chargée :")
print(f"Modèle LLM: {LLM_MODEL}")
print(f"Modèle d'embedding: {EMBEDDING_MODEL}")
print(f"Collection PGVector: {COLLECTION_NAME}")


def main():
    try:
        # 2. Initialisation des composants LangChain
        print("\nInitialisation des composants LangChain")

        # Initialisation du modèle d'embedding (il est le même que pour l'ingestion)
        embeddings = OllamaEmbeddings(
            model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)

        # Connexion à la base de données vectorielle existante
        vector_store = PGVector(
            connection_string=CONNECTION_STRING,
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
        )

        # Création du retriever
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
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
            for chunk in rag_chain.stream(question):
                print(chunk, end="", flush=True)
            print("\n" + "="*50)

    except Exception as e:
        print(f"\n❌ Erreur critique")
        print(f"Détails de l'erreur: {e}")
        print("\nVérifications possibles:")
        print("1. Avez-vous bien lancé le script d'ingestion `ingest.py` au préalable ?")
        print("2. Les services Docker (Ollama, PostgreSQL) sont-ils en cours d'exécution ?")


if __name__ == "__main__":
    main()
