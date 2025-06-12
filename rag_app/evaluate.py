import os
import time
import pandas as pd

# Composants LangChain nécessaires
from langchain_community.vectorstores.pgvector import PGVector
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.schema.output_parser import StrOutputParser
from langchain.retrievers.multi_query import MultiQueryRetriever


# Configuration Initiale
DB_USER = os.getenv("POSTGRES_USER", "rag_user")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "rag_password")
DB_HOST = os.getenv("POSTGRES_HOST", "postgres_db")
DB_PORT = os.getenv("POSTGRES_PORT", "5433")
DB_NAME = os.getenv("POSTGRES_DB", "rag_db")
CONNECTION_STRING = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
COLLECTION_NAME = "rag_knowledge_base"

# Configuration Ollama
EMBEDDING_MODEL = "nomic-embed-text:latest"
LLM_MODEL = "gemma3:4b"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL").strip(
) if os.getenv("OLLAMA_BASE_URL") else None
if OLLAMA_BASE_URL is None:
    print("Erreur: OLLAMA_BASE_URL n'est pas défini. Veuillez vérifier votre docker-compose.yml")
    exit(1)

# Définition des questions de test
test_questions = [
    {
        "type": "Dans la base de connaissances",
        "question": "Advise me some podcasts about love and relationships."
    },
    # la question dans la base de connaissances : What are the five love and relationship podcasts mentioned in the context?
    {
        "type": "Dans la base de connaissances",
        "question": "Why must we recycle concrete in the construction industry?"
    },
    # la question : What is the purpose of recycling concrete in construction?
    {
        "type": "Dans la base de connaissances",
        "question": "How many followers has the most followed person on Instagram?"
    },
    # la question : Who is the most followed person on Instagram according to the context?

    {
        "type": "Sujet proche mais hors base",
        "question": "What are the fashion trends for next fall?"
    },
    # la question : What are some trendy short hairstyles for fall?

    {
        "type": "Totalement hors base (Connaissance générale)",
        "question": "Who was the first First Lady of the United States?"
    },
    {
        "type": "Totalement hors base (Créatif)",
        "question": "What is the meaning of life?"
    }
]

# Fonction de débogage pour afficher le contexte


def log_retrieved_context(input_dict):
    print("\n[DEBUG] Contexte récupéré pour la question")
    if not input_dict.get('context'):
        print(">>> Le retriever n'a retourné aucun document.")
    else:
        for i, doc in enumerate(input_dict['context']):
            print(f"  [Doc {i+1}]: {doc.page_content[:150]}...")
    print("[DEBUG] Fin du contexte\n")
    return input_dict


def setup_chains():

    # Initialisation des composants communs
    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
    llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)

    # RAG chain
    vector_store = PGVector(
        connection_string=CONNECTION_STRING,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )

    # simple retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    # multi-query pour évaluer la performance avec le retriever simple :
    # base_retriever = vector_store.as_retriever()
    # multi_query_retriever = MultiQueryRetriever.from_llm(
    #    retriever=base_retriever, llm=llm)

    rag_prompt_template = """
SYSTEM: Vous êtes un assistant expert. Utilisez UNIQUEMENT le contexte suivant pour répondre à la question. 
Si vous ne connaissez pas la réponse, dites simplement que vous ne savez pas. Ne tentez pas d'inventer une réponse.
CONTEXTE: {context}
QUESTION: {question}
RÉPONSE:"""
    rag_prompt = PromptTemplate.from_template(rag_prompt_template)
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | RunnableLambda(log_retrieved_context)
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    # Non-RAG chain (avec LLM seul)
    non_rag_prompt_template = """
SYSTEM: Vous êtes un assistant expert. Répondez à la question suivante en utilisant uniquement vos connaissances générales.
QUESTION: {question}
RÉPONSE:"""
    non_rag_prompt = PromptTemplate.from_template(non_rag_prompt_template)
    non_rag_chain = (
        {"question": RunnablePassthrough()}
        | non_rag_prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, non_rag_chain


def run_evaluation():
    print("Lancement de l'évaluation...")
    rag_chain, non_rag_chain = setup_chains()
    results = []

    for i, item in enumerate(test_questions):
        question = item["question"]
        print(
            f"\nProcessing question {i+1}/{len(test_questions)}: \"{question}\"")

        # Obtenir la réponse avec RAG et mesurer le temps
        start_time_rag = time.time()
        rag_response = rag_chain.invoke(question)
        end_time_rag = time.time()
        rag_duration = end_time_rag - start_time_rag

        # Obtenir la réponse sans RAG et mesurer le temps
        start_time_non_rag = time.time()
        non_rag_response = non_rag_chain.invoke(question)
        end_time_non_rag = time.time()
        non_rag_duration = end_time_non_rag - start_time_non_rag

        results.append({
            "Type de Question": item["type"],
            "Question": question,
            "Réponse AVEC RAG": rag_response,
            "Temps RAG (s)": f"{rag_duration:.2f}",
            "Réponse SANS RAG (LLM Seul)": non_rag_response,
            "Temps LLM Seul (s)": f"{non_rag_duration:.2f}"
        })

    print("\nÉvaluation terminée")
    return results


def main():
    try:
        evaluation_results = run_evaluation()

        # Conversion des résultats en DataFrame pandas pour un affichage propre
        df = pd.DataFrame(evaluation_results)

        # Création du tableau au format Markdown
        markdown_table = df.to_markdown(index=False)

        # Affichage du tableau dans la console
        print("\n\nTableau Comparatif des Résultats\n")
        print(markdown_table)

        # Sauvegarde du tableau dans un fichier
        report_filename = "evaluation_report.md"
        with open(report_filename, "w", encoding="utf-8") as f:
            f.write("# Rapport d'Évaluation du Système RAG\n\n")
            f.write(markdown_table)

        print(f"\n✅ Rapport sauvegardé sous le nom : {report_filename}")

    except Exception as e:
        print(f"\n❌ Erreur durant l'évaluation")
        print(f"Détails: {e}")


if __name__ == "__main__":
    main()
