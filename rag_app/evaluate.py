# Ce script a pour but d'évaluer l'efficacité de notre RAG.
# Il compare les réponses d'un LLM de base avec celles du même LLM
# enrichi par le contexte récupéré dans la base de connaissances ingérée.
# Le résultat est un tableau comparatif au format Markdown.

import os
import time
import pandas as pd
from dotenv import load_dotenv

# Composants LangChain nécessaires
from langchain_community.vectorstores.pgvector import PGVector
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.schema.output_parser import StrOutputParser

# 1. Configuration Initiale
load_dotenv()

# Configuration de la connexion à la base de données
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

# 2. Définition des Questions de Test
test_questions = [
    {
        "type": "Dans la base de connaissances",
        "question": "What is the recipe for Rieska ?"
    },
    {
        "type": "Dans la base de connaissances",
        "question": "What is the Toronto Alternative Art Fair International ?"
    },
    {
        "type": "Dans la base de connaissances",
        "question": "Why was it the chaos in transportation in France, Germany and Belgium ?"
    },
    {
        # question contraire (qui n'est PAS éligible)
        "type": "Sujet proche mais hors base",
        "question": "Who is eligible to enter the competition ?"
    },
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
    """
    Affiche le contexte récupéré par le retriever -> étape de débogage
    """
    print("\n[DEBUG] Contexte récupéré pour la question")
    if not input_dict.get('context'):
        print(">>> Le retriever n'a retourné aucun document.")
    else:
        for i, doc in enumerate(input_dict['context']):
            print(f"  [Doc {i+1}]: {doc.page_content[:150]}...")
    print("--- [DEBUG] Fin du contexte ---\n")
    return input_dict


def setup_chains():
    """Initialise et retourne les deux chaînes à comparer : RAG et non-RAG."""
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
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={'k': 5, 'fetch_k': 20}
    )

    rag_prompt_template = """
SYSTEM: Vous êtes un assistant expert. Utilisez UNIQUEMENT le contexte suivant pour répondre à la question. Si vous ne connaissez pas la réponse, dites "Je ne sais pas en me basant sur le contexte fourni."
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
