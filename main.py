import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import HTMLHeaderTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_classic.chains.question_answering import load_qa_chain
from langchain_classic.prompts import PromptTemplate
from langchain_classic.chains import conversational_retrieval
from langchain_classic.chains import retrieval_qa

# Load environment and configure Gemini
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# Extract text from uploaded PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text


# Split text into smaller chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000
    )
    chunks = text_splitter.split_text(text)
    return chunks


# Create and persist Chroma vector store
def get_vector_store(text_chunks):
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create Chroma vector store and persist it locally
    vector_store = Chroma.from_texts(
        texts=text_chunks,
        embedding=embeddings,
        persist_directory="chroma_db"
    )

    vector_store.persist()
    return vector_store


# Build a conversational QA chain (Gemini)
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context.
    Make sure to provide all the details.
    If the answer is not in the provided context, just say,
    "answer is not available in the context" — don't make up an answer.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# def get_conversational_chain(retriever):
#     llm = ChatGoogleGenerativeAI(model="gemini-flash-2.5", temperature=0.3)

#     # New ChatPromptTemplate (multi-turn chat style)
#     prompt = ChatPromptTemplate.from_messages([
#         ("system",
#          "You are a helpful AI assistant. Use the context provided to answer accurately. "
#          "If the answer is not in the context, say 'answer is not available in the context'."),
#         ("human", "Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:")
#     ])

#     chain = retrieval_qa.from_llm(
#         llm=llm,
#         retriever=retriever,
#         return_source_documents=True,
#         combine_docs_chain_kwargs={"prompt": prompt}
#     )

#     return chain

# Process user query
def user_input(user_question):
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # Load existing Chroma vector DB
    try:
        new_db = Chroma(
            persist_directory="chroma_db",
            embedding_function=embeddings
        )
    except Exception as e:
        st.error(f"Failed to load Chroma database: {e}")
        return None

    # Retrieve top-k relevant chunks
    try:
        docs = new_db.similarity_search(user_question, k=5)
    except Exception as e:
        st.error(f"Search failed: {e}")
        return None

    # Build QA chain
    chain = get_conversational_chain()

    # Run chain and handle output
    try:
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    except Exception as e:
        st.error(f"Chain execution failed: {e}")
        return None

    # Extract and display result
    if isinstance(response, dict):
        text = response.get("output_text") or response.get("answer") or str(response)
    else:
        text = str(response)

    st.write("Reply:", text)
    return text


# Streamlit UI
def main():
    st.set_page_config("Chat PDF")
    st.header("💬 Chat with your PDFs using Gemini + ChromaDB")

    user_question = st.text_input("Ask a question about your uploaded PDFs:")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("📂 Menu")
        pdf_docs = st.file_uploader(
            "Upload your PDF files and click 'Submit & Process'",
            accept_multiple_files=True
        )
        if st.button("Submit & Process"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF.")
                return

            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("✅ Done! You can now ask questions.")


# Entry point
if __name__ == "__main__":
    main()
