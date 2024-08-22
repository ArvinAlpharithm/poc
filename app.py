import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import pickle
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from groq import Groq
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize the Groq language model with API key from environment
def initialize_groq_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("API key is missing. Please check your .env file.")
        return None
    return Groq(api_key=api_key)

# Extract text from PDF and create vector store
def get_pdf_text_from_file(file_path):
    text = ""
    with open(file_path, 'rb') as pdf_file:
        pdf_reader = PdfReader(pdf_file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Convert text to chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

# Create or load vector store from the PDF
def create_vector_store_for_pdf(pdf_path, embeddings, store_name):
    if os.path.exists(f"{store_name}.pkl"):
        with open(f"{store_name}.pkl", "rb") as f:
            return pickle.load(f)
    else:
        text = get_pdf_text_from_file(pdf_path)
        chunks = get_text_chunks(text)
        vector_store = FAISS.from_texts(chunks, embedding=embeddings)
        with open(f"{store_name}.pkl", "wb") as f:
            pickle.dump(vector_store, f)
        return vector_store

# Generate a conversational chain using Groq LLM
def get_conversational_chain(llm):
    prompt_template = """
    Answer the user's question as thoroughly as possible using the provided context. Ensure your response is structured, clear, and engaging. 
    Context:
    {context}
    Question: 
    {question}
    Answer:
    """

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    def chain(user_question, docs):
        snippets = " ".join([doc.page_content for doc in docs])
        prompt_text = prompt.format(context=snippets, question=user_question)

        try:
            result = llm.chat.completions.create(
                model="llama3-70b-8192",
                messages=[
                    {"role": "system", "content": "Provide detailed and clear responses based on the provided document snippets."},
                    {"role": "user", "content": prompt_text}
                ]
            )
            response_content = result.choices[0].message.content
        except Exception as e:
            st.write(f"Error occurred: {e}")
            response_content = "An error occurred while generating the response."

        return response_content

    return chain

# Streamlit application
def main():
    st.title("Magma Poc - Alpharithm")

    llm = initialize_groq_llm()
    if llm is None:
        return  # Exit if the API key is not available

    # Specify the directory for PDF files
    directory = "./pdfs"  # Replace with the actual path to your PDF directory

    # Create a dropdown for selecting PDFs
    pdf_files = [f for f in os.listdir(directory) if f.endswith('.pdf')]
    selected_pdf = st.selectbox("Select a PDF file:", pdf_files)

    if selected_pdf:
        pdf_path = os.path.join(directory, selected_pdf)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = create_vector_store_for_pdf(pdf_path, embeddings, f"vector_store_{selected_pdf.replace('.pdf', '')}")

        user_question = st.text_input("Ask a question about the Document:")
        if user_question:
            docs = vector_store.similarity_search(user_question, k=3)
            if docs:
                chain = get_conversational_chain(llm)
                response = chain(user_question, docs)
                st.write("Reply: ", response)
            else:
                st.write("No relevant information found in the document.")

if __name__ == "__main__":
    main()
