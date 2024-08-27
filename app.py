import streamlit as st
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import pickle
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain_ibm import WatsonxLLM
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize the Watsonx language model with API key from environment
@st.cache_resource
def initialize_watsonx_llm():
    api_key = os.getenv("WATSONX_API_KEY")
    if not api_key:
        st.error("API key is missing. Please check your .env file.")
        return None

    parameters = {
        "decoding_method": "greedy",
        "max_new_tokens": 1000,
        "min_new_tokens": 1
    }
    return WatsonxLLM(
        model_id="mistralai/mixtral-8x7b-instruct-v01",
        apikey=api_key,
        url="https://us-south.ml.cloud.ibm.com",
        project_id="947b56d0-1ff3-4881-9f41-d2e6eb004377",  # Replace with your actual project ID
        params=parameters,
    )

# Extract text from PDF using pdfplumber
def get_pdf_text_from_file(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text()
            if page_text:
                text += page_text
            else:
                st.warning(f"Warning: Page {page_num} might not have been processed correctly.")
    return text

# Convert text to chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
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

# Generate a conversational chain using Watsonx LLM
def get_conversational_chain(llm):
    prompt_template = """
    You are an AI assistant. Answer the user's question using only the information provided in the context. 
    Do not speculate or add information not present in the context. 
    If the question cannot be answered using the given context, explicitly state that the information is not found in the document.

    Context:
    {context}

    Question: 
    {question}

    Answer:
    """

    def chain(user_question, docs):
        if not docs:
            return "No relevant information found in the document to answer this question."

        snippets = " ".join([doc.page_content for doc in docs])
        
        # Check if the snippets are relevant to the question
        if not any(word.lower() in snippets.lower() for word in user_question.split()):
            return "The document does not contain information relevant to this question."

        prompt_text = prompt_template.format(context=snippets, question=user_question)

        try:
            # Using Watsonx LLM for generating the response
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            result = chain.invoke({"input_documents": docs, "question": user_question})
            response_content = result["output_text"]

            # Additional check to ensure the response is not speculative
            if "not mentioned in the document" in response_content.lower() or "no information" in response_content.lower():
                return "The document does not contain information to answer this question."

            # Guard rail: Ensure that the response is strictly based on the document
            if "I don't know" in response_content.lower() or "not found" in response_content.lower():
                return "The document does not contain information to answer this question."

            return response_content
        except Exception as e:
            st.write(f"Error occurred: {e}")
            return "An error occurred while generating the response."

    return chain

# Streamlit application
def main():
    st.title("Magma Poc - Alpharithm")

    llm = initialize_watsonx_llm()
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

        # Add buttons for selecting the format of the answer
        bullet_button = st.radio("Select the format of the answer:", ("In Bullet Points", "In Table"))
        
        user_question = st.text_input("Ask a question about the Document:")
        
        if user_question:
            # Modify the user question based on the selected format
            if bullet_button == "In Bullet Points":
                user_question += ", give the answers in the form of bullet points. give only one sentence in one line. give space between one bullet point and another.answer only from the document, do not answer on your own"
            elif bullet_button == "In Table":
                user_question += ", give the answers in the form of bullet points. give only one sentence in one line. give space between one bullet point and another.answer only from the document, do not answer on your own present the answers in the form of a table."

            docs = vector_store.similarity_search(user_question, k=3)
            if docs:
                chain = get_conversational_chain(llm)
                response = chain(user_question, docs)
                st.write("Reply: ", response)
            else:
                st.write("No relevant information found in the document.")

if __name__ == "__main__":
    main()