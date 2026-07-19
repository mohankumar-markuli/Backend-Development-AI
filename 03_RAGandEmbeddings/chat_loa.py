import streamlit as st
import os
import numpy as np
import pandas as pd
import faiss
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import LatexTextSplitter
from sentence_transformers import SentenceTransformer
import cohere
# pip install cohere
# pip install faiss-cpu
# pip install sentence-transformers
# pip install streamlit
# pip install -qU langchain-community pypdf
# streamlit run chat_loa.py


# pyrefly: ignore [missing-import]
from cohere import ClientV2

# Initialize Sentence Transformer model
model_name = "all-MiniLM-L6-v2"  # Adjust model as needed
sentence_transformer_model = SentenceTransformer(model_name)

# Load and process the PDF file



pdf_path = os.path.join(os.path.dirname(__file__), 'Resume.pdf')
loader = PyPDFLoader(pdf_path)
documents = loader.load() # filled with pdfs

embeddings = []
documents_text = []
sources = []

#To chunkify the docs 
latex_splitter = LatexTextSplitter(chunk_size=500, chunk_overlap=100)


for docu in documents: # one page at a time from the pdf
    docs = latex_splitter.create_documents([docu.page_content]) #splitting the the docs into chunks
    for document in docs:
        document_embedding = sentence_transformer_model.encode(document.page_content)
        embeddings.append(document_embedding)
        documents_text.append(document.page_content)
        sources.append("www.rheadata.com")

# Create FAISS index
embedding_dimension = len(embeddings[0])
index = faiss.IndexFlatL2(embedding_dimension)
index.add(np.array(embeddings, dtype='float32'))

# Save index and document details
if not os.path.exists("Vector_Store"):
    os.makedirs("Vector_Store")
df = pd.DataFrame({'documents': documents_text, 'source': sources})
df.to_csv('Vector_Store/docs.csv', index=False)
faiss.write_index(index, 'Vector_Store/vector_db.index')
# ingestion ends above this


# Streamlit app layout
st.title("HR Q&A")

query = st.text_input("Enter your query:")

if query:
    query_embedding = sentence_transformer_model.encode(query).reshape(1, -1)
    distances, indices = index.search(query_embedding, k=2)
    
    threshold = 70# Define your threshold for distance match
    print(distances[0][0])
    if distances[0][0] > threshold:
        st.write("Please ask a relevant question.")
    else:
        
        combined_similar_documents_content = []
        similar_documents_sources = []
        for i in indices[0]:
            similar_document_content = df.loc[i, 'documents']
            combined_similar_documents_content.append(similar_document_content)
            print(i)
            similar_document_source = df.loc[i, 'source']
            similar_documents_sources.append(similar_document_source)
        
        combined_similar_documents_content = ' '.join(combined_similar_documents_content)

        print(combined_similar_documents_content)
        print(list(set(similar_documents_sources)))
        cohere_prompt = f"Based on the document page {combined_similar_documents_content}, answer the question: '{query}'"

                # Call Cohere API for response generation
        co = ClientV2(api_key="i8NVKCovHE1xde0DBIWtp8mDoiIVd72HkePhDhIz")
        
        cohere_response = co.chat(
            model="command-a-03-2025",  # latest stable chat model
            messages=[{"role": "user", "content": cohere_prompt}],
            temperature=0.3
        )
        
        st.write("Bot Response:")
        st.write(cohere_response.message.content[0].text)
        st.write("Sources:")
        st.write(list(set(similar_documents_sources)))

