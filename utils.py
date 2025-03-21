import os
import uuid
import numpy as np
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.schema import Document
from transformers import pipeline
from PyPDF2 import PdfReader
from langchain.chains.summarize import load_summarize_chain
from langchain import HuggingFaceHub
from sklearn.metrics.pairwise import cosine_similarity


def get_pdf_text(pdf_doc):
    text = ""
    pdf_reader = PdfReader(pdf_doc)
    for page in pdf_reader.pages:
        text += page.extract_text() if page.extract_text() else ""
    return text


def create_docs(user_pdf_list, unique_id):
    docs = []
    for filename in user_pdf_list:
        chunks = get_pdf_text(filename)
        docs.append(Document(
            page_content=chunks,
            metadata={
                "name": filename.name,
                "id": uuid.uuid4().hex,
                "type": filename.type,
                "size": filename.size,
                "unique_id": unique_id,
            }
        ))
    return docs


def create_embeddings_load_data():
    return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


def find_similar_resumes(query, docs, embeddings, top_k=5):
    query_embedding = np.array([embeddings.embed_query(query)]).astype('float32')
    resume_embeddings = [embeddings.embed_query(doc.page_content) for doc in docs]

 
    scores = cosine_similarity(query_embedding, resume_embeddings)[0]

 
    ranked_resumes = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

    return ranked_resumes[:top_k] 

###def get_summary(current_doc):
    ##llm = HuggingFaceHub(repo_id='mistral-7b-v0.1.Q5_K_M.gguf', model_kwargs={'temperature': 1e-10},huggingfacehub_api_token="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")
    ##chain = load_summarize_chain(llm, chain_type="map_reduce")
    ##summary = chain.run([current_doc])
    ##return summary###
