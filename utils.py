import uuid
import numpy as np
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.schema import Document
from PyPDF2 import PdfReader
from sklearn.metrics.pairwise import cosine_similarity
import openrouter
import openai



# Set your OpenRouter API Key
openai.api_key = "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"  # Replace with your OpenRouter API key
openai.api_base = "https://openrouter.ai/api/v1"

# Extract text from PDF files
def get_pdf_text(pdf_doc):
    """
    Extract text content from a PDF file.
    """
    text = ""
    pdf_reader = PdfReader(pdf_doc)
    for page in pdf_reader.pages:
        text += page.extract_text() if page.extract_text() else ""
    return text

# Create Document objects from uploaded PDFs
def create_docs(user_pdf_list, unique_id):
    """
    Create Document objects from uploaded PDFs and their metadata.
    """
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

# Create embeddings for the job description and resumes
def create_embeddings_load_data():
    """
    Load the SentenceTransformer embeddings for the model "all-MiniLM-L6-v2".
    """
    return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Find similar resumes based on the job description
def find_similar_resumes(query, docs, embeddings, top_k=5):
    """
    Find the top K most similar resumes to the job description using cosine similarity.
    """
    query_embedding = np.array([embeddings.embed_query(query)]).astype('float32')
    resume_embeddings = [embeddings.embed_query(doc.page_content) for doc in docs]

    # Calculate cosine similarity between the job description and each resume
    scores = cosine_similarity(query_embedding, resume_embeddings)[0]

    # Rank resumes based on similarity score
    ranked_resumes = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

    return ranked_resumes[:top_k]

# Generate a summary of the resume based on the job description
def get_summary(doc, job_description, score, current_doc):
    """
    Generate a structured summary of the resume using an OpenRouter-hosted LLM.
    This summary helps HR quickly evaluate a candidate against the job description
    """
    prompt = f"""
    You are an expert HR assistant.

    Given the following job description:

    {job_description}

    And the following candidate resume:

    {doc.page_content}

    Your task is to analyze how well the resume matches the job description. Provide a structured summary with the following format:

    ### Resume Screening Report

    **Match Score (out of 100):**  
    [Give a number based on how well the candidate matches the job description with score  ]
    the score is
   { round(score * 100, 2)}

    **Key Matching Skills:**  
    - Skill 1  
    - Skill 2  
    - Skill 3  

    **Relevant Experience:**  
    - Experience 1  
    - Experience 2  

    **Missing Skills or Gaps:**  
    - Missing Skill 1  
    - Missing Skill 2  

    **Final Verdict:**  
    Strong Fit / Partial Fit / Not a Fit

    **Justification Summary:**  
    [Brief paragraph explaining the final verdict]
    """

    response = openai.ChatCompletion.create(
        model="deepseek/deepseek-v3-base:free",
        messages=[
            {"role": "system", "content": "You are an expert HR assistant analyzing resume fit."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
        max_tokens=700,
    )

    summary = response['choices'][0]['message']['content']
    return summary

