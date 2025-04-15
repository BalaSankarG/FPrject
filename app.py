import streamlit as st
from dotenv import load_dotenv
from utils import *
import uuid

if 'unique_id' not in st.session_state:
    st.session_state['unique_id'] = ''

def main():
    load_dotenv()
    st.set_page_config(page_title="Resume Screening Assistance")
    st.markdown('<style>p {font-size: 20px;}</style>', unsafe_allow_html=True)
    st.title("AI Resume Screening  ")
    st.subheader("")

    job_description = st.text_area("'JOB DESCRIPTION'", key="1")
    document_count = st.text_input("No. of 'RESUMES' to return", key="2")

    pdf = st.file_uploader("Upload resumes here, only PDF files allowed", type=["pdf"], accept_multiple_files=True)

    submit = st.button("Analysis")

    if submit:
        with st.spinner('Analyzing resumes...'):

            st.session_state['unique_id'] = uuid.uuid4().hex

            final_docs_list = create_docs(pdf, st.session_state['unique_id'])
            st.write(f"**Resumes uploaded:** {len(final_docs_list)}")

            embeddings = create_embeddings_load_data()

            relevant_docs = find_similar_resumes(job_description, final_docs_list, embeddings, int(document_count))

            st.write(":heavy_minus_sign:" * 30)

            for i, (doc, score) in enumerate(relevant_docs):
                st.subheader(f"👉 Resume {i + 1}")
                st.write(f"**File:** {doc.metadata['name']}")
                st.info(f"**Match Score:** {round(score * 100, 2)}%")  

                with st.expander('Show More 👀'):
                    summary = get_summary(doc, job_description, score, current_doc=None)
                    st.write(f"**Summary:** {summary}")

# Run the app
if __name__ == '__main__':
    main()
