import streamlit as st
from utils.rag_pipeline import process_repo_and_embed, answer_query

# st.set_page_config(page_title="RAG with Git Repo", layout="centered")

# st.title("📚 GitHub Repo QA Assistant")
# st.write("Provide a GitHub repository URL and ask questions about its documentation.")

# repo_url = st.text_input("🔗 Enter GitHub Repo URL", "")
# process_button = st.button("🔄 Process Repo")

# if "repo_ready" not in st.session_state:
#     st.session_state.repo_ready = False

# if process_button and repo_url:
#     with st.spinner("Cloning and embedding repo..."):
#         try:
#             process_repo_and_embed(repo_url)
#             st.session_state.repo_ready = True
#             st.success("Repository processed and embedded successfully!")
#         except Exception as e:
#             st.error(f"Error: {e}")
#             st.session_state.repo_ready = False

# if st.session_state.repo_ready:
#     user_query = st.text_input("💬 Ask a question about the documentation")
#     if st.button("Ask"):
#         with st.spinner("Generating answer..."):
#             try:
#                 response = answer_query(user_query)
#                 st.success("Answer:")
#                 st.write(response)
#             except Exception as e:
#                 st.error(f"Error: {e}")
