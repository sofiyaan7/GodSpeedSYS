import streamlit as st
from rag_engine import clone_or_update_repo, get_changed_md_files, chunk_documents, update_chroma
from query_engine import multimodal_query_openrouter
import os

st.set_page_config(page_title="Git RAG Chat", layout="wide")

st.title("📚 GitHub Repo RAG Chatbot")

git_url = st.text_input("Enter GitHub Repository URL:")
branch = st.text_input("Enter branch name:", value="main")
api_key = st.text_input("Enter your OpenRouter API Key:", type="password")

if st.button("🛠️ Process Repo"):
    if git_url:
        doc_path, updated = clone_or_update_repo(git_url, branch)
        if updated:
            st.info("Repository updated. Processing files...")
            md_files = get_changed_md_files(doc_path)
            chunks = chunk_documents(md_files)
            update_chroma(chunks)
            st.success("Embedding complete and added to ChromaDB.")
        else:
            st.warning("No new updates in the repo.")
    else:
        st.error("Please enter a valid GitHub URL.")

st.markdown("---")
st.subheader("💬 Ask a Question")
query = st.text_input("Enter your question here:")
image = st.file_uploader("Optional: Upload an image", type=["png", "jpg", "jpeg"])

if st.button("🔍 Query"):
    if not api_key:
        st.error("Please provide your OpenRouter API Key.")
    elif not query:
        st.warning("Enter a query to proceed.")
    else:
        image_path = None
        if image:
            image_path = os.path.join("temp", image.name)
            os.makedirs("temp", exist_ok=True)
            with open(image_path, "wb") as f:
                f.write(image.read())
        with st.spinner("Querying..."):
            response = multimodal_query_openrouter(query, image_path, api_key)
            st.markdown("### 📥 Response")
            st.write(response)
