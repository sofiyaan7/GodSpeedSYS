import os
import git
import hashlib
from textwrap import wrap
from markdown import markdown
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import chromadb
import requests

embed_model = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="gs_docs")

OPENROUTER_API_KEY = "sk-..."  # Replace securely for deployment

def clone_repo(git_url, branch="main"):
    repo_name = git_url.split('/')[-1].replace(".git", "")
    if os.path.isdir(repo_name):
        repo = git.Repo(repo_name)
        repo.remotes.origin.pull()
    else:
        git.Repo.clone_from(git_url, repo_name, branch=branch)
    return repo_name

def md_to_text(md_content):
    html = markdown(md_content)
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text()

def process_repo_and_embed(repo_url):
    repo_name = clone_repo(repo_url)
    md_files = []
    for root, dirs, files in os.walk(repo_name):
        for file in files:
            if file.endswith(".md"):
                md_files.append(os.path.join(root, file))

    chunks = []
    for file in md_files:
        with open(file, "r", encoding="utf-8") as f:
            text = md_to_text(f.read())
            content_chunks = wrap(text, 500)
            for i, chunk in enumerate(content_chunks):
                chunks.append({
                    "id": f'{file}_chunk_{i}',
                    "text": chunk
                })

    existing_ids = set(collection.get()['ids'])

    for chunk in chunks:
        doc_id = chunk["id"]
        if doc_id not in existing_ids:
            embedding = embed_model.encode([chunk["text"]])[0]
            collection.add(documents=[chunk["text"]], ids=[doc_id], embeddings=[embedding])

def answer_query(query):
    embedding = embed_model.encode([query])[0]
    results = collection.query(query_embeddings=[embedding], n_results=5)
    context = "\n\n".join(results["documents"][0])

    prompt = f"""Answer the question based on the context below.

Context:
{context}

Question:
{query}
"""

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "http://localhost",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "mistralai/mistral-7b-instruct",
        "messages": [{"role": "user", "content": prompt}]
    }

    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        raise Exception(f"API Error: {response.text}")
