import os
import git
import json
import hashlib
from markdown import markdown
from bs4 import BeautifulSoup
from textwrap import wrap
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient

EMBED_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
CHROMA_PATH = "./chroma_db"
CHUNK_SIZE = 500

def clone_or_update_repo(git_url, branch="main", meta_file="repo_meta.json"):
    doc_path = git_url.split('/')[-1].replace(".git", "")
    if os.path.isdir(doc_path):
        repo = git.Repo(doc_path)
        origin = repo.remotes.origin
        origin.fetch()
        latest_commit = origin.refs[branch].commit.hexsha
        if os.path.exists(meta_file):
            with open(meta_file, "r") as f:
                meta = json.load(f)
        else:
            meta = {}
        last_commit = meta.get(doc_path)
        if last_commit != latest_commit:
            origin.pull()
            meta[doc_path] = latest_commit
            with open(meta_file, "w") as f:
                json.dump(meta, f)
            return doc_path, True
        return doc_path, False
    else:
        repo = git.Repo.clone_from(git_url, doc_path, branch=branch)
        latest_commit = repo.head.commit.hexsha
        with open(meta_file, "w") as f:
            json.dump({doc_path: latest_commit}, f)
        return doc_path, True

def md_to_text(md_content):
    html = markdown(md_content)
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text()

def get_changed_md_files(doc_path, hash_file="file_hashes.json"):
    if os.path.exists(hash_file):
        with open(hash_file, "r") as f:
            old_hashes = json.load(f)
    else:
        old_hashes = {}
    new_hashes = {}
    updated_md_files = []
    for root, _, files in os.walk(doc_path):
        for file in files:
            if file.endswith(".md"):
                file_path = os.path.join(root, file)
                sha256 = hashlib.sha256()
                with open(file_path, "rb") as f:
                    while chunk := f.read(8192):
                        sha256.update(chunk)
                file_hash = sha256.hexdigest()
                new_hashes[file_path] = file_hash
                if old_hashes.get(file_path) != file_hash:
                    updated_md_files.append(file_path)
    with open(hash_file, "w") as f:
        json.dump(new_hashes, f)
    return updated_md_files

def chunk_documents(md_files):
    docs = []
    for file in md_files:
        with open(file, "r", encoding="utf-8") as f:
            md = f.read()
            text = md_to_text(md)
            docs.append({"file_path": file, "content": text})
    chunks = []
    for doc in docs:
        content_chunks = wrap(doc["content"], CHUNK_SIZE)
        for i, chunk in enumerate(content_chunks):
            chunks.append({
                "id": f'{doc["file_path"]}_chunk_{i}',
                "text": chunk
            })
    return chunks

def update_chroma(chunks):
    client = PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(name="gs_docs")
    existing_ids = set(collection.get()['ids'])
    for chunk in chunks:
        if chunk["id"] not in existing_ids:
            embedding = EMBED_MODEL.encode([chunk["text"]])[0]
            collection.add(
                documents=[chunk["text"]],
                ids=[chunk["id"]],
                embeddings=[embedding]
            )
