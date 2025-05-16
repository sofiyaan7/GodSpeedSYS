import requests
from PIL import Image
# from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import CLIPProcessor, CLIPModel
import torch
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient

embed_model = SentenceTransformer('all-MiniLM-L6-v2')
client = PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="gs_docs")

# blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
# blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("cuda" if torch.cuda.is_available() else "cpu")

# def generate_image_caption(image_path):
#     image = Image.open(image_path).convert('RGB')
#     inputs = blip_processor(image, return_tensors="pt").to(blip_model.device)
#     caption_ids = blip_model.generate(**inputs)
#     return blip_processor.decode(caption_ids[0], skip_special_tokens=True)

def multimodal_query_openrouter(query, image_path=None, api_key="", top_k=5):
    query_embedding = embed_model.encode([query])[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    retrieved_docs = results["documents"][0]
    context = "\n\n".join(retrieved_docs)
    if image_path:
        caption = generate_image_caption(image_path)
        context = f"[Image Insight]: {caption}\n\n{context}"

    prompt = f"""Answer the question based on the context below.

Context:
{context}

Question:
{query}
"""
    headers = {
        "Authorization": f"Bearer {api_key}",
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
    return f"Error: {response.status_code} - {response.text}"
