import heapq
import numpy as np
from wikipediaapi import Wikipedia
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
wiki = Wikipedia('RAGBot/0.0', 'en')

doc = wiki.page('Hayao_Miyazaki').text
paragraphs = doc.split('\n\n')
docs_embed = model.encode(paragraphs, normalize_embeddings=True)

query = "What was Studio Ghibli's first film?"
query_embed = model.encode([query], normalize_embeddings=True)

similarities = np.dot(docs_embed, query_embed.T)
top_3_idx = heapq.nlargest(3, range(len(similarities)), similarities.take)
most_similar_documents = [paragraphs[idx] for idx in top_3_idx]

def main():
    top = most_similar_documents

    for t in top:
        print(t)
