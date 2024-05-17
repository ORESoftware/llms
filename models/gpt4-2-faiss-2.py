import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import openai

# Initialize FAISS index
dimension = 384  # This should match the dimension of your embeddings
index = faiss.IndexFlatL2(dimension)

# Initialize SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Set OpenAI API key
openai.api_key = 'YOUR_OPENAI_API_KEY'

# Store texts and their embeddings
texts = []
embeddings = []

def extract_text_from_pdf(pdf_path):
    text = ""
    document = fitz.open(pdf_path)
    for page_num in range(document.page_count):
        page = document.load_page(page_num)
        text += page.get_text()
    return text

def generate_embeddings(text):
    return model.encode(text)

def add_embeddings_to_faiss(text, embedding):
    global texts, embeddings
    texts.append(text)
    embeddings.append(embedding)
    index.add(np.array([embedding]))  # FAISS expects a 2D array

def query_embeddings(query, k=5):
    query_embedding = generate_embeddings(query)
    distances, indices = index.search(np.array([query_embedding]), k)
    return [(texts[i], distances[0][idx]) for idx, i in enumerate(indices[0])]

def generate_chat_completion(prompt, context):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Context: {context}"},
            {"role": "user", "content": prompt}
        ]
    )
    return response['choices'][0]['message']['content']

# Example usage
pdf_path = 'your_pdf_file.pdf'
pdf_text = extract_text_from_pdf(pdf_path)
embedding = generate_embeddings(pdf_text)
add_embeddings_to_faiss(pdf_text, embedding)

query_result = query_embeddings("What is the summary of the document?")
context = ' '.join([res[0] for res in query_result])
chat_response = generate_chat_completion("Can you summarize the document?", context)
print(chat_response)
