from openai import OpenAI
import fitz  # PyMuPDF
import pandas as pd
import torch
import os
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForCausalLM
import faiss
import numpy as np

# Initialize OpenAI API client
openai_client = OpenAI(
    api_key=os.getenv("open_ai_key"),
)

def display_menu(question, options):
    print(question + " - Please select an option:")
    for i, option in enumerate(options, 1):
        print(f"{i}. {option}")

def get_user_choice(question, options):
    while True:
        try:
            choice = int(input("(Choose an option by integer): "))
            if 1 <= choice <= len(options):
                return choice
            else:
                print("Invalid choice. Please select a number from the list.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def select_model():
    question = "Please select a model engine."
    options = ["OpenAI/GPT4", "Mistral", "Titan", "Gemini"]
    display_menu(question, options)
    choice = get_user_choice(question, options)
    print(f"You selected: {options[choice - 1]}")
    return options[choice - 1]

def select_machine():
    question = "Please select a region to run the client on."
    options = ["us-west-1", "us-east-1", "ap-east-2", "eu-west-1"]
    display_menu(question, options)
    choice = get_user_choice(question, options)
    print(f"You selected: {options[choice - 1]}")
    return options[choice - 1]

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def load_pdf(pdf_str):
    question = "Load a PDF file?"
    options = ["Yes", "No"]
    display_menu(question, options)
    choice = get_user_choice(question, options)
    print(f"You selected: {options[choice - 1]}")
    if options[choice - 1] == "No":
        return pdf_str
    pdf_path = input("Enter the path to the PDF file: ")
    try:
        pdf_str += extract_text_from_pdf(pdf_path)
        print('Loaded file successfully.')
    except Exception as e:
        print(e)
    finally:
        return load_pdf(pdf_str)

def query_the_data():
    query = input("Please input your data query: ")
    return query

def main():
    select_model()
    select_machine()
    pdf_str = load_pdf("")
    user_query = query_the_data()

    # Convert PDF text to vectors using Sentence-Transformers
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    pdf_vectors = model.encode([pdf_str])

    # Initialize FAISS index
    index = faiss.IndexFlatL2(pdf_vectors.shape[1])
    index.add(np.array(pdf_vectors, dtype=np.float32))

    # Store user query as vector
    query_vector = model.encode([user_query])

    # Search FAISS index
    D, I = index.search(np.array(query_vector, dtype=np.float32), k=1)
    print(f"Closest match in the database: {pdf_str}")

    # Define your conversation
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": pdf_str},
        {"role": "user", "content": user_query}
    ]

    # Generate a response
    response = openai_client.chat.completions.create(
        model="gpt-4",  # Specify the GPT-4 model
        messages=messages,
        max_tokens=1000
    )

    # Print the response
    print('raw response:')
    print(response)

if __name__ == "__main__":
    main()
