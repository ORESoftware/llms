from openai import OpenAI
import fitz  # PyMuPDF
import pandas as pd
import torch
import os
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForCausalLM
import faiss
import csv
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
            choice_str = input("(Choose an option by integer): ")

            if choice_str in options:
                return choice_str

            choice = int(choice_str)

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

def load_csv(csv_str):
    question = "Load a CSV file?"
    options = ["Yes", "No"]
    display_menu(question, options)
    choice = get_user_choice(question, options)
    print(f"You selected: {options[choice - 1]}")
    if options[choice - 1] == "No":
        return csv_str
    csv_path = input("Enter the path to the CSV file: ")
    try:
        csv_str += extract_text_from_csv(csv_path) + "\n"
        print('Loaded file successfully.')
    except Exception as e:
        print(e)
    finally:
        return load_csv(csv_str)

def get_str_from_template(template, *args):
    # Format the template using the provided arguments
    return template.format(*args)

def get_templates_from_csv_header(csv_path):
    templates = []
    with open(csv_path, mode='r', encoding='utf-8') as file:
        for line in file:
            if line.startswith("#"):
                # Remove the first character '#' and strip any trailing newline characters
                templates.append(line[1:].strip())
    return templates

def extract_text_from_csv(csv_path):
    templates = get_templates_from_csv_header(csv_path)
    df = pd.read_csv(csv_path, skiprows=len(templates))
    print("Columns:")
    print("Columns:", df.columns)  # Print the columns for inspection
    combined_text = ""
    for index, row in df.iterrows():
        row_tuple = tuple(row)
        for template in templates:
            combined_text += get_str_from_template(template, *row_tuple) + "\n"

    return combined_text

def query_the_data():
    query = input("Please input your data query: ")
    return query

def truncate_string(input_string, max_length=8000):
    return input_string[:max_length]

# Example usage:

def main():
    select_model()
    select_machine()
    pdf_str = load_pdf("")
    csv_str = load_csv("")
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

    truncated_pdf_string = truncate_string(pdf_str)
    truncated_csv_string = truncate_string(csv_str)

    print(truncated_csv_string)

    # Define your conversation
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": truncated_pdf_string},
        {"role": "user", "content": truncated_csv_string},
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
