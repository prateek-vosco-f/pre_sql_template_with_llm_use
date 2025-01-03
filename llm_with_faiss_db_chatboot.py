import os
import re
import faiss
import psycopg2
import pandas as pd
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from urllib.parse import urlparse
from langchain.chains.question_answering import load_qa_chain  # Import QA Chain
from langchain.prompts import PromptTemplate  # Import for custom QA templates
from langchain.llms import OpenAI

# Load environment variables
load_dotenv()

# Retrieve the OpenAI API key from the .env file
openai_api_key = os.getenv('OPENAI_API_KEY')

if not openai_api_key:
    raise ValueError("OpenAI API key not found. Please check your .env file.")

# Retrieve DATABASE_URL from environment variables
DATABASE_URL = os.getenv('DATABASE_URL')

if not DATABASE_URL:
    raise ValueError("DATABASE_URL not found. Please check your .env file.")

# Parse DATABASE_URL to extract connection parameters
url = urlparse(DATABASE_URL)
db_params = {
    "dbname": url.path[1:],
    "user": url.username,
    "password": url.password,
    "host": url.hostname,
    "port": url.port or 5432
}

# Local file paths for FAISS index and metadata
metadata_file_1 = "schema_metadata.txt"
index_file_1 = "faiss_index_file.index"
metadata_file = "metadata.txt"
index_file = "faiss_index.index"

# Check if metadata and FAISS files exist
if not os.path.exists(metadata_file) or not os.path.exists(index_file):
    raise FileNotFoundError("Metadata or FAISS index file not found.")

# Function to load metadata
def load_metadata(file_path):
    """
    Load metadata from the file, ensuring multi-line SQL queries are correctly associated.
    """
    metadata = {}
    key = None
    value = []
    with open(file_path, "r") as f:
        for line in f:
            if ": " in line:  # Start of a new key-value pair
                if key and value:  # Save the previous key-value pair
                    metadata[key] = "\n".join(value).strip()
                key, sql_part = line.split(": ", 1)
                key = key.strip()
                value = [sql_part.strip()]  # Initialize new value list
            else:  # Continuation of the SQL query
                value.append(line.strip())
        # Save the last key-value pair
        if key and value:
            metadata[key] = "\n".join(value).strip()
    return metadata


# Load metadata
metadata = load_metadata(metadata_file)

# Load FAISS index
faiss_index = faiss.read_index(index_file)

# Initialize SentenceTransformer
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Function to extract dynamic parameters from user question
def extract_parameters(user_question):
    """
    Extract dynamic parameters like `age` or `branch` from the user's question.
    """
    patterns = {
        "age": r"older than (\d+)",
        "branch": r"in the (\w+) branch",
    }
    extracted_params = {}
    for param, pattern in patterns.items():
        match = re.search(pattern, user_question)
        if match:
            extracted_params[param] = match.group(1)
    return extracted_params

# Function to dynamically replace parameters in SQL query
def format_query(sql_template, parameters):
    """
    Replaces placeholders in SQL with extracted parameters.
    """
    try:
        return sql_template.format(**parameters)
    except KeyError as e:
        raise ValueError(f"Missing required parameter: {e}")

# Function to get best matching SQL query using FAISS
def get_best_query(user_question, metadata, k=5, threshold=0.5):
    """
    Searches FAISS for the best matching query using vector embeddings.

    Args:
    - user_question (str): The question input by the user.
    - metadata (dict): Metadata containing questions and SQL queries.
    - k (int): Number of nearest neighbors to consider.
    - threshold (float): Minimum similarity threshold.

    Returns:
    - tuple: (Matched question, SQL template) or None if no match is found.
    """
    user_embedding = embedding_model.encode([user_question], convert_to_numpy=True)
    distances, indices = faiss_index.search(user_embedding, k=k)
    

    # Filter results by threshold
    matched_queries = [
        (list(metadata.keys())[idx], metadata[list(metadata.keys())[idx]])
        for idx, distance in zip(indices[0], distances[0]) if distance < threshold
    ]

    if matched_queries:
        return matched_queries[0]  # Return the best match (closest distance)
    return None

# Load additional FAISS index and metadata
if not os.path.exists(metadata_file_1) or not os.path.exists(index_file_1):
    raise FileNotFoundError("Secondary metadata or FAISS index file not found.")

index_1 = faiss.read_index(index_file_1)
with open(metadata_file_1, 'r') as f:
    metadata_1 = f.read().splitlines()

# Define OpenAI prompt template for SQL generation
template = """Based on the FAISS vector database metadata below, write an SQL query to answer the user's question, use alias in joins:
{metadata_2}

Question: {question}

SQL Query:"""

prompt = ChatPromptTemplate.from_template(template)

llm = ChatOpenAI(openai_api_key=openai_api_key)

vector_chain = (
    RunnablePassthrough.assign(metadata=lambda _: "\n".join(metadata))
    | prompt
    | llm
    | StrOutputParser()
)

# Function to execute SQL query
def execute_sql_query(sql_query):
    try:
        with psycopg2.connect(**db_params) as conn:
            with conn.cursor() as cur:
                cur.execute(sql_query)
                results = cur.fetchall()
                df = pd.DataFrame(results, columns=[desc[0] for desc in cur.description])
                return df
    except Exception as e:
        return f"Error executing query: {e}"

# Define custom QA template
qa_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""\
You are a helpful assistant. Using the context below, generate a detailed yet concise summary of the query results.

Context:
{context}

Question:
{question}

Summary:
"""
)

# Function to generate summary using a custom QA chain
def summarize_query_results(question, query_results):
    if isinstance(query_results, pd.DataFrame):
        context = query_results.to_string(index=False)  # Convert DataFrame to string context
    else:
        context = query_results  # Handle case for errors or non-DataFrame results

    formatted_query = qa_template.format(context=context, question=question)

    # Load the QA chain
    chain = load_qa_chain(OpenAI(openai_api_key=openai_api_key), chain_type="stuff")
    summary = chain.run(input_documents=[], question=formatted_query)  # No input docs, use context in formatted query
    return summary

# Function to get embeddings
def get_embedding(query):
    return embedding_model.encode(query, convert_to_numpy=True)

# Streamlit UI
st.title("SQL Query Generator with Dynamic Parameters")

user_question = st.text_input("Enter your question:")
execute_query = st.button("Generate and Execute Query")

if execute_query and user_question:
    # Step 1: Extract parameters from the user's question
    extracted_params = extract_parameters(user_question)

    # Step 2: Search for the best matching query
    matched_result = get_best_query(user_question, metadata)
    

    if matched_result:
        matched_key, sql_template = matched_result

        # Step 3: Format the SQL query with extracted parameters
        sql_query = format_query(sql_template, extracted_params)
        st.write("Use SQL Template:")
          # Display the formatted SQL query with dynamic parameters

    if not matched_result: 
         # Use secondary index and metadata if primary SQL query is null
        query_embedding = get_embedding(user_question)

        # Step 2: Search FAISS index
        distances, indices = index_1.search(query_embedding.reshape(1, -1), k=6)

        # Step 3: Extract metadata results
        relevant_metadata = [metadata_1[i] for i in indices[0]]
        

        # Step 4: Generate SQL query
        sql_query = vector_chain.invoke({"metadata_2": "\n".join(relevant_metadata), "question": user_question})
        st.write("Generated SQL Query using LLM:")
          # Display the generated SQL query from the LLM

    # Step 4: Execute the SQL query
    query_results = execute_sql_query(sql_query)

    # Step 5: Display results
    summary = summarize_query_results(user_question, query_results)

    if isinstance(query_results, pd.DataFrame):
        st.write("### Summary")
        st.write(summary)
        st.write("### Query Results")
        st.dataframe(query_results)
    else:
        st.error(query_results)
