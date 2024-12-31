import os
import faiss
import psycopg2
import pandas as pd
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv  # Import dotenv to load environment variables
from urllib.parse import urlparse

# Load environment variables from .env file
load_dotenv()

# Retrieve the OpenAI API key from the .env file
openai_api_key = os.getenv('OPENAI_API_KEY')

# Ensure OpenAI API key is loaded
if not openai_api_key:
    raise ValueError("OpenAI API key not found. Please check your .env file.")

# Retrieve DATABASE_URL from environment variables
DATABASE_URL = os.getenv('DATABASE_URL')

# Ensure DATABASE_URL is loaded
if not DATABASE_URL:
    raise ValueError("DATABASE_URL not found. Please check your .env file.")

# Parse the DATABASE_URL to extract database connection parameters
url = urlparse(DATABASE_URL)
db_params = {
    "dbname": url.path[1:],  # Database name (after the slash)
    "user": url.username,     # Username
    "password": url.password, # Password
    "host": url.hostname,     # Hostname
    "port": url.port or 5432  # Port (default to 5432 if not provided)
}

# Local file paths for FAISS index and metadata
metadata_file = "schema_metadata.txt"
index_file = "faiss_index_file.index"

# Ensure necessary files exist
if not os.path.exists(metadata_file) or not os.path.exists(index_file):
    raise FileNotFoundError("Metadata or FAISS index file not found.")

# Load the FAISS index and metadata
index = faiss.read_index(index_file)
with open(metadata_file, 'r') as f:
    metadata = f.read().splitlines()

# Initialize SentenceTransformer model for embedding queries
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Define the OpenAI prompt template
template = """Based on the FAISS vector database metadata below, write an SQL query to answer the user's question, use alias in joins:
{metadata}

Question: {question}

SQL Query:"""

prompt = ChatPromptTemplate.from_template(template)

# Set up LangChain pipeline with OpenAI
llm = ChatOpenAI(openai_api_key=openai_api_key)  # Use API key from .env file

vector_chain = (
    RunnablePassthrough.assign(metadata=lambda _: "\n".join(metadata))
    | prompt
    | llm
    | StrOutputParser()
)

# Function to query the PostgreSQL database and return results
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

# Function to get the embedding for a query
def get_embedding(query):
    return embedding_model.encode(query, convert_to_numpy=True)

# Streamlit UI
st.title("SQL Query Generator with PostgreSQL and FAISS")

# Input for user question
user_question = st.text_input("Enter your question:")

# Button to trigger query generation and execution
execute_query = st.button("Generate and Execute Query")

if execute_query and user_question:
    # Step 1: Generate embedding for the user's question
    query_embedding = get_embedding(user_question)
    
    # Step 2: Perform search in FAISS index
    distances, indices = index.search(query_embedding.reshape(1, -1), k=6)  # Search for top 5 matches
    
    # Step 3: Extract metadata results from FAISS
    relevant_metadata = [metadata[i] for i in indices[0]]
    st.write("### Relevant Schema Metadata Based on Your Query:")
    st.write("\n".join(relevant_metadata))  # Display relevant metadata
    
    # Step 4: Use LangChain to generate SQL query based on metadata and question
    sql_query = vector_chain.invoke({"metadata": "\n".join(relevant_metadata), "question": user_question})
    
    # Display generated SQL query
    st.markdown(f"### Generated SQL Query:\n```sql\n{sql_query}\n```")

    # Step 5: Execute the generated SQL query in PostgreSQL
    query_results = execute_sql_query(sql_query)
    
    # Display results in a DataFrame format
    if isinstance(query_results, pd.DataFrame):
        st.write("### Query Results")
        st.dataframe(query_results)
    else:
        st.error(query_results)
