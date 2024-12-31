import os
import psycopg2
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from urllib.parse import urlparse

# Load environment variables from .env file (only needed locally)
load_dotenv()

# Retrieve the OpenAI API key and PostgreSQL DATABASE_URL from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OpenAI API key not found in environment variables. Please add 'OPENAI_API_KEY'.")

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL not found in environment variables. Please add it to Render's environment.")

# Parse the DATABASE_URL to extract database connection parameters
url = urlparse(DATABASE_URL)
db_params = {
    "dbname": url.path[1:],  # Database name (after the slash)
    "user": url.username,     # Username
    "password": url.password, # Password
    "host": url.hostname,     # Hostname
    "port": url.port or 5432  # Port (default to 5432 if not provided)
}

# Define the OpenAI prompt template
template = """Based on the table schema below, write a SQL query that would answer the user's question, use alias in joins:
{schema}

Question: {question}

SQL Query:"""

# Define the LangChain prompt object
prompt = ChatPromptTemplate.from_template(template)

# Set up LangChain pipeline with OpenAI
llm = ChatOpenAI(openai_api_key=openai_api_key)

sql_chain = (
    RunnablePassthrough.assign(schema=lambda _: get_schema(db_params))
    | prompt
    | llm.bind(stop=["\nSQLResult:"])
    | StrOutputParser()
)

# Function to retrieve PostgreSQL schema
def get_schema(db_params):
    try:
        # Connect to the PostgreSQL database
        with psycopg2.connect(**db_params) as conn:
            with conn.cursor() as cur:
                # Query to get table schema
                cur.execute("""
                    SELECT table_name, column_name, data_type 
                    FROM information_schema.columns 
                    WHERE table_schema = 'public';
                """)
                schema = cur.fetchall()
                schema_str = "\n".join([f"{row[0]}.{row[1]} ({row[2]})" for row in schema])
                return schema_str
    except Exception as e:
        return f"Error retrieving schema: {e}"

# Streamlit interface
st.title("Ask Question Based on Database")
user_question = st.text_input("Enter your question:")
execute_query = st.button("Generate and Execute Query")

if execute_query and user_question:
    # Generate the SQL query using LangChain
    schema_str = get_schema(db_params)
    
    sql_query = sql_chain.invoke({"schema": schema_str, "question": user_question})

    st.markdown(f"### Generated SQL Query:\n```sql\n{sql_query}\n```")

    # Execute the SQL query in PostgreSQL
    try:
        with psycopg2.connect(**db_params) as conn:
            with conn.cursor() as cur:
                cur.execute(sql_query)
                results = cur.fetchall()

                # Convert results to a DataFrame for better display
                df = pd.DataFrame(results, columns=[desc[0] for desc in cur.description])
                st.write("### Query Results")
                st.dataframe(df)
    except Exception as e:
        st.error(f"Error executing query: {e}")
