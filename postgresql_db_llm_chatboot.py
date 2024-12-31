from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
import psycopg2
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Retrieve OpenAI API key from the environment
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OpenAI API key not found in .env file. Please add 'OPENAI_API_KEY'.")

# PostgreSQL connection parameters
db_params = {
    "dbname": "work1",  # Your database name
    "user": "postgres",  # Your database user
    "password": "admin123",  # Your database password
    "host": "localhost",  # Your host (localhost if local)
    "port": "5432"  # Default PostgreSQL port
}

# Define the OpenAI prompt template
template = """Based on the table schema below, write a SQL query that would answer the user's question, use alias in joins:
{schema}

Question: {question}

SQL Query:"""

prompt = ChatPromptTemplate.from_template(template)

# Define function to get PostgreSQL schema
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

# Set up LangChain pipeline with OpenAI
llm = ChatOpenAI(openai_api_key=openai_api_key)

sql_chain = (
    RunnablePassthrough.assign(schema=lambda _: get_schema(db_params))
    | prompt
    | llm.bind(stop=["\nSQLResult:"])
    | StrOutputParser()
)

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
