import re
import faiss
from sentence_transformers import SentenceTransformer
import os

# Initialize SentenceTransformer model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Questions and SQL templates
questions_to_queries = {
    "How many students are there in total?": "SELECT COUNT(*) AS total_students FROM student;",
    "How many students are there in each state?": """
        SELECT address.state, COUNT(student.id) AS student_count
        FROM student
        JOIN address ON student.id = address.student_id
        GROUP BY address.state;
    """,
    "Which teachers teach more than one distinct course, and how many courses do they teach?": """
        SELECT teacher, COUNT(DISTINCT course) AS course_count
        FROM class
        GROUP BY teacher
        HAVING COUNT(DISTINCT course) > 0;
    """,
    "What are the names and details of courses for students older than {age}?": """
        SELECT student.name, class.course, class.branch, class.year
        FROM student
        JOIN class ON student.id = class.student_id
        WHERE student.age > {age};
    """,
    
    "What are the details of students including name, email, and their address?": """
        SELECT student.name, student.email_id, address.city, address.state, address.house_number
        FROM student
        JOIN address ON student.id = address.student_id;
    """,
    "How many students are enrolled in each course?": """
        SELECT class.course, COUNT(student.id) AS student_count
        FROM class
        JOIN student ON class.student_id = student.id
        GROUP BY class.course;
    """,
    "What are the details of all students in the database?": "SELECT * FROM student;"
}


# Prepare FAISS index
keys = list(questions_to_queries.keys())
embeddings = embedding_model.encode(keys, convert_to_numpy=True)

# FAISS index setup (local file)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Save FAISS index to disk
faiss.write_index(index, 'faiss_index.index')

# Metadata mapping
metadata = {i: (keys[i], questions_to_queries[keys[i]]) for i in range(len(keys))}

# Save metadata to a text file
with open('metadata.txt', 'w') as f:
    for key, query in metadata.values():
        f.write(f"{key}: {query}\n")

# Function to extract conditions (age, branch, etc.)
def extract_conditions(user_question):
    # Define patterns to extract values from user questions
    patterns = {
        "age": r"older than (\d+)",
        "branch": r"in the (\w+) branch"
    }
    
    conditions = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, user_question)
        if match:
            conditions[key] = match.group(1)
    
    return conditions

# Function to get SQL query from FAISS and conditions
def get_query_from_faiss(user_question):
    # Embed user question
    user_embedding = embedding_model.encode([user_question], convert_to_numpy=True)
    
    # Load the FAISS index from the local file (if necessary)
    faiss_index = faiss.read_index('faiss_index.index')
    
    # Search FAISS index for the closest match
    distances, indices = faiss_index.search(user_embedding, k=1)
    matched_question, query_template = metadata[indices[0][0]]
    
    # Extract conditions from the user's question
    conditions = extract_conditions(user_question)
    
    # Format the SQL query with the extracted conditions
    formatted_query = query_template.format(**conditions)
    
    return matched_question, formatted_query

# Example Usage
user_question = "Who are the students in the Finance branch, and what is their year of study?"
matched_question, sql_query = get_query_from_faiss(user_question)

print(f"Matched Question: {matched_question}")
print(f"Generated SQL Query:\n{sql_query}")

# Output:
# Matched Question: Who are the students in the {branch} branch, and what is their year of study?
# Generated SQL Query:
# SELECT student.name, class.year
# FROM student
# JOIN class ON student.id = class.student_id
# WHERE class.branch = 'Finance';
