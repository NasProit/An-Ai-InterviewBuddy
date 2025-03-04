import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from io import BytesIO

# Load environment variables
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Initialize the model
model = ChatGroq(model="Gemma2-9b-It")

# Define prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", """
     You are an expert coding interviewer conducting a high-stakes technical interview. Your task is to generate the most commonly asked and important coding interview questions related to the given topic.

Instructions for Output:
Generate a list of coding questions that are commonly asked in technical interviews.

Include basic, intermediate, and advanced coding problems.
Cover different aspects such as data structures, algorithms, system design, and optimization techniques.
For each coding question, provide:

A clear problem statement.
Constraints and edge cases that need to be handled.
A step-by-step explanation of the approach to solving the problem.
Optimized solutions with time and space complexity analysis.
Python code implementation (or another language if specified).
Alternative solutions, trade-offs, and best practices.
Maintain a structured and engaging format to ensure clarity and ease of understanding.

Provide a tabular summary at the end, listing all coding problems with key techniques used.

 """),
    ("user", "{input}")
])

# Create the chain
chain = prompt | model

# Streamlit UI
st.set_page_config(page_title="AI Topic Explainer", page_icon="🧠", layout="centered")

st.markdown("<h1 style='text-align: center; color: #2596be;'>🤖 An-AI Interview Buddy by ProitBridge</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #2596be;'>Enter a topic, and the AI will explain it from scratch in simple terms. 🚀</h3>", unsafe_allow_html=True)

# User input
user_input = st.text_input("Enter a topic:")

# Function to create a downloadable text file
def create_text_file(content):
    buffer = BytesIO()
    buffer.write(content.encode())
    buffer.seek(0)
    return buffer

# Generate response
if st.button("📖 Coding Question and Answer"):
    if user_input:
        response = chain.invoke({"input": user_input})
        explanation = response.content if hasattr(response, "content") else response
        
        st.subheader("📌 Explanation:")
        st.write(explanation)

        # Create downloadable file
        text_file = create_text_file(explanation)

        # Provide download button
        st.download_button(
            label="📥 Download as Text File",
            data=text_file,
            file_name=f"{user_input.replace(' ', '_')}_Interview_Answer.txt",
            mime="text/plain"
        )
    else:
        st.warning("⚠️ Please enter a topic to explain.")
