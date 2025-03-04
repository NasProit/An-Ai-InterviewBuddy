
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Initialize the model
model = ChatGroq(model="gemma2-9b-it")

# Define prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", """
    Explain the topic in simple terms, assuming the reader has no prior knowledge.

Start with a clear definition to establish what the topic is.
Provide a basic introduction that explains its significance and relevance.
Define and break down key concepts so that fundamental terms are easy to understand.
Offer a step-by-step explanation to logically and clearly break down how it works.
Include a detailed explanation, diving deeper into the mechanics, variations, or advanced aspects of the topic.
Use real-world examples to illustrate practical applications and enhance understanding.
Maintain a smooth, logical flow, ensuring clarity while transitioning from basic to advanced aspects.
Conclude with a summary of key takeaways to reinforce learning.
.
    """),
    ("user", "{input}")
])

# Create the chain
chain = prompt | model

# Streamlit UI with custom styling
st.set_page_config(page_title="AI Topic Explainer", page_icon="🧠", layout="centered")
st.markdown(
    """
    <style>
        body {
            background-color: white;
        }
        .title {
            color: #2596be;
            text-align: center;
            font-size: 36px;
            font-weight: bold;
        }
        .subtitle {
            color: #2596be;
            text-align: center;
            font-size: 20px;
        }
        .stTextInput input {
            border: 2px solid #cb6ce6;
            border-radius: 10px;
            padding: 8px;
        }
        .stButton>button {
            background-color: #2596be;
            color: white;
            text-align: center
            border-radius: 10px;
            padding: 10px;
            font-size: 16px;
        }
        .stButton>button:hover {
            background-color: #0b0039;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# UI Content
st.markdown("<div class='title'>🤖 An-Ai InterviewBuddy by ProitBridge </div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Enter a topic, and the AI will explain it from scratch in simple terms. 🚀</div>", unsafe_allow_html=True)

# User input
user_input = st.text_input("Enter a topic:")

# Generate response
import re

# Generate response
if st.button("📖 Explain"):
    if user_input:
        response = chain.invoke({"input": user_input})
        explanation = response.content if hasattr(response, "content") else response
        
        # Remove unwanted thoughts (if they appear)
        explanation = re.sub(r"<think>.*?</think>", "", explanation, flags=re.DOTALL).strip()

        st.subheader("📌 Explanation:")
        st.write(explanation)
    else:
        st.warning("⚠️ Please enter a topic to explain.")

