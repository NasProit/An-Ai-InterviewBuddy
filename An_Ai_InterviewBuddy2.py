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
    "Imagine you are an expert being interviewed for a high-stakes job role. The interviewer asks you to explain [Topic]. Your response should be structured, clear, and engaging, demonstrating both depth of knowledge and effective communication skills.

Start with a concise definition to establish a strong foundation.
Provide a brief introduction explaining its significance and relevance in the industry.
Break down key concepts in simple terms** to ensure clarity.
Walk through a step-by-step explanation, beginning with fundamental principles before progressing to advanced aspects.
Offer a detailed explanation, covering technical details, variations, and nuances where necessary.
Use real-world examples to illustrate practical applications and enhance understanding.
Maintain a professional yet conversational tone, ensuring smooth transitions and logical flow.
Conclude with key takeaways that reinforce your expertise and leave a strong impression.
at the End Give me  explanation flow in a structured tabular format, providing a detailed breakdown for clarity and quick reference.    """),
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
if st.button("📖 Interview Answer"):
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
