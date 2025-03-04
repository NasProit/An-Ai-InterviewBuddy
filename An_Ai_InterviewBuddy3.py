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
     Role: Act as an expert interviewer conducting a high-stakes job interview. Your task is to generate the most commonly asked and important interview questions related to the given topic.

For each question, provide a detailed, structured, and well-explained answer that demonstrates deep expertise and clear communication.

Instructions for Output:
Generate a list of the most frequently asked and critical interview questions on the given topic. Ensure a mix of basic, intermediate, and advanced questions.
For each question, provide a well-structured answer:
Start with a concise definition to establish clarity.
Explain its significance in the industry and why it matters.
Break down key concepts in simple terms.
Provide a step-by-step explanation, starting from the fundamentals to advanced aspects.
Include variations and nuances where relevant.
Use real-world examples to illustrate practical applications.
Mention best practices, common mistakes, and optimization techniques where applicable.
Maintain a professional yet conversational tone for clarity and engagement.
Ensure smooth transitions and logical flow between points.
Conclude with key takeaways that reinforce expertise and leave a strong impression.
At the end, provide a structured tabular format summarizing all the questions and key points from the answers for quick reference.
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
if st.button("📖 Interview Question and Answer"):
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
