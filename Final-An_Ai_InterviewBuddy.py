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
def initialize_model():
    return ChatGroq(model="Gemma2-9b-It")

# Define prompt templates
def get_prompt(prompt_type):
    prompts = {
        "explanation": """
        Explain the topic in simple terms, assuming the reader has no prior knowledge.

        Start with a clear definition to establish what the topic is.
        Provide a basic introduction that explains its significance and relevance.
        Define and break down key concepts so that fundamental terms are easy to understand.
        Offer a step-by-step explanation to logically and clearly break down how it works.
        Include a detailed explanation, diving deeper into the mechanics, variations, or advanced aspects of the topic.
        Use real-world examples to illustrate practical applications and enhance understanding.
        Maintain a smooth, logical flow, ensuring clarity while transitioning from basic to advanced aspects.
        Conclude with a summary of key takeaways to reinforce learning..
        At the end, provide a structured table summarizing key points.
        """
        ,

        "interview_questions": """
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

        """,
        "coding_questions": """
       You are an expert coding interviewer conducting a high-stakes technical interview. Your task is to generate the most commonly asked and important coding interview questions related to the given topic.

        Instructions for Output:
        Generate a list of atleast 10 coding questions that are commonly asked in technical interviews.

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
        """
        ,
        "Real-Time Use Cases and Future Advancementsons":

        """Role: You are an industry expert explaining how a given concept is applied in real-world scenarios. Your task is to generate detailed real-time applications, showcasing its practical significance, industry use cases, and the next advancements that improve or replace it.

        Instructions for Output:
        Start with a concise definition of the concept to establish clarity.
        Explain its significance in solving real-world problems.
        Provide real-world use cases, covering:
        Industry-Specific Applications (e.g., Finance, Healthcare, E-commerce, Manufacturing).
        How businesses leverage this concept to solve problems efficiently.
        Well-known companies or products using it in their systems.
        Describe technical implementation details, including:
        How the concept is integrated into real-world systems.
        Common tools, frameworks, or technologies used for implementation.
        Challenges faced and how they are addressed.
        Discuss limitations and drawbacks of the existing method.
        Explain recent advancements and alternatives that are solving those limitations.
        Provide a future outlook, discussing potential improvements and upcoming trends.
        Summarize key takeaways, emphasizing real-world impact and ongoing evolution.
        At the end, include a structured summary table, comparing traditional and modern advancements.
"""


    }
    return ChatPromptTemplate.from_messages([
        ("system", prompts[prompt_type]),
        ("user", "{input}")
    ])

# Function to create a downloadable text file
def create_text_file(content):
    buffer = BytesIO()
    buffer.write(content.encode())
    buffer.seek(0)
    return buffer

# Streamlit UI
st.set_page_config(page_title="AI Interview Buddy", page_icon="🧠", layout="centered")
st.markdown("<h1 style='text-align: center; color: #2596be;'>🤖 An  AI Interview Buddy by ProitBridge</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #2596be;'>Enter a topic and select the type of response 🚀</h3>", unsafe_allow_html=True)

# User input
topic = st.text_input("Enter a topic:")

# Generate response based on button clicks
col1, col2, col3,col4 = st.columns(4)

with col1:
    if st.button("📖  Explanation"):
        selected_option = "explanation"
with col2:
    if st.button("📝  Interview Q&A"):
        selected_option = "interview_questions"
with col3:
    if st.button("💻  Coding Q&A"):
        selected_option = "coding_questions"

with col4:
    if st.button("💻 Real-Time Use Cases and Future Advancementsons"):
        selected_option = "Real-Time Use Cases and Future Advancementsons"


        

if topic and 'selected_option' in locals():
    model = initialize_model()
    prompt = get_prompt(selected_option)
    chain = prompt | model
    response = chain.invoke({"input": topic})
    explanation = response.content if hasattr(response, "content") else response
    
    st.subheader("📌 Response:")
    st.write(explanation)
    
    # Create downloadable file
    text_file = create_text_file(explanation)
    
    # Provide download button
    st.download_button(
        label="📥 Download as Text File",
        data=text_file,
        file_name=f"{topic.replace(' ', '_')}_Response.txt",
        mime="text/plain"
    )
