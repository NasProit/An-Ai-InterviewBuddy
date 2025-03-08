import os
import streamlit as st
import pandas as pd
import json
import uuid
import hashlib
from datetime import datetime
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory, FileChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from io import BytesIO

# Load environment variables
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Directory for storing user data and chat histories
DATA_DIR = "user_data"
HISTORY_DIR = "chat_histories"

# Ensure directories exist
for directory in [DATA_DIR, HISTORY_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# User authentication functions
def get_user_file_path(username):
    return os.path.join(DATA_DIR, f"{username}.json")

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password):
    user_file = get_user_file_path(username)
    if os.path.exists(user_file):
        return False  # User already exists
    
    user_data = {
        "username": username,
        "password_hash": hash_password(password),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "last_login": None
    }
    
    with open(user_file, 'w') as f:
        json.dump(user_data, f)
    
    return True

def verify_user(username, password):
    user_file = get_user_file_path(username)
    if not os.path.exists(user_file):
        return False
    
    with open(user_file, 'r') as f:
        user_data = json.load(f)
    
    is_valid = user_data["password_hash"] == hash_password(password)
    
    if is_valid:
        # Update last login time
        user_data["last_login"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(user_file, 'w') as f:
            json.dump(user_data, f)
    
    return is_valid

# Chat history functions
def get_history_file_path(username, topic, subtopic):
    safe_topic = topic.replace(" ", "_")
    safe_subtopic = subtopic.replace(" ", "_").replace("(", "").replace(")", "")
    filename = f"{username}_{safe_topic}_{safe_subtopic}.json"
    return os.path.join(HISTORY_DIR, filename)

def get_session_histories(username):
    user_histories = []
    for filename in os.listdir(HISTORY_DIR):
        if filename.startswith(f"{username}_") and filename.endswith(".json"):
            parts = filename[:-5].split("_")  # Remove .json and split
            if len(parts) >= 3:
                topic = parts[1]
                subtopic = "_".join(parts[2:])
                subtopic = subtopic.replace("_", " ")
                user_histories.append({
                    "filename": filename,
                    "topic": topic.replace("_", " "),
                    "subtopic": subtopic,
                    "last_modified": os.path.getmtime(os.path.join(HISTORY_DIR, filename))
                })
    
    # Sort by last modified date (newest first)
    user_histories.sort(key=lambda x: x["last_modified"], reverse=True)
    return user_histories

# Initialize the model
def initialize_model():
    return ChatGroq(model="Gemma2-9b-It")

# Define prompt templates
def get_prompt(prompt_type):
    prompts = {
        "explanation": """
        Explain the data science topic in simple terms, assuming the reader has no prior knowledge.

        Start with a clear definition to establish what the topic is.
        Provide a basic introduction that explains its significance and relevance in data science.
        Define and break down key concepts so that fundamental terms are easy to understand.
        Offer a step-by-step explanation to logically and clearly break down how it works.
        Include a detailed explanation, diving deeper into the mechanics, variations, or advanced aspects of the topic.
        Use real-world data science examples to illustrate practical applications and enhance understanding.
        Maintain a smooth, logical flow, ensuring clarity while transitioning from basic to advanced aspects.
        Conclude with a summary of key takeaways to reinforce learning.
        At the end, provide a structured table summarizing key points.
        """,

        "coding_questions": """
        You are an expert data science interviewer conducting a technical interview. Your task is to generate the most commonly asked and important coding interview questions related to the given data science topic.

        Instructions for Output:
        Generate a list of 5 coding questions that are commonly asked in data science technical interviews.

        Include basic, intermediate, and advanced coding problems.
        Cover different aspects such as implementation, analysis, and optimization techniques.
        For each coding question, provide:

        A clear problem statement related to data science.
        Constraints and edge cases that need to be handled.
        A step-by-step explanation of the approach to solving the problem.
        Optimized solutions with time and space complexity analysis.
        Python code implementation with common data science libraries if relevant.
        Alternative solutions, trade-offs, and best practices.
        Maintain a structured and engaging format to ensure clarity and ease of understanding.

        Provide a tabular summary at the end, listing all coding problems with key techniques used.
        """,
        
        "suggestions": """
        You are a data science mentor guiding a student through their learning journey. For the given topic, provide comprehensive learning resources and suggestions.

        Instructions for Output:
        Start with a brief overview of why this topic is important in data science.
        Provide a structured learning path from beginner to advanced level.
        
        For each level (Beginner, Intermediate, Advanced), recommend:
        - 2-3 specific online courses or tutorials (with actual names, not generic references)
        - 1-2 books that cover the topic well
        - 1-2 practical projects to reinforce learning
        - Common pitfalls to avoid when learning this topic
        
        Include suggestions for:
        - Free online resources (documentation, YouTube channels, blogs)
        - Communities or forums where learners can get help
        - Tools or libraries that professionals use when working with this topic
        - How this topic connects to other areas in data science
        
        End with a motivational note on how mastering this topic will help in a data science career.
        At the end, provide a structured table summarizing all key resources.
        """
    }
    return ChatPromptTemplate.from_messages([
        ("system", prompts[prompt_type]),
        ("user", "The data science topic is: {topic} - {subtopic}")
    ])

# Load data from Excel
def load_roadmap_data(file_path=None):
    # This is a placeholder. In a real app, you would load data from an Excel file
    # For now, we'll create sample data
    data = {'Python': [
        'Syntax and Variables', 'Data Types and Type Conversion', 'Operators',
        'Conditional Statements', 'Loops (For, While, Nested Loops)', 
        'Functions (Built-in, User-defined, Lambda, Recursion)', 'Exception Handling', 
        'File Handling (Read, Write, Append, CSV, JSON, Pickle)', 'OOP Concepts (Classes, Objects, Inheritance, Polymorphism, Encapsulation)', 
        'Modules and Packages', 'Regular Expressions', 'List Comprehensions', 'Iterators and Generators',
        'Multithreading and Multiprocessing', 'Virtual Environments and Package Management (pip, conda)'
    ],
    
    'Data Structures': [
        'Lists, Tuples, Dictionaries, Sets', 'Stack (LIFO)', 'Queue (FIFO, Priority Queue, Deque)',
        'Linked Lists (Singly, Doubly, Circular)', 'Trees (Binary Tree, BST, AVL, Red-Black Tree)',
        'Graphs (Adjacency List, Adjacency Matrix, DFS, BFS)', 'Hashing (Hash Functions, Collisions, Chaining)'
    ],
    
    'Libraries (NumPy, Pandas)': [
        'NumPy Arrays (Operations, Indexing, Broadcasting)', 'Mathematical and Statistical Functions',
        'Linear Algebra with NumPy', 'Pandas DataFrames and Series', 'Reading/Writing Data (CSV, Excel, JSON, SQL)',
        'Handling Missing Data', 'GroupBy and Aggregations', 'Merging, Joining, and Concatenation',
        'Pivot Tables', 'Time Series Analysis', 'Multi-indexing'
    ],

    'Machine Learning': [
        'Types of ML (Supervised, Unsupervised, Reinforcement Learning)',
        'Feature Engineering (Scaling, Encoding, Imputation, Feature Selection)',
        'Linear Regression (Simple, Multiple, Ridge, Lasso)', 'Logistic Regression', 'Decision Trees (Gini, Entropy, Pruning)',
        'Random Forests (Bagging, Feature Importance)', 'Support Vector Machines (SVM, Kernel Trick)',
        'K-Nearest Neighbors (KNN)', 'Naive Bayes', 'Gradient Boosting (AdaBoost, XGBoost, LightGBM, CatBoost)',
        'Neural Networks (Perceptron, MLP, Activation Functions, Backpropagation)', 'Hyperparameter Tuning (Grid Search, Random Search, Bayesian Optimization)',
        'Model Evaluation Metrics (Confusion Matrix, Precision, Recall, F1-score, ROC-AUC, Log Loss)'
    ],

    'Deep Learning': [
        'Perceptron and Artificial Neural Networks (ANN)', 'Activation Functions (ReLU, Sigmoid, Tanh, Softmax)',
        'Loss Functions (Cross-Entropy, MSE, MAE)', 'Gradient Descent and Backpropagation',
        'Optimization Algorithms (SGD, Adam, RMSprop)', 'Convolutional Neural Networks (CNNs)',
        'Recurrent Neural Networks (RNNs, LSTM, GRU)', 'Transformers and Self-Attention',
        'Generative Adversarial Networks (GANs)', 'Autoencoders', 'Transfer Learning', 'Reinforcement Learning (Q-Learning, Policy Gradient)',
        'Frameworks (TensorFlow, PyTorch, Keras)'
    ],

    'NLP (Natural Language Processing)': [
        'Text Preprocessing (Tokenization, Lemmatization, Stopwords, Stemming)',
        'Word Embeddings (TF-IDF, Word2Vec, GloVe, FastText)', 'Text Classification',
        'Named Entity Recognition (NER)', 'Part-of-Speech Tagging', 'Sentiment Analysis',
        'Topic Modeling (LDA, LSA)', 'Sequence Models (RNN, LSTM, Transformers)', 
        'Attention Mechanisms', 'Transformers (BERT, GPT, T5)', 'Speech Recognition and Text-to-Speech (TTS)',
        'Chatbots and Conversational AI'
    ],

    'Statistics': [
        'Descriptive Statistics (Mean, Median, Mode, Variance, Standard Deviation)', 'Inferential Statistics',
        'Probability Theory (Bayes’ Theorem, Conditional Probability)', 'Distributions (Normal, Poisson, Binomial, Exponential, Chi-Square)',
        'Hypothesis Testing (Z-test, T-test, ANOVA, Chi-Square Test, A/B Testing)', 'Correlation and Covariance',
        'Central Limit Theorem', 'Bias-Variance Tradeoff', 'Resampling Methods (Bootstrapping, Cross-validation)'
    ],

    'SQL & Databases': [
        'Database Concepts (Normalization, ACID Properties, Transactions)',
        'Basic Queries (SELECT, INSERT, UPDATE, DELETE)', 'WHERE and HAVING Clauses',
        'Joins (Inner, Outer, Left, Right, Cross)', 'Subqueries (Nested Queries, CTEs)',
        'Aggregate Functions (COUNT, SUM, AVG, MIN, MAX)', 'Window Functions (ROW_NUMBER, RANK, DENSE_RANK, LEAD, LAG)',
        'Indexes and Performance Optimization', 'Stored Procedures and Triggers', 'Views and Materialized Views'
    ],

    'Data Visualization': [
        'Matplotlib (Line, Bar, Scatter, Histogram, Customization)',
        'Seaborn (Heatmaps, Pairplots, Boxplots, KDE Plots)',
        'Plotly (Interactive Visualizations)', 'Tableau (Dashboards, Filters, Calculated Fields)',
        'Power BI (Data Import, DAX, Reports)', 'Best Practices for Dashboard Design', 'Storytelling with Data'
    ],

    'Big Data': [
        'Introduction to Big Data', 'Hadoop Ecosystem (HDFS, MapReduce, YARN)',
        'Spark (RDDs, DataFrames, SparkSQL, Streaming)', 'Kafka (Streaming Data Processing)',
        'Hive (Data Warehousing)', 'HBase (NoSQL Databases)', 'Data Lakes',
        'Batch vs Stream Processing', 'Real-Time Analytics'
    ],

    'MLOps & Deployment': [
        'Model Deployment Strategies (Batch, Real-time, Edge AI)', 'Flask & FastAPI for ML APIs',
        'Docker & Kubernetes for Model Deployment', 'CI/CD for ML', 'Monitoring Models in Production (Drift Detection)',
        'Cloud Platforms (AWS SageMaker, Google Vertex AI, Azure ML)', 'Feature Stores'
    ],

    'Cloud Computing': [
        'AWS (EC2, S3, Lambda, SageMaker)', 'GCP (BigQuery, AutoML, Vertex AI)', 'Azure (Blob Storage, ML Studio)',
        'Serverless Computing (Lambda, Cloud Functions)', 'Data Engineering on Cloud'
    ],

    'AI Ethics & Responsible AI': [
        'Bias in AI Models', 'Fairness & Transparency', 'Explainable AI (XAI)', 'Ethical AI Frameworks',
        'Privacy-Preserving AI (Federated Learning, Differential Privacy)', 'Adversarial Attacks in AI'
    ],

    'Other Important Topics': [
        'Web Scraping (BeautifulSoup, Selenium, Scrapy)',
        'Data Engineering (ETL, Data Pipelines, Airflow)', 
        'AutoML (H2O.ai, AutoKeras, TPOT)', 'Edge AI (Deploying AI on IoT Devices)',
        'Quantum Machine Learning (IBM Qiskit, PennyLane)'
    ]
}
    return data

# Function to create a downloadable text file
def create_text_file(content):
    buffer = BytesIO()
    buffer.write(content.encode())
    buffer.seek(0)
    return buffer

# Custom CSS
def apply_custom_css():
    st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton button {
        background-color: #2596be;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        font-weight: bold;
    }
    .stButton button:hover {
        background-color: #1a7a9c;
    }
    .stSelectbox label, .stMultiselect label {
        color: #0b0039;
        font-weight: bold;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #2596be;
    }
    .stDownloadButton button {
        background-color: #2596be;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        font-weight: bold;
    }
    .stDownloadButton button:hover {
        background-color: #1a7a9c;
    }
    div[data-baseweb="select"] {
        background-color: white;
    }
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        color: #0b0039;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 20px;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2596be;
        color: white;
    }
    .stTabs [data-baseweb="tab-panel"] {
        background-color: white;
        color: #0b0039;
        padding: 20px;
        border-radius: 0px 0px 4px 4px;
        border: 1px solid #e0e0e0;
    }
    .sidebar-header {
        background-color: rgba(11, 0, 57, 0.1);
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .main-header {
        background-color: rgba(37, 150, 190, 0.1);
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    .footer {
        background-color: rgba(11, 0, 57, 0.05);
        color: #0b0039;
        padding: 10px;
        border-radius: 5px;
        margin-top: 30px;
    }
    .chat-message {
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .user-message {
        background-color: rgba(37, 150, 190, 0.1);
        border-left: 5px solid #2596be;
    }
    .ai-message {
        background-color: white;
        border-left: 5px solid #0b0039;
    }
    .login-container {
        max-width: 500px;
        margin: 0 auto;
        padding: 20px;
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 0 10px rgba(0,0,0,0.1);
    }
    .history-item {
        padding: 10px;
        background-color: white;
        border-radius: 5px;
        margin-bottom: 10px;
        cursor: pointer;
        border-left: 3px solid #2596be;
    }
    .history-item:hover {
        background-color: rgba(37, 150, 190, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# Streamlit UI for login and registration
def show_auth_ui():
    st.markdown("<div class='main-header'><h1 style='text-align: center;'>📊 Data Science Learning Roadmap</h1><h3 style='text-align: center;'>Please login or register to continue</h3></div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h3 style='text-align: center;'>Login</h3>", unsafe_allow_html=True)
        login_username = st.text_input("Username", key="login_username")
        login_password = st.text_input("Password", type="password", key="login_password")
        
        if st.button("Login"):
            if verify_user(login_username, login_password):
                st.session_state["authenticated"] = True
                st.session_state["username"] = login_username
                
            else:
                st.error("Invalid username or password")
    
    with col2:
        st.markdown("<h3 style='text-align: center;'>Register</h3>", unsafe_allow_html=True)
        reg_username = st.text_input("Username", key="reg_username")
        reg_password = st.text_input("Password", type="password", key="reg_password")
        reg_confirm_password = st.text_input("Confirm Password", type="password")
        
        if st.button("Register"):
            if reg_password != reg_confirm_password:
                st.error("Passwords do not match")
            elif not reg_username or not reg_password:
                st.error("Username and password are required")
            else:
                if register_user(reg_username, reg_password):
                    st.success("Registration successful! You can now login.")
                else:
                    st.error("Username already exists")

# Main Streamlit UI
def main():
    st.set_page_config(
        page_title="An-Ai Proitbridge Data Science Roadmap",
        page_icon="🧑‍🏫", 
        layout="wide"
    )
    
    apply_custom_css()
    
    # Initialize session state variables
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
    
    if "username" not in st.session_state:
        st.session_state["username"] = None
    
    if "current_session_id" not in st.session_state:
        st.session_state["current_session_id"] = str(uuid.uuid4())
    
    if "history_messages" not in st.session_state:
        st.session_state["history_messages"] = []
    
    # Show authentication UI if not authenticated
    if not st.session_state["authenticated"]:
        show_auth_ui()
        return
    
    # Load roadmap data
    roadmap_data = load_roadmap_data()
    
    # Header
    st.markdown("<div class='main-header'><h1 style='text-align: center;'>🧑‍🏫 An-Ai Proitbridge Data Science Roadmap</h1><h3 style='text-align: center;'>Navigate your Data Science journey</h3></div>", unsafe_allow_html=True)
    
    # Sidebar for navigation
    st.sidebar.markdown("<div class='sidebar-header'><h2 style='color: #0b0039;'>Navigation</h2></div>", unsafe_allow_html=True)
    
    # User info in sidebar
    st.sidebar.markdown(f"<p>Logged in as: <b>{st.session_state['username']}</b></p>", unsafe_allow_html=True)
    if st.sidebar.button("Logout"):
        st.session_state["authenticated"] = False
        st.session_state["username"] = None
        st.experimental_rerun()
    
    # Previous conversations section
    st.sidebar.markdown("<h3>Previous Conversations</h3>", unsafe_allow_html=True)
    user_histories = get_session_histories(st.session_state["username"])
    
    if user_histories:
        for idx, history in enumerate(user_histories[:5]):  # Show only the 5 most recent
            history_text = f"**{history['topic']}**: {history['subtopic']}"
            if st.sidebar.button(history_text, key=f"history_{idx}"):
                # Set up the UI to load this history
                parts = history['filename'][:-5].split('_')
                topic = parts[1].replace("_", " ")
                subtopic = "_".join(parts[2:]).replace("_", " ")
                
                # Load this conversation
                history_file = os.path.join(HISTORY_DIR, history['filename'])
                with open(history_file, 'r') as f:
                    history_data = json.load(f)
                
                # Update session state
                st.session_state["selected_topic"] = topic
                st.session_state["selected_subtopic"] = subtopic
                st.session_state["history_messages"] = history_data
                st.experimental_rerun()
    else:
        st.sidebar.write("No previous conversations")
    
    # Main category selection
    main_category = "Data Science"  # Fixed as per requirement
    st.sidebar.markdown(f"<h3 style='color: #2596be;'>Category: {main_category}</h3>", unsafe_allow_html=True)
    
    # Topic selection
    if "selected_topic" not in st.session_state:
        st.session_state["selected_topic"] = list(roadmap_data.keys())[0]
    
    selected_topic = st.sidebar.selectbox(
        "Select Topic", 
        list(roadmap_data.keys()),
        index=list(roadmap_data.keys()).index(st.session_state["selected_topic"])
    )
    st.session_state["selected_topic"] = selected_topic
    
    # Subtopic selection
    if "selected_subtopic" not in st.session_state or st.session_state["selected_topic"] != selected_topic:
        st.session_state["selected_subtopic"] = roadmap_data[selected_topic][0]
    
    available_subtopics = roadmap_data[selected_topic]
    try:
        subtopic_index = available_subtopics.index(st.session_state["selected_subtopic"])
    except ValueError:
        subtopic_index = 0
    
    selected_subtopic = st.sidebar.selectbox(
        "Select Subtopic", 
        available_subtopics,
        index=subtopic_index
    )
    st.session_state["selected_subtopic"] = selected_subtopic
    
    # Main content area
    st.markdown(f"<h2 style='color: #2596be;'>{selected_topic} > {selected_subtopic}</h2>", unsafe_allow_html=True)
    
    # Feature tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📚 Explanation", "💻 Coding Questions", "🔍 Suggestions", "💬 Chat History"])
    
    # Set up the history file path for this conversation
    history_file_path = get_history_file_path(
        st.session_state["username"], 
        selected_topic, 
        selected_subtopic
    )
    
    # Initialize the model and chat history
    try:
        model = initialize_model()
        
        # Display history tab content
        with tab4:
            st.markdown("## 💬 Chat History")
            
            # Show stored messages
            for message in st.session_state["history_messages"]:
                if message.get("role") == "user":
                    st.markdown(f"<div class='chat-message user-message'><b>You:</b> {message.get('content', '')}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='chat-message ai-message'><b>AI:</b> {message.get('content', '')}</div>", unsafe_allow_html=True)
            
            # Allow clearing history
            if st.button("Clear History"):
                st.session_state["history_messages"] = []
                if os.path.exists(history_file_path):
                    os.remove(history_file_path)
                st.success("Chat history cleared!")
                st.experimental_rerun()
        
        # Define a function to save messages to history
        def save_message_to_history(role, content):
            message = {"role": role, "content": content, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            st.session_state["history_messages"].append(message)
            
            # Save to file
            with open(history_file_path, 'w') as f:
                json.dump(st.session_state["history_messages"], f)
        
        # Generate content tabs
        with tab1:
            if st.button("Generate Explanation", key="explanation_button"):
                with st.spinner("Generating explanation..."):
                    prompt = get_prompt("explanation")
                    chain = prompt | model
                    response = chain.invoke({"topic": selected_topic, "subtopic": selected_subtopic})
                    explanation = response.content if hasattr(response, "content") else response
                    
                    # Save to history
                    save_message_to_history("user", f"Generate explanation for {selected_topic} - {selected_subtopic}")
                    save_message_to_history("assistant", explanation)
                    
                    st.markdown("## 📌 Detailed Explanation")
                    st.write(explanation)
                    
                    # Create downloadable file
                    text_file = create_text_file(explanation)
                    st.download_button(
                        label="📥 Download Explanation",
                        data=text_file,
                        file_name=f"{selected_topic}_{selected_subtopic}_Explanation.txt",
                        mime="text/plain"
                    )
        
        with tab2:
            if st.button("Generate Coding Questions", key="coding_button"):
                with st.spinner("Generating coding questions..."):
                    prompt = get_prompt("coding_questions")
                    chain = prompt | model
                    response = chain.invoke({"topic": selected_topic, "subtopic": selected_subtopic})
                    coding_questions = response.content if hasattr(response, "content") else response
                    
                    # Save to history
                    save_message_to_history("user", f"Generate coding questions for {selected_topic} - {selected_subtopic}")
                    save_message_to_history("assistant", coding_questions)
                    
                    st.markdown("## 💻 Coding Questions")
                    st.write(coding_questions)
                    
                    # Create downloadable file
                    text_file = create_text_file(coding_questions)
                    st.download_button(
                        label="📥 Download Coding Questions",
                        data=text_file,
                        file_name=f"{selected_topic}_{selected_subtopic}_Coding_Questions.txt",
                        mime="text/plain"
                    )
        
        with tab3:
            if st.button("Generate Learning Suggestions", key="suggestions_button"):
                with st.spinner("Generating suggestions..."):
                    prompt = get_prompt("suggestions")
                    chain = prompt | model
                    response = chain.invoke({"topic": selected_topic, "subtopic": selected_subtopic})
                    suggestions = response.content if hasattr(response, "content") else response
                    
                    # Save to history
                    save_message_to_history("user", f"Generate learning suggestions for {selected_topic} - {selected_subtopic}")
                    save_message_to_history("assistant", suggestions)
                    
                    st.markdown("## 🔍 Learning Suggestions")
                    st.write(suggestions)
                    
                    # Create downloadable file
                    text_file = create_text_file(suggestions)
                    st.download_button(
                        label="📥 Download Suggestions",
                        data=text_file,
                        file_name=f"{selected_topic}_{selected_subtopic}_Suggestions.txt",
                        mime="text/plain"
                    )
    
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.warning("Please make sure your GROQ_API_KEY is set correctly in the .env file.")

    # Footer
    st.markdown("""
    <div class='footer' style='text-align: center;'>
        <p>© 2025 An-Ai Proitbridge Data Science Roadmap | Created with ❤️ by ProitBridge</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
