import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from dotenv import load_dotenv
import os

# Load API key
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="Agentic AI Tool Recommender", page_icon="🧠")
st.title("🧠 Agentic AI Tool Recommender")

# Load and index PDF (only once)
@st.cache_resource
def load_knowledge():
    loader = PyPDFLoader("agentic_ai_research.pdf")
    docs = loader.load_and_split()
    embedding = OpenAIEmbeddings()
    db = FAISS.from_documents(docs, embedding)
    return db.as_retriever()

retriever = load_knowledge()

# 10 questions
questions = [
    "What is your business or application domain?",
    "What kind of tasks should the agent handle?",
    "Should the agent be autonomous or human-in-the-loop?",
    "What level of real-time responsiveness is required?",
    "What deployment environment are you targeting (cloud, edge, on-prem)?",
    "Do you have preferences for open-source vs closed-source tools?",
    "What is your team’s technical expertise level?",
    "What kind of integrations does your system need?",
    "What kind of memory or context handling should the agent support?",
    "What is your budget range?"
]

# Track steps in session
if "step" not in st.session_state:
    st.session_state.step = 0
    st.session_state.answers = []
    
# Ask questions one by one
if st.session_state.step < len(questions):
    question = questions[st.session_state.step]
    answer = st.text_input(question, key=f"q{st.session_state.step}")
    if st.button("Next") and answer:
        st.session_state.answers.append(answer)
        st.session_state.step =+ 1
        
    else:
        st.success("Generating your tool recommendations....")
        profile = "\n".join(f"{questions[i]}:{a}" for i, a in enumerate(st.session_state.answers))
        
        llm = OpenAI(temperature = 0)
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
        
        result = qa_chain.run(f"Based on this profile:\n{profile}\n\nSuggest the best Agentic AI tools and explain why.")
        
        st.subheader("✅ Recommended Tools")
        st.write(result)
        st.download_button("📄 Download Report", result, file_name="agentic_ai_recommendation.txt")
        
