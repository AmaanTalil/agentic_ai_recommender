import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
#from langchain_community.llms import OpenAI
from langchain.chat_models import ChatOpenAI
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
    pages = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    docs = text_splitter.split_documents(pages)
    
    embedding = OpenAIEmbeddings()
    db = FAISS.from_documents(docs, embedding)
    return db.as_retriever()

retriever = load_knowledge()

# Track steps in session
if "step" not in st.session_state:
    st.session_state.step = 0
    st.session_state.answers = []
    
if "generated_questions" not in st.session_state:
    st.session_state.generated_questions = []

if "dynamic_mode" not in st.session_state:
    st.session_state.dynamic_mode = False

# 2 questions
static_questions = [
    "What is your business or application domain?",
    "What kind of tasks should the agent handle?"
]
    
# Ask questions one by one
total_static = len(static_questions)

# Determine which question to ask
if st.session_state.step < total_static:
    question = static_questions[st.session_state.step]
else:
    # If dynamic questions aren't generated yet, do it now
    if not st.session_state.dynamic_mode:
        with st.spinner("Thinking of smart follow-up questions..."):
            static_answers = [
                st.session_state.get(f"q{i}", "") for i in range(total_static)
            ]
            base_context = "\n".join(
                f"{static_questions[i]}: {static_answers[i]}"
                for i in range(total_static)
            )

            followup_prompt = f"""
Given this context about the user’s goal:

{base_context}

Generate 3 to 6 precise, concise follow-up questions that would help narrow down the best AI tools or agent frameworks to use. Number the questions as a plain list (1., 2., ...).
"""

            llm = ChatOpenAI(model="gpt-4o", temperature=0)
            response = llm.invoke(followup_prompt)

            lines = response.content.splitlines()
            st.session_state.generated_questions = [
                line.split(". ", 1)[1] if ". " in line else line for line in lines if line.strip()
            ]
            st.session_state.dynamic_mode = True

    # Get next generated question
    dynamic_index = st.session_state.step - total_static
    if dynamic_index < len(st.session_state.generated_questions):
        question = st.session_state.generated_questions[dynamic_index]
    else:
        question = None

# Display question and handle form
if question:
    st.write(f"### Question {st.session_state.step + 1}: {question}")
    key = f"q{st.session_state.step}"

    # Create a form to combine input and button together
    with st.form(key=f"form_{st.session_state.step}"):
        answer = st.text_area("Your Answer:", key=key)
        submitted = st.form_submit_button("➡ Next")
        
        if submitted:
            if answer and answer.strip():
                st.session_state.answers.append(answer.strip())
                st.session_state.step += 1
            else:
                st.warning("Please enter an answer before clicking Next.")

else:
    # No more questions left — show recommendation
    st.success("Generating your tool recommendations...")

    all_questions = static_questions + st.session_state.generated_questions
    profile = "\n".join(
        f"{all_questions[i]}: {st.session_state.answers[i]}"
        for i in range(len(st.session_state.answers))
    )

    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    st.session_state.recommendation_result = qa_chain.run(query = f"""
You are a professional agentic AI systems consultant.

Your only source of knowledge is the following research document on agentic AI tools and frameworks, which has already been provided to you in context:

- agentic_ai_research.pdf

Below is a detailed description of a user’s goals and requirements. Based only on this document and nothing else, generate a clear, beginner-friendly recommendation that maps the user’s needs to specific agentic AI tools, frameworks, memory components, LLM models, orchestration layers, or planning modules.

Also describe the directional **workflow architecture** — explain how these tools will be connected or used in sequence to build a functioning agent.

The explanation should:
- Be written so that both technical and non-technical readers can understand it
- Highlight **why** each tool fits their context
- Be structured with clear headings like: `LLM Model`, `Memory Tool`, `Planning Tool`, `Workflow Flow`, etc.
- Avoid external information not present in the provided document

Here is the user’s profile context:

{profile}
""")

    st.subheader("✅ Recommended Tools")
    st.write(st.session_state.recommendation_result)
    st.download_button("📄 Download Report", st.session_state.recommendation_result, file_name="agentic_ai_recommendation.txt")

    # 🔁 Start Over button
    if st.button("🔄 Start Over"):
        st.session_state.step = 0
        st.session_state.answers = []
        st.session_state.generated_questions = []
        st.session_state.dynamic_mode = False
        for key in list(st.session_state.keys()):
            if key.startswith("q"):
                del st.session_state[key]
        st.rerun()
        
