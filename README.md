# Agentic_AI_Recommendation Tool

A simple and interactive Streamlit application that helps recommend the right AI tools based on user requirements. Powered by LangChain, OpenAI, and FAISS for intelligent PDF-based search and retrieval.

A tool that will help you shortlist the agentic tools based on your business requirements so that you can build your own Agentic AI.
It recommends tools from a detailed research done on 150+ agentic ai tools.

User Input → Streamlit → Session State → LLM → Document Retriever → Tool Recommendations → Summary Report

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # On Linux/Mac
.\venv\Scripts\activate         # On Windows (use '/' instead of '\'if using bash in VS Code) 

# Install dependencies
pip install -r requirements.txt

# Create a .env file
OPENAI_API_KEY=your_openai_key_here

# Run the app
streamlit run app.py
