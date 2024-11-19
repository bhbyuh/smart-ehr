import streamlit as st
from streamlit_chat import message
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore as lang_pinecone
from langchain_core.prompts import ChatPromptTemplate
from operator import itemgetter
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ['PINECONE_API_KEY'] = os.getenv('PINECONE_API_KEY')

# Initialize chat model and embeddings
chat_llm = ChatOpenAI(model='gpt-4o-mini', openai_api_key=os.getenv("OPENAI_API_KEY"), temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Function to generate response
def _combine_documents3(docs, document_separator="\n\n"):
    try:
        if docs:
            combined_list = [
                f"content:{doc.page_content} " for index, doc in enumerate(docs)
            ]
            combined = document_separator.join(combined_list)
        else:
            combined = ""  # No documents found
        return combined
    except Exception as ex:
        raise

def get_response(user_query, namespace):
    prompt_str = """
        You are an AI assistant. According to user_query, You must generate a concise and too the point response.
        Guidelines:
        -If user_query is related to greetings than greet properly.

        user_query: {query1}
        Context: {context}
        """
    num_chunks = 5
    vectordb = lang_pinecone.from_existing_index(index_name="smart-ehr", embedding=embeddings, namespace=namespace)
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": num_chunks})
    _prompt = ChatPromptTemplate.from_template(prompt_str)
    query_fetcher = itemgetter("query1")
    setup = {"query1": query_fetcher, "context": query_fetcher | retriever | _combine_documents3}
    _chain = (setup | _prompt | chat_llm)
    response = _chain.invoke({"query1": user_query}).content
    return response

# Styling for better UI
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f8f9fa;
        font-family: 'Arial', sans-serif;
    }
    h1 {
        color: #3d5a80;
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 20px;
    }
    .sidebar .block-container {
        background-color: #e0fbfc;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
    }
    .css-1aumxhk {
        color: #293241;
    }
    .stTextInput>div>div {
        border: 1px solid #3d5a80 !important;
        border-radius: 8px !important;
    }
    .stButton>button {
        background-color: #3d5a80;
        color: #ffffff;
        font-size: 16px;
        font-weight: 600;
        border-radius: 8px;
        padding: 10px 20px;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #293241;
    }
    .st-alert-warning {
        color: #856404;
        background-color: #fff3cd;
        border-color: #ffeeba;
        border-radius: 8px;
        padding: 15px;
        font-size: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# App title
st.markdown("<h1>Smart-EHR</h1>", unsafe_allow_html=True)

# Sidebar for Patient ID
with st.sidebar:
    st.header("Select Patient ID")
    patient_id = st.selectbox("Patient ID:", options=["2416237", "3365563"])

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# Main Chat Interface
if not patient_id:
    st.warning("Please select a Patient ID in the sidebar to proceed.")
else:
    with st.form(key='chat_form', clear_on_submit=True):
        st.markdown("<h2 style='color: #3d5a80;'>Chat Interface</h2>", unsafe_allow_html=True)
        user_input = st.text_input("Ask a question about the patient:", "")
        submit_button = st.form_submit_button(label='Submit')

    if submit_button and user_input:
        st.session_state['messages'].append({"user": user_input})
        response = get_response(user_input, patient_id)
        st.session_state['messages'].append({"bot": response})

    if st.session_state['messages']:
        st.markdown("<h2 style='color: #3d5a80;'>Chat History</h2>", unsafe_allow_html=True)
        for i, msg in enumerate(st.session_state['messages']):
            if "user" in msg:
                message(msg["user"], is_user=True, key=f"user_{i}")
            else:
                message(msg["bot"], is_user=False, key=f"bot_{i}")
