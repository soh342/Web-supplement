import streamlit as st
from transformers import pipeline, AutoTokenizer
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings  # Hugging Face embeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Summarization pipeline using Hugging Face
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# GPT-2 language model pipeline
def get_llm():
    try:
        return pipeline("text-generation", model="gpt2", max_new_tokens=50)  # Limit response length
    except Exception as e:
        st.error(f"Error loading language model: {e}")
        return None

# Load and split website content into chunks
def get_vectorstore_from_url(url):
    try:
        loader = WebBaseLoader(url)
        document = loader.load()

        # Split content into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter()
        document_chunks = text_splitter.split_documents(document)

        # Use Hugging Face embeddings to create a FAISS vector store
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(document_chunks, embeddings)
        return vector_store
    except Exception as e:
        st.error(f"Error loading URL: {e}")
        return None

# Clean up irrelevant content from documents
def clean_text(text):
    lines = text.split('\n')
    return "\n".join([line.strip() for line in lines if line.strip() and not line.startswith(("Related", "Subscribe", "Contact"))])

# Summarize long contexts to reduce token size
def summarize_context(context):
    try:
        summary = summarizer(context, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
        return summary
    except Exception as e:
        st.error(f"Error summarizing context: {e}")
        return context  # Fallback to original context if summarization fails

# Truncate prompt if it exceeds the token limit
def truncate_text(text, tokenizer, max_length):
    tokens = tokenizer.encode(text, truncation=True, max_length=max_length)
    return tokenizer.decode(tokens)

# Generate a response from the language model
def get_response(user_input):
    try:
        llm = get_llm()
        if not llm:
            return "Failed to load language model."

        # Retrieve relevant documents
        retriever = st.session_state.vector_store.as_retriever()
        docs = retriever.get_relevant_documents(user_input)[:3]  # Limit to top 3 docs

        # Clean and summarize the context
        context = "\n".join([clean_text(doc.page_content) for doc in docs])
        summarized_context = summarize_context(context)

        # Construct a concise prompt
        prompt = (
            f"Context:\n{summarized_context}\n\n"
            f"Q: {user_input}\n"
            f"A:"
        )

        # Tokenize and truncate the prompt to fit the model's limits
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        max_length = tokenizer.model_max_length
        truncated_prompt = truncate_text(prompt, tokenizer, max_length)

        # Generate the response
        response = llm(truncated_prompt)

        if not response or len(response) == 0:
            return "Error: No response generated."

        # Extract only the generated part of the response
        generated_text = response[0]["generated_text"].strip()
        answer = generated_text[len(truncated_prompt):].strip()

        if not answer:
            return "Error: Empty response."

        return answer
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return "Error generating response."

# Streamlit app setup
st.set_page_config(page_title="Chat with Websites", page_icon="🤖")
st.title("Chat with Websites")

# Sidebar settings
with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Website URL")

# Handle user input and cache vector store
if website_url:
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vectorstore_from_url(website_url)

    user_query = st.chat_input("Type your message here...")
    if user_query:
        response = get_response(user_query)
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        st.session_state.chat_history.append({"role": "assistant", "content": response})

# Initialize chat history if not present
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [{"role": "assistant", "content": "Hello, I am a bot. How can I help you?"}]

# Display chat history
for message in st.session_state.chat_history:
    if message["role"] == "assistant":
        with st.chat_message("AI"):
            st.write(message["content"])
    elif message["role"] == "user":
        with st.chat_message("Human"):
            st.write(message["content"])
