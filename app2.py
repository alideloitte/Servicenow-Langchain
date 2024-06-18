import os
import pickle
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain_community.vectorstores import FAISS
from nemoguardrails import LLMRails, RailsConfig
import asyncio
import nest_asyncio
import pypdf


nest_asyncio.apply()

# Load .env file
load_dotenv()

# Load and preprocess data
#def load_and_preprocess(file_path):
#    article_data = pd.read_excel(file_path, usecols='D')
#    raw_text = ''
#    for _, row in article_data.iterrows():
#        text = row['Article body']
#        if pd.notna(text):
#            raw_text += text
#    return raw_text

#def load_and_preprocess(file_path):
#    with open(file_path, 'r', encoding='utf-8') as file:
#        raw_text = file.read()
#    return raw_text

def load_and_preprocess(file_path):
    raw_text = ''
    with open(file_path, 'rb') as pdf_file:
        pdf_reader = pypdf.PdfReader(pdf_file)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text()
            raw_text += page_text
    return raw_text

# Function to split text into chunks
def split_text_into_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_text(raw_text)
    st.write("File is split into chunks:", len(texts))
    return texts

# Function to create and save embeddings data
def create_and_save_embeddings_data(texts, embeddings_model, file_path="embeddings.pkl"):
    embeddings = embeddings_model.embed_documents(texts)
    with open(file_path, 'wb') as f:
        pickle.dump(embeddings, f)
    return embeddings

def load_embeddings_data(file_path="embeddings.pkl"):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

# Function to create document search
def create_document_search(texts, embeddings_model):
    return FAISS.from_texts(texts, embeddings_model)

# Function to generate output
async def generate_output(**kwargs):
    user_prompt = kwargs.get('$1')
    print(user_prompt)
    docs = docsearch.similarity_search(user_prompt)
    chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")
    ai_output = chain.run(input_documents=docs, question=user_prompt)
    ai_output = ai_output.replace("\n\n--", "").replace("\n--", "").strip()
    return ai_output

# Ensure there is an event loop
def get_or_create_event_loop():
    try:
        return asyncio.get_event_loop()
    except RuntimeError as ex:
        if "There is no current event loop in thread" in str(ex):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop

# Initialize Streamlit app
st.title("Deloitte Talanx Workshop - Chat")
st.write("This app creates a chatbot based on HDI Insurance data")

# Load and preprocess data
raw_text = load_and_preprocess('data/7001000038.pdf')

texts = split_text_into_chunks(raw_text)

embeddings_model = OpenAIEmbeddings()

# Ensure embeddings are created and saved properly
create_and_save_embeddings_data(texts, embeddings_model, "embeddings.pkl")

# Create document search with correct embeddings model instance
docsearch = create_document_search(texts, embeddings_model)

# Load COLANG_CONFIG from a file
#with open('rails-Config/off_topic.co', 'r') as file:
#    COLANG_CONFIG = file.read()

#with open('rails-Config/off_topic.co', 'r') as file:
# Load COLANG_CONFIG from files
#with open('/home/hassan/Servicenow-Langchain/rails-Config/topics.co', 'r') as file:
#    TOPICS_CONFIG = file.read()

#with open('/home/hassan/Servicenow-Langchain/rails-Config/off_topic.co', 'r') as file:
#    OFF_TOPIC_CONFIG = file.read()

#COLANG_CONFIG = TOPICS_CONFIG + "\n" + OFF_TOPIC_CONFIG

# Load YAML_CONFIG from a file
#with open('rails-Config/config.yml', 'r') as file:
#    YAML_CONFIG = file.read()

# Combine configurations
#config = RailsConfig.from_content(COLANG_CONFIG, YAML_CONFIG)

with open('rails-Config/topics.co', 'r') as file:
    COLANG_CONFIG = file.read()


#Load YAML_CONFIG from a file
with open('rails-Config/config.yml', 'r') as file:
    YAML_CONFIG = file.read()

config = RailsConfig.from_content(COLANG_CONFIG, YAML_CONFIG)

get_or_create_event_loop()  # Ensure there is an event loop
app = LLMRails(config)

# Register generate_answer as an action
app.register_action(action=generate_output, name="generate_output")

# Streamlit interface setup
st.subheader("Step 1: Ask your question")
form = st.form(key="user_settings")

with form:
    st.write("Enter a question related to your Insurance coverage")
    user_input = st.text_input("Question", key="user_input")

    generate_button = form.form_submit_button("Submit Question")
    if generate_button:
        if user_input == "":
            st.error("Question cannot be blank")
        else:
            my_bar = st.progress(0.05)
            st.subheader("Answer:")
            st.markdown("""---""")
            history = [
                {"role": "user", "content": user_input}
            ]

            # Use generate_async to handle async execution
            async def get_response():
                return await app.generate_async(messages=history)

            result = asyncio.run(get_response())

            st.write(result["content"])
            my_bar.progress(1)
