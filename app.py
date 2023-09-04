from argparse import Action
import streamlit as st
import pandas as pd
import pickle
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from nemoguardrails import LLMRails, RailsConfig
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Load and preprocess data
def load_and_preprocessing(file_path):
   article_data = pd.read_excel(file_path, usecols='D')
   raw_text = ''
   for _, row in article_data.iterrows():
        text = row['Article body']
        if pd.notna(text):
            raw_text += text

   return raw_text

#function to split text into chunks
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

#function to create and load embeddings
def create_and_load_embeddings(embeddings, file_path="foo.pkl"):
    home_dir = os.path.expanduser("~")
    full_path = os.path.join(home_dir, file_path)

    with open(file_path, 'wb') as f:
        pickle.dump(embeddings, f)
    with open(file_path, 'rb') as f:
        return pickle.load(f)

#function to create document search
def create_document_search(texts, embeddings):
    return FAISS.from_texts(texts, embeddings)

#Function to generate output
async def generate_output(**kwargs):
    user_prompt = kwargs.get('$1')
    print(user_prompt)
    docs = docsearch.similarity_search(user_prompt)

    chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")

    ai_output = chain.run(input_documents=docs, question=user_prompt)

    ai_output = ai_output.replace("\n\n--", "").replace("\n--", "").strip()
    return ai_output

# Initialize Streamlit app
st.title("Deloitte ServiceNow Chat")
st.write("This app creates a chatbot based on a ServiceNow data")


# Load and preprocess data
raw_text = load_and_preprocessing('data/CLEAN_20230809_Knowledge_Arcticles_V1.xlsx')

texts = split_text_into_chunks(raw_text)

embeddings = OpenAIEmbeddings()

new_docsearch = create_and_load_embeddings(embeddings, "foo.pkl")

docsearch = create_document_search(texts, new_docsearch)

# Load COLANG_CONFIG from a file
with open('rails-Config/topics.co', 'r') as file:
    COLANG_CONFIG = file.read()
# Load YAML_CONFIG from a file
with open('rails-Config/config.yml', 'r') as file:
    YAML_CONFIG = file.read()



config = RailsConfig.from_content(COLANG_CONFIG, YAML_CONFIG)
app = LLMRails(config)

# Register generate_answer as an action
app.register_action(action=generate_output, name="generate_output")


# Streamlit interface setup
st.subheader("Step 1: Ask your question")
form = st.form(key="user_settings")

with form:
  st.write("Enter a question related to ServiceNow articles")
  user_input = st.text_input("Question", key = "user_input")

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
      result = app.generate(messages=history)
      st.write(result["content"])
      my_bar.progress(1)
