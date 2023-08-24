import streamlit as st
import pandas as pd
import pickle
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from nemoguardrails import LLMRails, RailsConfig

# Load .env file
from dotenv import load_dotenv
load_dotenv()

# Initialize Streamlit app
st.title("Deloitte ServiceNow Chat")
st.write("This app creates a chatbot based on a ServiceNow data")

# Load and preprocess data
article_data = pd.read_excel('CLEAN_20230809_Knowledge_Arcticles_V1.xlsx', usecols='D')

if article_data is not None:
    raw_text = ''
    for index, row in article_data.iterrows():
        text = row['Article body']
        if pd.notna(text):
            raw_text += text

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_text(raw_text)

    st.write("File is split into chunks:", len(texts))

    embeddings = OpenAIEmbeddings()

    with open("foo.pkl", 'wb') as f:
        pickle.dump(embeddings, f)

    with open("foo.pkl", 'rb') as f:
        new_docsearch = pickle.load(f)

    docsearch = FAISS.from_texts(texts, new_docsearch)

# Load NeMo Guardrails configuration from COLANG_CONFIG and YAML_CONFIG strings
COLANG_CONFIG = """
define user express greeting
  "Hello"
  "Good morning!"
  "Good evening!"
  "Hi"
  "Good day!"

define bot express greeting 
  "Hello there, how can I assist you today?"
  "Good to see you, what can I help you with?"
  "Hi! How can I serve you today?"

define flow
  user express greeting
  bot express greeting

define user express insult
  "You are stupid"

define flow
  user express insult
  bot express calmly willingness to help

define flow
  user ...
  $answer = execute generate_output($last_user_message)
  bot $answer
"""

YAML_CONFIG = """
models:
  - type: main
    engine: openai
    model: text-davinci-003
actions:
  - type: generate_output
    engine: python
"""

config = RailsConfig.from_content(COLANG_CONFIG, YAML_CONFIG)
app = LLMRails(config)

async def generate_output(**kwargs):
    user_prompt = kwargs.get('$1')
    print(user_prompt)
    docs = docsearch.similarity_search(user_prompt)

    chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")

    ai_output = chain.run(input_documents=docs, question=user_prompt)

    ai_output = ai_output.replace("\n\n--", "").replace("\n--", "").strip()
    return ai_output

# Register generate_answer as an action
app.register_action(action=generate_output, name="generate_output")


# Streamlit interface setup
st.subheader("Step 1: Ask your question")
form = st.form(key="user_settings")
with form:
  st.write("Enter a question related to ServiceNow articles")
  user_input = st.text_input("Question", key = "user_input")

  generate_button = form.form_submit_button("Submit Question")
  num_input = 1
  if generate_button:
    if user_input == "":
      st.error("Question cannot be blank")
    else:
      my_bar = st.progress(0.05)
      st.subheader("Answer:")

      #for i in range(num_input):
      st.markdown("""---""")
      history = [
              {"role": "user", "content": user_input}
      ]
      result = app.generate(messages=history)
      st.write(result["content"])
      my_bar.progress(1)

st.write( '')