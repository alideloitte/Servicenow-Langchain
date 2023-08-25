from flask import Flask, request, jsonify
import pandas as pd
import pickle
import os
import logging
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from nemoguardrails import LLMRails, RailsConfig
from dotenv import load_dotenv

flask_app = Flask(__name__)

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
    return texts

#function to create and load embeddings
def create_and_load_embeddings(embeddings, file_path="embeddings.pkl"):
    
    # Check if the embeddings have already been computed and saved.
    if not os.path.exists(file_path):
        # Compute the embeddings and save them to a file.
        with open(file_path, 'wb') as f:
            pickle.dump(embeddings, f)
    
    # Load the embeddings
    with open(file_path, 'rb') as f:
        return pickle.load(f)

#function to create document search
def create_document_search(texts, embeddings):
    return FAISS.from_texts(texts, embeddings)

#Function to generate output
async def generate_output(**kwargs):
    user_prompt = kwargs.get('$1')
    docs = docsearch.similarity_search(user_prompt)

    chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")

    ai_output = chain.run(input_documents=docs, question=user_prompt)

    ai_output = ai_output.replace("\n\n--", "").replace("\n--", "").strip()
    return ai_output

# Load and preprocess data
raw_text = load_and_preprocessing('data/CLEAN_20230809_Knowledge_Arcticles_V1.xlsx')

texts = split_text_into_chunks(raw_text)

embeddings = OpenAIEmbeddings()

new_docsearch = create_and_load_embeddings(embeddings, "embeddings.pkl")

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

# Flask API endpoint
@flask_app.route('/api', methods=['POST'])
def api():
  try:
      #check if request content type is application/json
      if not request.is_json:
          logging.info('Invalid request received')
          return jsonify({"error": "Invalid: Content type is not json"}), 415 
      data = request.get_json()  # get data passed to API
      
      # check if user_input key exists in request data
      if 'user_input' not in data:
           return jsonify({"error": "Invalid: No user_input key in the request"}), 400          
      user_input = data['user_input']
  
      if user_input == "":
           return jsonify({"error": "Question cannot be blank"}), 400
      
      # Check if user_input is of type string
      if not isinstance(user_input, str):
           return jsonify({"error": "Invalid: user_input must be a string"}), 400
    
      history = [
              {"role": "user", "content": user_input}
      ]
      result = app.generate(messages=history)
      response =  {"content": result["content"]}

      return jsonify(response)
  except Exception as e:
      #log the error here
      logging.error(str(e))
      # Return a generic server error to the user
      return jsonify({"error": "A server error occurred"}), 500

if __name__ == '__main__':
  flask_app.run(host='0.0.0.0', port=105, debug=True) #debug=False for production