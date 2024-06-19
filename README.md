# Setup Guide

### Assumptions: We Assume that following things are already installed:
- Python3.10 & Pip
- Pipenv 

**Step 1:** Run the bash script            
```
sh setup.sh  //for Mac
./setup.sh   //for Ubunut   
```

**Step 2:** Copy actions_time_based_greetings.py from notRandom Folder in following places on your machine. 
```
/nemoguardrails/
/nemoguardrails/actions/
/nemoguardrails/actions_server/
/nemoguardrails-0.9.0.dist-info/
```

**Step 3:** Activate virtual enviornment 
```
pipenv shell
```

**Step 4:** Export OpenAI Key 
```
export OPENAI_API_KEY="Your Key"
```

***Step 5:*** Run app.py
```
streamlit run app.py
```

#### Rails Config:
All the related Rails are in topics.co
Off_topic.co contains some more usecase. You can use them also in topics.co and extend your application.