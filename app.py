import streamlit as st
from model import QuestAnsweringDistilBERT, CustomDataset
import torch
import transformers
import pandas as pd
import torch.utils.data as data_utils
from transformers import DistilBertConfig, DistilBertTokenizer, DistilBertModel

st.markdown("## 🤖Check the fact!🤖")
st.markdown("-----------------------------------------------------------")
st.markdown("### Challenge the bot awaraness with contextual statements!", unsafe_allow_html=True)
st.markdown("Provide a passage, suggest a statement (with verdict) about it and wait the bot's opinion!", unsafe_allow_html=True)


@st.cache_resource
def load_utils():
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
    model = QuestAnsweringDistilBERT(bert, DistilBertConfig(), 'part', n_layers=1,
                                 hidden_dim=384, dim=768).to('cpu')
    model.load_state_dict(torch.load("deploy_model_weights.pt", map_location='cpu'))
    model.eval()
    return model, tokenizer

model, tokenizer = load_utils()

@st.cache_data
def process(passage, question, answer):
    df = pd.DataFrame({'passage': [passage.lower()],
		       'question': [question.lower()],
		       'answer': [1.0 if answer.lower() == "true" else 0.0]})
    dataset = CustomDataset(df, tokenizer)
    data = data_utils.DataLoader(dataset=dataset, batch_size=1)
    return next(iter(data))[0]

passage = st.text_input("Enter an interesting passage", value="A long long time ago ...")

question = st.text_input("Enter a challenging statement about passage", value="Was Darth Vader the father?")

answer = st.text_input("Enter a 'true' or 'false' verdict about the statement", value="true")

submitted = st.button("Compare Results!")

if submitted:
    
    ok = 1
    
    if answer.lower() not in ["true", "false"]:
        st.error("Wrong answers. Only 'true' and 'false' are acceptable!", icon="🚨")
        ok = 0


    if len(question) == 0:
        st.error('We need a question!', icon="🔥") 
        ok = 0


    if len(passage) == 0:
        st.error('We need something exciting to investigate!', icon="🤖") 
        ok = 0
    
    if ok == 1:

        batch = process(passage, question, answer)

        model_answer = "true" if torch.sigmoid(model(batch)).item() > 0.5 else "false"

        if model_answer == answer:
            st.markdown("The model thinks that the answer is '{}'. The model guessed RIGHT!".format(model_answer))
        else:
            st.markdown("The model thinks that the answer is '{}'. The model is WRONG. We are sorry for the mistake!".format(model_answer))