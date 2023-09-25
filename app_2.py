import streamlit as st
import time
import sys, fitz
import pdfplumber
import base64
from pathlib import Path
from PIL import Image
import re
import replicate
import pandas as pd
import ast
import gzip
import os
from torch import cuda, bfloat16
import transformers
import torch
from transformers import StoppingCriteria, StoppingCriteriaList
from langchain.llms import HuggingFacePipeline
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from transformers import AutoTokenizer, pipeline, logging
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from trubrics.integrations.streamlit import FeedbackCollector
from trubrics import Trubrics


st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align: center; color: black;'> ContentSculpt </h1>", unsafe_allow_html=True)
st.markdown("<h6 style='text-align: center; color: black;'> Intelligent Content Drafing Suite </h6>", unsafe_allow_html=True)
#uploaded_file = st.sidebar.file_uploader("Upload a PDF File",type= 'pdf' , key="file")
#if uploaded_file is not None:

def page6():
  st.markdown("<h3 style='text-align: center; color: grey;'> Instruction Based Promotional Content Generation </h3>", unsafe_allow_html=True)
  #######
  # Get the input text from the user
  st.title("Marketing Content  Generator")
  option1 = st.sidebar.selectbox(
  'Product',
  ('Phesgo', 'Tecentriq'))
  option2 = st.sidebar.selectbox(
  'Target Audience',
  ('HCP', 'Patients', 'Patients and their Families'))

  option3 = st.sidebar.selectbox(
  'Tone of Generation',
  ('Professional','Empathetic', 'Informative', 'Patient-centered','Ethical', 'Engaging','Trustworthy', 'Compassionate and Reassuring'
  ))

  option4 = st.sidebar.selectbox(
  'Content Type',
  ('None','Newsletter','Email', 'Executive', 'Regular Content','Blog Post' 
      ))
  option5 = st.sidebar.selectbox(
  'Objective',
  ('Increase User Engagement','Generate Interest', 'Share Product Update', 'Increase Product Adoption', ' Provide Hope and Information'
      ))

  option6 = st.sidebar.selectbox(
  'Output Language',
  ('English','French', 'Spanish', 'German', 
      'Italian'))
  
  option8 = st.sidebar.selectbox(
  'Target Audience Expectation',
  ('Alternative Treatment', 'Ease of Access', 'Higher Safety', 'Higher Efficacy', 'Quality of life', 'Lower Price'))
  
  option7 = st.text_input('Input your prompt here',"Write an executive short email for internal purposes based on document summary?")
  
  default_prompt = ["Create persuasive marketing content in " + option6 + " for " + option2+ ", emphasizing the " +option3+ " tone. Craft a "+ option4+ " that educates them about " + option1 +" role in cancer treatment and its potential benefits. The objective is to " + option5 + " to those seeking "+ option8+" options. " + option7]
  #prompt = st.text_input('Input your prompt here')
  prompt = st.write(default_prompt[0])
  #Create a Side bar
  with st.sidebar:
      st.title("🦙💬 Interactive Content Generation Assistant")
      st.header("Settings")
  
      add_replicate_api=st.text_input('Enter the Replicate API token', type='password')
      if not (add_replicate_api.startswith('r8_') and len(add_replicate_api)==40):
          st.warning('Please enter your credentials', icon='⚠️')
      else:
          st.success('Proceed to entering your prompt message!', icon='👉')
  
      st.subheader("Models and Parameters")
  
      select_model=st.selectbox("Choose a Llama 2 Model", ['Llama 2 7b', 'Llama 2 13b', 'Llama 2 70b'], key='select_model')
      if select_model=='Llama 2 7b':
          llm = 'a16z-infra/llama7b-v2-chat:4f0a4744c7295c024a1de15e1a63c880d3da035fa1f49bfd344fe076074c8eea'
      elif select_model=='Llama 2 13b':
          llm = 'a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5'
      else:
          llm = 'replicate/llama70b-v2-chat:e951f18578850b652510200860fc4ea62b3b16fac280f83ff32282f87bbd2e48'
  
      temperature=st.slider('temperature', min_value=0.01, max_value=5.0, value=0.1, step=0.01)
      top_p=st.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
      max_length=st.slider('max_length', min_value=64, max_value=4096, value=512, step=8)
  
      #st.markdown('I make content on AI on regular basis do check my Youtube channel [link](https://www.youtube.com/@muhammadmoinfaisal/videos)')

  os.environ['REPLICATE_API_TOKEN']=add_replicate_api
  #Store the LLM Generated Reponese
  
  if "messages" not in st.session_state.keys():
      st.session_state.messages=[{"role": "assistant", "content":"How may I assist you today?"}]
  
  # Diplay the chat messages
  
  for message in st.session_state.messages:
      with st.chat_message(message["role"]):
          st.write(message["content"])
  
  
  # Clear the Chat Messages
  def clear_chat_history():
      st.session_state.messages=[{"role":"assistant", "content": "How may I assist you today"}]
  
  st.sidebar.button('Clear Chat History', on_click=clear_chat_history)
  
  # Create a Function to generate the Llama 2 Response
  def generate_llama2_response(prompt_input):
      default_system_prompt="You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'."
      for data in st.session_state.messages:
          print("Data:", data)
          if data["role"]=="user":
              default_system_prompt+="User: " + data["content"] + "\n\n"
          else:
              default_system_prompt+="Assistant" + data["content"] + "\n\n"
      output=replicate.run(llm, input={"prompt": f"{default_system_prompt} {prompt_input} Assistant: ",
                                       "temperature": temperature, "top_p":top_p, "max_length": max_length, "repititon_penalty":1.15})
  
      return output
  
  
  #User -Provided Prompt
  
  if prompt := st.chat_input(disabled=not add_replicate_api):
      st.session_state.messages.append({"role": "user", "content":prompt})
      with st.chat_message("user"):
          st.write(prompt)
  
  # Generate a New Response if the last message is not from the asssistant
  
  if st.session_state.messages[-1]["role"] != "assistant":
      with st.chat_message("assistant"):
          dx = pd.read_csv('generate_outcome - Sheet1.csv')
          with st.spinner("Thinking..."):
              tab1, tab2, tab3 = st.tabs(["Generated Outcome 1","Generated Outcome 2","Generated Outcome 3"])
              with tab1:
                if "feedback_key" not in st.session_state:
                  st.session_state.feedback_key = 0
                response1= tab1.write(dx['generation_1'][0])
                collector = FeedbackCollector(
                  project="llm_gen",
                  email='smnitrkl50@gmail.com',
                  password='Ram@2107',
                )
        
                collector.st_feedback(
                  component="default1",
                  feedback_type="faces",
                  model="llama2_13b",
                  prompt_id=None,  # see prompts to log prompts and model generations
                  open_feedback_label='Provide Feedback'
                )
              with tab2:
                if "feedback_key" not in st.session_state:
                  st.session_state.feedback_key = 0
                response2=tab2.write(dx['generation_2'][0])
                collector = FeedbackCollector(
                  project="llm_gen",
                  email='smnitrkl50@gmail.com',
                  password='Ram@2107',
                )
        
                collector.st_feedback(
                  component="default1",
                  feedback_type="faces",
                  model="llama2_13b",
                  prompt_id=None,  # see prompts to log prompts and model generations
                  open_feedback_label='Provide Feedback'
                )
              with tab3:
                if "feedback_key" not in st.session_state:
                  st.session_state.feedback_key = 0
                response3= tab3.write(dx['generation_3'][0])
                collector = FeedbackCollector(
                  project="llm_gen",
                  email='smnitrkl50@gmail.com',
                  password='Ram@2107',
                )
        
                collector.st_feedback(
                  component="default1",
                  feedback_type="faces",
                  model="llama2_13b",
                  prompt_id=None,  # see prompts to log prompts and model generations
                  open_feedback_label='Provide Feedback'
                )
              #response=generate_llama2_response(prompt)
              placeholder=st.empty()
              full_response=''
              response1 = ''
              response2 = ''
              response3 = ''
              for item in (response1 or response2 or response3):
                  full_response+=item
                  placeholder.markdown(full_response)
              placeholder.markdown(full_response)
  
      message= {"role":"assistant", "content":full_response}

      # collector = FeedbackCollector(
      #     project="llm_gen",
      #     email='smnitrkl50@gmail.com',
      #     password='Ram@2107',
      # )

      # collector.st_feedback(
      #     component="default1",
      #     feedback_type="faces",
      #     model="llama2_13b",
      #     prompt_id=None,  # see prompts to log prompts and model generations
      #     open_feedback_label='Provide Feedback'
      # )
      

      st.session_state.messages.append(message)

      
      
      # if user_feedback:
      #     st.write("#### Raw feedback saved to Trubrics:")
      #     st.write(user_feedback)
 
  # input_text = st.text_input("Write an executive short email for internal purposes based on document summary? ")
  
  # dg_g = pd.read_csv(os.path.join(os.getcwd(),'Demo_lab_1 - Demo_lab.csv'))
  # if uploaded_file.name == 'Residual Disease Management In HER2+ve Early Breast Cancer Setting - Case Discussion.pdf':
  #     output_text = st.write(dg_g['Email Based on Summary'][0])
      
  # elif uploaded_file.name == 'test_breast_file.pdf':
  #     output_text = st.write(dg_g['Email Based on Summary'][1])
     
  # elif uploaded_file.name == 'APAC DAN Lung PPoC Insight WP (last updated 2023.08.08).pdf':
  #     output_text = st.write(dg_g['Email Based on Summary'][2])
      
  # begin initializing HF items, you need an access token
  # import requests

  # API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-2-13b-chat-hf"
  # headers = {"Authorization": "Bearer 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"}

  # def query(payload):
  #     response = requests.post(API_URL, headers=headers, json=payload)
  #     return response.json()
      
  # output = query({
  #     "inputs": "Can you please let us know more details about your ",
  # })

  # # Get the input text from the user
  # st.title("Marketing Compaign Generator")
  # input_text = st.text_input("Enter the prompt/instruction for the newsletter:")
  # prompt = input_text #"Generate a newsletter that captures the logical dependencies present in the provided context. Identify and highlight the relationships, cause-and-effect connections, and sequential progressions among different pieces of information. Ensure that the generated newsletter effectively communicates the flow of events, insights, or concepts by showcasing their interconnectedness and logical coherence if any from text with formal word included "  + str('""" ') + str(text) + str(' """')

  # prompt_template = f'''

  # USER: {prompt}

  # ASSISTANT: Write an impressive newsletter with Retrospectives and Prospectives finds for mentioned product. Please say Not Applicable if you are not confident or honest about outcome:
  # '''
  
  # output_text= query({
  #     "inputs": prompt_template,
  # })

  # Get the input text from the user
  # st.title("Generated Outcome")
  # st.write(output_text)
  
  # st.title("Visuals Generation for Marketing Compaign")

  # # Get the input text from the user
  # input_text = st.text_input("Enter the prompt/instruction for the Visuals:" )





page_names_to_funcs = {
  #"Intelligent Data Parsing": main_page,
  #"Contextual Tags (Based on Hypothesis)": page3,
  #"Full Document Summary (Image and Text)": page5,
  "Content Generation" : page6,
  #"Non - Contextual Tags, Iterate over pages": page2,
  #"Search across Document": page4,
  #"Query Based Evidence Generation": page8,
}

selected_page = st.sidebar.selectbox("# Analysis Selection", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()
