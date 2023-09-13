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
# from torch import cuda, bfloat16
# import transformers
# import torch
# from transformers import StoppingCriteria, StoppingCriteriaList
# from langchain.llms import HuggingFacePipeline
# from langchain.document_loaders import PyPDFDirectoryLoader
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import FAISS
# from langchain.chains import ConversationalRetrievalChain

# import os 
# from langchain.chains import RetrievalQA
# from langchain.llms import OpenAI
# from langchain.document_loaders import TextLoader
# from langchain.document_loaders import PyPDFLoader
# from langchain.indexes import VectorstoreIndexCreator
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.vectorstores import Chroma
# import tempfile
# import numpy as np
# import torch
# import os
# import pandas as pd
# from sentence_transformers import SentenceTransformer
# import faiss
# import time
# import json
# from sentence_transformers import SentenceTransformer, CrossEncoder, util
# import gzip
# import os
# import torch
# from rank_bm25 import BM25Okapi
# from sklearn.feature_extraction import _stop_words
# import string
# from tqdm.autonotebook import tqdm
# import numpy as np

# import streamlit as st
from helpers import (
    upload_pdf_file, 
    create_space, 
    image_extraction_component, 
    get_pdf_from_link, 
    return_pdf_data, 
    text_summary_component,
    set_session_state_key,
    sidebar_widget,
    get_text_data_from_pdf,
    load_state,
    load_pdf_report_summary
)

from config import PDF_DATA_KEY, TEXT_DATA_KEY

st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align: center; color: black;'> ContentSculpt </h1>", unsafe_allow_html=True)
st.markdown("<h6 style='text-align: center; color: black;'> Intelligent Content Drafing Suite </h6>", unsafe_allow_html=True)
uploaded_file = st.sidebar.file_uploader("Upload a PDF File",type= 'pdf' , key="file")
if uploaded_file is not None:
    def main_page(uploaded_file = uploaded_file):
        st.sidebar.markdown("## Intelligent Document Extraction Assesment ")
        #uploaded_file = st.sidebar.file_uploader("Upload a PDF File",type= 'pdf' , key="file")
        #if uploaded_file is not None:
            
        # progress_bar = st.sidebar.progress(0)
        # for perc_completed in range(100):
        #     time.sleep(0.005)
        #     progress_bar.progress(perc_completed)
        st.sidebar.metric(label=" # Document Uploaded", value = " 100 " , delta = " 5 ")
        st.sidebar.success(" File Uploaded Successfully")
        create_space(1)
        with st.expander(label="# This Page Explains Difference between Intelligent Data Extraction vs Raw Extraction", expanded=True):
            st.markdown("""

            1) Upload a PDF file 

            2) Visualize the page

            3) Check Raw Text

            4) Observe Outcome from Intelligent Data Extraction

            5) Analyze meta data 
            """)
        "---"

    # option = st.sidebar.selectbox(
    # 'Document Granularity Selection',
    # ('Page Level', 'Paragraph Level', 'Sentence Level', 'Word Level'))
    #col1, col2, col3 = st.columns([2,1,1])
        col1, col2= st.columns([1,1])
        col1.markdown("<h3 style='text-align: center; color: grey;'> Document Quick Visuals </h3>", unsafe_allow_html=True)

    # pdf_path = Path("/Users/mishrs39/Downloads/test_breast_file.pdf")
    # base64_pdf = base64.b64encode(pdf_path.read_bytes()).decode("utf-8")
    # pdf_display = f"""
    #     <iframe src="data:application/pdf;base64,{base64_pdf}" width="800px" height="2100px" type="application/pdf"></iframe>
    # """
    # col1.markdown(pdf_display, unsafe_allow_html=True)
    # pdf = pdfplumber.open("/Users/mishrs39/Downloads/test_breast_file.pdf")
    # p0 = pdf.pages[0]
    # im = p0.to_image()
    # im.save("/Users/mishrs39/Downloads/temp.jpeg")
    # col1.image(im, output_format='JPEG')

    #fname = uploaded_doc #sys.argv[1]  # get document filename
    # try: 
    #     doc = fitz.open("pdf", st.session_state.file.read())  # open document
    # except:
    #     doc = fitz.open("/Users/mishrs39/Downloads/test_breast_file.pdf")
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf") 
        page_option = st.sidebar.selectbox(
        'Page Selection',
        (range(0,len(doc))))

        # text_option = st.sidebar.selectbox(
        # 'Page Selection',
        # ('text', 'blocks', 'words', 'html', 
        #     'dict', 'json', 'rawDict', 'xhtml', 'xml'))
            

        zoom = 4
        mat = fitz.Matrix(zoom, zoom)
        val = f"image_{page_option+1}.png"
        page = doc.load_page(page_option)
        pix = page.get_pixmap(matrix=mat)
        pix.save(os.path.join(os.getcwd(),str(val)))

        imager = Image.open(os.path.join(os.getcwd(),str(val)))
        col1.image(imager, caption=val)
        col1.markdown('### Raw Text from Page')
        page = doc[page_option]
        col1.write(page.get_text("spans",sort=True))

        col2.markdown("<h3 style='text-align: center; color: grey;'> Intelligent Extraction Based Outcome </h3>", unsafe_allow_html=True)
        page = doc[page_option]


        # text = page.get_text(text_options)
        #page = doc[1]
        mnb = []
        all_infos = page.get_text("dict", sort=True)
        for i in range(0, len(all_infos['blocks'])):
            try:
                for n in  range(0, len(all_infos['blocks'][i]['lines'])):
                    m = pd.DataFrame.from_dict(all_infos)['blocks'][i]['lines'][n]['spans'][0]
                    res = {key: m[key] for key in m.keys()
                            & {'size', 'flags', 'font', 'color', 'ascender', 'descender', 'text'}}
                    print(res)

                    mm = pd.DataFrame(list(res.keys()), columns = ['Key Text Attribute'])
                    mm['Text Attribute Value'] = list(res.values())
                    


                    mmm = mm.T
                    mmm.columns = mmm.iloc[0]
                    mmm = mmm[1:]
                    mmm['page'] = page_option
                    mmm['blocks'] = i
                    mmm['lines'] = n
                    mnb.append(mmm)
            except:
                pass

        tt = pd.concat(mnb).reset_index(drop=True)

        option = st.sidebar.selectbox(
        'Document Granularity Selection',
        ('blocks','page', 'lines', 'size', 
            'flags'))

        tom = tt.groupby(option).agg({'text':' '.join,'font':'unique','size':'unique'})[:]

        col2.dataframe(tom)

        col2.markdown(' ###### MetaData Description')
        tomtt = pd.DataFrame(page.get_texttrace())
        col2.write(page.get_texttrace())


        #col3.markdown(' ##### Extracted Concepts from Documents')
    def page2(uploaded_file = uploaded_file):
        st.sidebar.markdown("## Base Concept Extraction")
        col1, col2  = st.columns([1,1])
        col1.markdown("<h3 style='text-align: center; color: grey;'> Model Based Medical/Pharma/Clinical Entity Parsing </h3>", unsafe_allow_html=True)
        doc =  fitz.open(stream=uploaded_file.read(), filetype="pdf") #fitz.open("pdf", st.session_state.file.read()) #doc #fitz.open("/Users/mishrs39/Downloads/test_breast_file.pdf")
        page_option = st.sidebar.selectbox(
        'Page Selection',
        (range(0,len(doc))))
        dg = pd.read_csv(os.path.join(os.getcwd(), 'test_breast_file_csv_updated_3.csv'))
        tot12 = dg[dg['page'] ==page_option][['page','blocks','text', 'med_ner','epid_ner']].replace('Reference:','').replace('References:','').drop_duplicates().groupby(['page','blocks']).agg({'text':''.join, 'med_ner':''.join, 'epid_ner':''.join}).drop_duplicates().reset_index(drop=True)
        col1.dataframe(tot12[:-3])

        col1.markdown("<h4 style='text-align: center; color: grey;'> Med-Model Based Key Entities </h4>", unsafe_allow_html=True)

        xc = []
        for m in dg.page.unique():
            for n in dg[dg['page']==m].blocks.unique():
                for k in  dg[(dg['page']==m) & (dg['blocks']==n)].text.unique():
                    bv = pd.DataFrame(dg[(dg['page']==m) & (dg['blocks']==n)& (dg['text']==k)]['keywords_of_interest'])
                    bv['page'] = m
                    bv['blocks'] = n
                    bv['text'] = k
                    xc.append(bv)

        temp = pd.concat(xc).reset_index(drop=True)
        tot13 = temp[temp['page']==page_option].reset_index(drop=True)
        sct = []
        for v in range(0,len(tot13)):
            fh = pd.DataFrame(ast.literal_eval(tot13['keywords_of_interest'][v]))
            sct.append(fh)

        tot12_1 = pd.concat(sct).reset_index(drop=True).drop_duplicates()
        tot12_1['page'] = page_option
        tot13_1 = tot12_1[tot12_1['page']==page_option].reset_index(drop=True)

        col1.dataframe(tot13_1)



        col2.markdown("<h3 style='text-align: center; color: grey;'> Ontology Based Medical/Pharma/Clinical Entity Parsing </h3>", unsafe_allow_html=True)
        def concept_from_entity_onto(data = dg, page_num = 1, entity_cat = 'umls_entity'):
            tt = []

            for i in range(0,len(dg[dg['page']==1])):
                try:
                    fc = pd.DataFrame(ast.literal_eval(data[data['page']==page_num][entity_cat].reset_index(drop=True)[i]))
                except:
                    fc = pd.DataFrame([['0','0','0','0','0','0','0']], columns = ['concept_id', 'canonical_name', 'aliases', 'types', 'definition',
                    'entity', 'text'])
                tt.append(fc)
            gc = pd.concat(tt).reset_index(drop=True)

            return gc


        def concept_from_entity_model(data = dg, page_num = 1, entity_cat = 'epid_ner'):
            tt = []

            for i in range(0,len(dg[dg['page']==1])):
                try:
                    fc = pd.DataFrame(data[data['page']==page_num][entity_cat].reset_index(drop=True)[i])
                except:
                    fc = pd.DataFrame([['0','0','0']], columns = ['entity_group', 'value', 'score'])
                tt.append(fc)
            gc = pd.concat(tt).reset_index(drop=True)

            return gc
        
        tot = []
        for j in ['umls_entity','rxnorm_entity','mesh_entity','go_entity','hpo_entity']:
            for i in dg.page.unique():
                nn = concept_from_entity_onto(data = dg, page_num = i, entity_cat = j)
                nn['page_num'] = i
                nn['entity_type'] = j
                tot.append(nn)

        tt34= pd.concat(tot).reset_index(drop=True)
        tt34x = tt34[tt34['page_num']==page_option].reset_index(drop=True)
        #tt34x = tt34x[tt34x['text'].isin(tot12.text.to_list())]
        col2.dataframe(tt34x)

    def page6(uploaded_file = uploaded_file):
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
        
        default_prompt = ["Create persuasive marketing content in " + option6 + " for " + option2+ ", emphasizing the " +option3+ " tone. Craft a "+ option4+ " that educates them about " + option1 +" role in cancer treatment and its potential benefits. The objective is to " + option5 + " to those seeking "+ option8+" options. The user-defined query is " + option7]
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
        
            return output[0]['generated_text']
        
        
        #User -Provided Prompt
        
        if prompt := st.chat_input(disabled=not add_replicate_api):
            st.session_state.messages.append({"role": "user", "content":prompt})
            with st.chat_message("user"):
                st.write(prompt)
        
        # Generate a New Response if the last message is not from the asssistant
        
        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response=generate_llama2_response(prompt)
                    placeholder=st.empty()
                    full_response=''
                    for item in response:
                        full_response+=item
                        placeholder.markdown(full_response)
                    placeholder.markdown(full_response)
        
            message= {"role":"assistant", "content":full_response}
            st.session_state.messages.append(message)
       
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


    def page3(uploaded_file = uploaded_file):
        st.markdown("<h3 style='text-align: center; color: grey;'> Document Understanding based on Hypothesis </h3>", unsafe_allow_html=True)
        col1, col2 = st.columns([1,1])
        # page_option = st.sidebar.selectbox(
        # 'Page Selection',
        # (range(0,len(doc))))
        col1.markdown("<h4 style='text-align: center; color: grey;'> Hypothesis: Larger the Fonts, Important the message </h4>", unsafe_allow_html=True)
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf") 
        mnbb = []
        for p in range(0,len(doc)):
            page = doc[p]
            # text = page.get_text(text_options)
            #page = doc[1]
            #mnb = []
            all_infos = page.get_text("dict", sort=True)
            for i in range(0, len(all_infos['blocks'])):
                try:
                    for n in  range(0, len(all_infos['blocks'][i]['lines'])):
                        m = pd.DataFrame.from_dict(all_infos)['blocks'][i]['lines'][n]['spans'][0]
                        res = {key: m[key] for key in m.keys()
                                & {'size', 'flags', 'font', 'color', 'ascender', 'descender', 'text'}}
                        print(res)

                        mm = pd.DataFrame(list(res.keys()), columns = ['Key Text Attribute'])
                        mm['Text Attribute Value'] = list(res.values())
                        


                        mmm = mm.T
                        mmm.columns = mmm.iloc[0]
                        mmm = mmm[1:]
                        mmm['page'] = p
                        mmm['blocks'] = i
                        mmm['lines'] = n
                        mnbb.append(mmm)
                except:
                    pass
        dg = pd.concat(mnbb).reset_index(drop=True)
        
        #dg = pd.read_csv(os.path.join(os.getcwd(),'test_breast_file_csv_updated_3.csv'))
        font_dg = pd.DataFrame(dg[['page','font','size']].value_counts()).reset_index().sort_values('size',ascending=False).reset_index(drop=True)
        #font_dg = font_dg[font_dg['page']==page_option]
        col1.dataframe(font_dg)

        font_dg.columns = ['page','font','size','count_para']
        if font_dg['count_para'][:10].sum()>=20:
            fontt = font_dg['font'][:10].to_list()
            rot = dg[dg['font'].isin(fontt)][['page','blocks','text']].replace('Reference:','').replace('References:','').drop_duplicates().groupby(['page']).agg({'text':''.join}).drop_duplicates().reset_index(drop=True)
            col1.dataframe(rot['text'])
        else:
            fontt = font_dg['font'][:15].to_list()
            rot = dg[dg['font'].isin(fontt)][['page','blocks','text']].replace('Reference:','').replace('References:','').drop_duplicates().groupby(['page']).agg({'text':''.join}).drop_duplicates().reset_index(drop=True)
            col1.dataframe(rot['text'])
        
        #col1.markdown("<h3 style='text-align: center; color: grey;'> Document Understanding Based on Fonts Size (Larger the Fonts Important the message) </h3>", unsafe_allow_html=True)

        col2.markdown("<h4 style='text-align: center; color: grey;'> Proposed Tags/ Concept for Document </h4>", unsafe_allow_html=True)
        dg_g = pd.read_csv(os.path.join(os.getcwd(),'Demo_lab_1 - Demo_lab.csv'))
        print(uploaded_file.name)
        if uploaded_file.name == 'Residual Disease Management In HER2+ve Early Breast Cancer Setting - Case Discussion.pdf':
            col2.write(ast.literal_eval(dg_g['Tags'][0]))
            col2.markdown("<h4 style='text-align: center; color: grey;'> Short Summary based on NLP Model </h4>", unsafe_allow_html=True)
            col2.write(dg_g['Summary'][0])
        elif uploaded_file.name == 'test_breast_file.pdf':
            col2.write(ast.literal_eval(dg_g['Tags'][1]))
            col2.markdown("<h4 style='text-align: center; color: grey;'> Short Summary based on NLP Model </h4>", unsafe_allow_html=True)
            col2.write(dg_g['Summary'][1])
        elif uploaded_file.name == 'APAC DAN Lung PPoC Insight WP (last updated 2023.08.08).pdf':
            col2.write(ast.literal_eval(dg_g['Tags'][2]))
            col2.markdown("<h4 style='text-align: center; color: grey;'> Short Summary based on NLP Model </h4>", unsafe_allow_html=True)
            col2.write(dg_g['Summary'][2])
    def page4():
        st.markdown("<h3 style='text-align: center; color: grey;'> Semantic Search within Document </h3>", unsafe_allow_html=True)
        st.sidebar.markdown("## Semantic Search ")

       
        # read hotel reviews dataframe

        data=pd.read_csv(os.path.join(os.getcwd(),'test_breast_file_csv_updated_3.csv'))
        # Load a pre-trained model

        model =SentenceTransformer('msmarco-MiniLM-L-12-v3')
        passages=list(set(data.groupby(['page', 'blocks']).agg({'text':' '.join,'font':'unique','size':'unique'})['text']))
        #We use the Bi-Encoder to encode all passages, so that we can use it with sematic search
        bi_encoder = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
        bi_encoder.max_seq_length = 256     #Truncate long passages to 256 tokens
        top_k = 32                          #Number of passages we want to retrieve with the bi-encoder

        #The bi-encoder will retrieve 100 documents. We use a cross-encoder, to re-rank the results list to improve the quality
        cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

        # As dataset, we use Simple English Wikipedia. Compared to the full English wikipedia, it has only
        # about 170k articles. We split these articles into paragraphs and encode them with the bi-encoder


        print("Passages:", len(passages))

        # We encode all passages into our vector space. This takes about 5 minutes (depends on your GPU speed)
        corpus_embeddings = bi_encoder.encode(passages, convert_to_tensor=True, show_progress_bar=True)
        # embed hotel reviews
        # We lower case our text and remove stop-words from indexing
        def bm25_tokenizer(text):
            tokenized_doc = []
            for token in text.lower().split():
                token = token.strip(string.punctuation)

                if len(token) > 0 and token not in _stop_words.ENGLISH_STOP_WORDS:
                    tokenized_doc.append(token)
            return tokenized_doc


        tokenized_corpus = []
        for passage in tqdm(passages):
            tokenized_corpus.append(bm25_tokenizer(passage))

        bm25 = BM25Okapi(tokenized_corpus)
        def search(query):
            print("Input question:", query)

            ##### BM25 search (lexical search) #####
            bm25_scores = bm25.get_scores(bm25_tokenizer(query))
            top_n = np.argpartition(bm25_scores, -5)[-5:]
            bm25_hits = [{'corpus_id': idx, 'score': bm25_scores[idx]} for idx in top_n]
            bm25_hits = sorted(bm25_hits, key=lambda x: x['score'], reverse=True)
            
            print("Top-3 lexical search (BM25) hits")
            for hit in bm25_hits[0:3]:
                print("\t{:.3f}\t{}".format(hit['score'], passages[hit['corpus_id']].replace("\n", " ")))

            ##### Sematic Search #####
            # Encode the query using the bi-encoder and find potentially relevant passages
            question_embedding = bi_encoder.encode(query, convert_to_tensor=True)
            question_embedding = question_embedding#.cuda()
            hits = util.semantic_search(question_embedding, corpus_embeddings, top_k=top_k)
            hits = hits[0]  # Get the hits for the first query

            ##### Re-Ranking #####
            # Now, score all retrieved passages with the cross_encoder
            cross_inp = [[query, passages[hit['corpus_id']]] for hit in hits]
            cross_scores = cross_encoder.predict(cross_inp)

            # Sort results by the cross-encoder scores
            for idx in range(len(cross_scores)):
                hits[idx]['cross-score'] = cross_scores[idx]

            # Output of top-5 hits from bi-encoder
            print("\n-------------------------\n")
            print("Top-3 Bi-Encoder Retrieval hits")
            hits = sorted(hits, key=lambda x: x['score'], reverse=True)
            for hit in hits[0:3]:
                print("\t{:.3f}\t{}".format(hit['score'], passages[hit['corpus_id']].replace("\n", " ")))

            # Output of top-5 hits from re-ranker
            print("\n-------------------------\n")
            print("Top-3 Cross-Encoder Re-ranker hits")
            hits = sorted(hits, key=lambda x: x['cross-score'], reverse=True)
            for hit in hits[0:3]:
                print("\t{:.3f}\t{}".format(hit['cross-score'], passages[hit['corpus_id']].replace("\n", " ")))
            return hits, bm25_hits
        question_asked = st.text_area('Input your query','')
        results=search(query = question_asked)

        bm = pd.DataFrame(results[1])
        bm['text'] = bm['corpus_id'].apply(lambda m :passages[m] )

        bm_bis = pd.DataFrame(results[0])
        bm_bis = bm_bis.sort_values('score',ascending=False)
        bm_bis['text'] = bm_bis['corpus_id'].apply(lambda m :passages[m])
      

        bm_co = pd.DataFrame(results[0])
        bm_co = bm_co.sort_values('cross-score',ascending=False)
        bm_co['text'] = bm_co['corpus_id'].apply(lambda m :passages[m])
   
        st.markdown('lexical search (BM25) hits')
        st.dataframe(bm[['score','text']])
        st.markdown('Bi-Encoder Retrieval hits')
        st.dataframe(bm_bis[['score','text']][:10])
        st.markdown('Cross-Encoder Re-ranker hits')
        st.dataframe(bm_co[['cross-score','text']][:10])
            



        # except:
        #     open_ai_key = st.sidebar.text_area('Input Your Open AI Key',' ')
        #     relevant_chunks = st.sidebar.selectbox(
        #     'Relevant Chunks',
        #     (range(2,7)))

        #     chain_type = st.sidebar.selectbox(
        #     'Chain Type',
        #     (['stuff', 'map_reduce', "refine", "map_rerank"]))

            
        #     question_asked = st.text_area('Input your query','')
            
        #     question_asked_predefined = st.sidebar.selectbox(
        #     'Pre-Defined Queries',
        #     (['identify all concepts discussed around quality of life', 'identify all concepts discussed aboutr efficacy']))


        #     #function to formulate question answer
        #     def qa(file, query, chain_type, k):
        #         # load document
        #         loader = PyPDFLoader(file)
        #         documents = loader.load()
        #         # split the documents into chunks
        #         text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        #         texts = text_splitter.split_documents(documents)
        #         # select which embeddings we want to use
        #         embeddings = OpenAIEmbeddings()
        #         # create the vectorestore to use as the index
        #         db = Chroma.from_documents(texts, embeddings)
        #         # expose this index in a retriever interface
        #         retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})
        #         # create a chain to answer questions 
        #         qa = RetrievalQA.from_chain_type(
        #             llm=OpenAI(), chain_type=chain_type, retriever=retriever, return_source_documents=True)
        #         result = qa({"query": query})
        #         print(result['result'])
        #         return result  
        #     #convos = []  # store all panel objects in a list

        #     def qa_result(_):
        #         os.environ["OPENAI_API_KEY"] = open_ai_key.value
                
        #         # save pdf file to a temp file 
        #         if uploaded_file.value is not None:
        #             uploaded_file.save("/.cache/temp.pdf")
        #             try:
        #                 prompt_text = question_asked.value
        #             except:
        #                 prompt_text = question_asked_predefined.value

        #             if prompt_text:
        #                 result = qa(file=uploaded_file, query=prompt_text, chain_type=chain_type.value, k=relevant_chunks.value) 
        #         return result
            
        #     if open_ai_key is not None:
        #         temp = qa_result

        #     st.write(temp)




    def page5(uploaded_file = uploaded_file):
        st.markdown("<h3 style='text-align: center; color: grey;'> PDF Summary & Image Extraction </h3>", unsafe_allow_html=True)
        #PDF_DATA_KEY = uploaded_file
        pdf_data = set_session_state_key(PDF_DATA_KEY, None)
        text_data = load_state(TEXT_DATA_KEY, "")

        #st.header("📝 GistR | PDF Summary & Image Extraction")

        create_space(1)
        with st.expander(label="How to use this Page", expanded=True):
            st.markdown("""

            1) Upload a PDF file or a valid PDF url

            2) Change the settings in the sidebar (optional)

            3) Click on the "Extract Images" button to see all images in the document

            4) Click on the "Download Image" button to select the image(s) you would like to download

            5) Click on the "Summarise Text" button to run the AI model on the PDF text (may take a while depending on the length of the file)

            6) Click on "Download Summary" to download the summary as a .txt file
            """)
        "---"
        create_space(1)
        #st.markdown("##### Upload A PDF File")
        pdf_file = fitz.open(stream=uploaded_file.read(), filetype="pdf")#upload_pdf_file()
        st.markdown("### You can also examine PDF URL")
        st.markdown("##### Input A PDF URL")
        st.caption("The link should be a pure http(s) path and should end in .pdf")

        pdf_link = get_pdf_from_link()
        pdf_data = return_pdf_data(pdf_file, pdf_link)

        sidebar_widget()

        create_space(1)

        if pdf_data:
            get_text_data_from_pdf(pdf_data)
            load_pdf_report_summary()
            "---"
            st.markdown("#### Image Extraction")
            image_extraction_component(pdf_data, verbose=False)
            create_space(1)


            st.markdown("#### Text Summary")
            text_summary_component()
            create_space(1)
            "---"
    def page8():
        st.markdown("<h3 style='text-align: center; color: grey;'> 🦙 Content Generation for Marketing Content </h3>", unsafe_allow_html=True)
        st.title('🦙 Content Query Builder')
        option1 = st.sidebar.selectbox(
        'Content Type',
        ('blocks','page', 'lines', 'size', 
            'flags'))
        option2 = st.sidebar.selectbox(
        'Tone Type',
        ('blocks','page', 'lines', 'size', 
            'flags'))

        option2 = st.sidebar.selectbox(
        'Disease Area',
        ('blocks','page', 'lines', 'size', 
            'flags'))
        prompt = st.text_input('Input your prompt here')
        st.write(result['source_documents'])
        
        # ### Query Based evidence identification:
        # model_name_or_path = "TheBloke/WizardLM-13B-V1.2-GPTQ"
        # #model_basename = "wizardlm-13b-v1.1-GPTQ-4bit-128g.no-act.order"
        
        # use_triton = False
        
        # tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        
        # model = AutoGPTQForCausalLM.from_quantized(model_name_or_path,
        #                                            #model_basename = model_basename,
        #                                            use_safetensors=True,
        #                                            trust_remote_code=True,
        #                                            device="cuda:0",
        #                                            use_triton=use_triton,
        #                                            quantize_config=None)
        
        
        # prompt = "Tell me about AI"
        # prompt_template=f'''A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
        
        # USER: {prompt}
        # ASSISTANT:
        
        # '''
        
        # print("\n\n*** Generate:")
        
        # input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
        # output = model.generate(inputs=input_ids, temperature=0.7, max_new_tokens=512)
        # print(tokenizer.decode(output[0]))
        
        # # Inference can also be done using transformers' pipeline
        
        # # Prevent printing spurious transformers error when using pipeline with AutoGPTQ
        # logging.set_verbosity(logging.CRITICAL)
        
        # print("*** Pipeline:")
        # pipe = pipeline(
        #     "text-generation",
        #     model=model,
        #     tokenizer=tokenizer,
        #     max_new_tokens=512,
        #     temperature=0.7,
        #     top_p=0.95,
        #     repetition_penalty=1.15
        # )
        
        # llm = HuggingFacePipeline(pipeline=pipe)
        
        
        # ### Data Loadinga and Embedding
        # loader = PyPDFDirectoryLoader("data/")
        # documents = loader.load()
        
        # from langchain.text_splitter import RecursiveCharacterTextSplitter
        
        # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
        # all_splits = text_splitter.split_documents(documents)
        
        
        # model_name = "sentence-transformers/all-mpnet-base-v2"
        # model_kwargs = {"device": "cuda"}
        # embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
        # # storing embeddings in the vector store
        # vectorstore = FAISS.from_documents(all_splits, embeddings)
        
        # ### Chain of discussion
        # chain = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(), return_source_documents=True)
        
        # chat_history = []
        # # Create centered main title
        # st.markdown("<h3 style='text-align: center; color: grey;'> 🦙 Content Query Builder to extract Evidence </h3>", unsafe_allow_html=True)
        # st.title('🦙 Content Query Builder')
        # # Create a text input box for the user
        # prompt = st.text_input('Input your prompt here')
        
        # # If the user hits enter
        # if prompt:
        #     response = chain({"question": prompt, "chat_history": chat_history})
        #     # ...and write it out to the screen
        #     st.write(response['answer'])
        
        #     # Display raw response object
        #     with st.expander('Response Object'):
        #         st.write(response['answer'])
        #     # Display source text
        #     with st.expander('Source Text'):
        #         st.write(result['source_documents'])

    
    page_names_to_funcs = {
        "Intelligent Data Parsing": main_page,
        "Contextual Tags (Based on Hypothesis)": page3,
        #"Full Document Summary (Image and Text)": page5,
        "Content Generation" : page6,
        #"Non - Contextual Tags, Iterate over pages": page2,
        #"Search across Document": page4,
        #"Query Based Evidence Generation": page8,
    }

    selected_page = st.sidebar.selectbox("# Analysis Selection", page_names_to_funcs.keys())
    page_names_to_funcs[selected_page]()
