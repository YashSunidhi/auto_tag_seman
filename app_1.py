import streamlit as st
import time
import sys, fitz
import pdfplumber
import base64
from pathlib import Path
from PIL import Image
import re

import pandas as pd
import ast
import gzip
import os
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
        st.title("Marketing Compaign Email Generator")
        input_text = st.text_input("Write an executive short email for internal purposes based on document summary? ")
        dg_g = pd.read_csv(os.path.join(os.getcwd(),'Demo_lab.csv'))
        if uploaded_file.name == 'Residual Disease Management In HER2+ve Early Breast Cancer Setting - Case Discussion.pdf':
            output_text = st.write(dg_g['Email'][0])
            
        elif uploaded_file.name == 'test_breast_file.pdf':
            output_text = st.write(dg_g['Email'][1])
           
        elif uploaded_file.name == 'APAC DAN Lung PPoC Insight WP (last updated 2023.08.08).pdf':
            output_text = st.write(dg_g['Email'][2])
            
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
        dg_g = pd.read_csv(os.path.join(os.getcwd(),'Demo_lab.csv'))
        print(uploaded_file.name)
        if uploaded_file.name == 'Residual Disease Management In HER2+ve Early Breast Cancer Setting - Case Discussion.pdf':
            col2.write(dg_g['Tags'][0])
            col2.markdown("<h4 style='text-align: center; color: grey;'> Short Summary based on NLP Model </h4>", unsafe_allow_html=True)
            col2.write(dg_g['Summary'][0])
        elif uploaded_file.name == 'test_breast_file.pdf':
            col2.write(dg_g['Tags'][1])
            col2.markdown("<h4 style='text-align: center; color: grey;'> Short Summary based on NLP Model </h4>", unsafe_allow_html=True)
            col2.write(dg_g['Summary'][1])
        elif uploaded_file.name == 'APAC DAN Lung PPoC Insight WP (last updated 2023.08.08).pdf':
            col2.write(dg_g['Tags'][2])
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
        ### Query Based evidence identification:
        # define custom stopping criteria object
        class StopOnTokens(StoppingCriteria):
            def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
                for stop_ids in stop_token_ids:
                    if torch.eq(input_ids[0][-len(stop_ids):], stop_ids).all():
                        return True
                return False
        
        stopping_criteria = StoppingCriteriaList([StopOnTokens()])
        
        ### Model Loading
        model_id = 'meta-llama/Llama-2-13b-chat-hf'
        #model_id = "conceptofmind/Yarn-Llama-2-13b-128k"
        
        device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
        
        # set quantization configuration to load large model with less GPU memory
        # this requires the `bitsandbytes` library
        bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=bfloat16
        )
        
        # begin initializing HF items, you need an access token
        hf_auth = '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'
        model_config = transformers.AutoConfig.from_pretrained(
            model_id,
            use_auth_token=hf_auth
        )
        
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            config=model_config,
            quantization_config=bnb_config,
            device_map='auto',
            use_auth_token=hf_auth
        )
        
        # enable evaluation mode to allow model inference
        model.eval()
        
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_id,
            use_auth_token=hf_auth
        )
        
        stop_list = ['\nHuman:', '\n```\n']
        stop_token_ids = [tokenizer(x)['input_ids'] for x in stop_list]
        stop_token_ids = [torch.LongTensor(x).to(device) for x in stop_token_ids]
        
        generate_text = transformers.pipeline(
            model=model,
            tokenizer=tokenizer,
            return_full_text=True,  # langchain expects the full text
            task='text-generation',
            # we pass model parameters here too
            stopping_criteria=stopping_criteria,  # without this model rambles during chat
            temperature=0.1,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
            max_new_tokens=512,  # max number of tokens to generate in the output
            repetition_penalty=1.1  # without this output begins repeating
        )
        
        llm = HuggingFacePipeline(pipeline=generate_text)
        
        
        ### Data Loadinga and Embedding
        loader = PyPDFDirectoryLoader("data/")
        documents = loader.load()
        
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
        all_splits = text_splitter.split_documents(documents)
        
        
        model_name = "sentence-transformers/all-mpnet-base-v2"
        model_kwargs = {"device": "cuda"}
        embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
        # storing embeddings in the vector store
        vectorstore = FAISS.from_documents(all_splits, embeddings)
        
        ### Chain of discussion
        chain = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(), return_source_documents=True)
        
        chat_history = []
        # Create centered main title
        st.markdown("<h3 style='text-align: center; color: grey;'> 🦙 Content Query Builder to extract Evidence </h3>", unsafe_allow_html=True)
        st.title('🦙 Content Query Builder')
        # Create a text input box for the user
        prompt = st.text_input('Input your prompt here')
        
        # If the user hits enter
        if prompt:
            response = chain({"question": prompt, "chat_history": chat_history})
            # ...and write it out to the screen
            st.write(response['answer'])
        
            # Display raw response object
            with st.expander('Response Object'):
                st.write(response['answer'])
            # Display source text
            with st.expander('Source Text'):
                st.write(result['source_documents'])

    
    page_names_to_funcs = {
        "Intelligent Data Parsing": main_page,
        "Contextual Tags (Based on Hypothesis)": page3,
        #"Full Document Summary (Image and Text)": page5,
        "Content Generation" : page6,
        #"Non - Contextual Tags, Iterate over pages": page2,
        #"Search across Document": page4,
        "Query Based Evidence Generation": page8,
    }

    selected_page = st.sidebar.selectbox("# Analysis Selection", page_names_to_funcs.keys())
    page_names_to_funcs[selected_page]()
