import streamlit as st
import time
import sys, fitz
import pdfplumber
import base64
from pathlib import Path
from PIL import Image
import re
from st_aggrid import AgGrid
import pandas as pd
import ast
from streamlit_chat import message

import os 
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import tempfile
import numpy as np
import torch
import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import time

import streamlit as st
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
st.markdown("<h1 style='text-align: center; color: black;'> Content Auto-Tagging Application </h1>", unsafe_allow_html=True)
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
        pix.save("/Users/mishrs39/Downloads/"+str(val))

        imager = Image.open("/Users/mishrs39/Downloads/"+str(val))
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
        dg = pd.read_csv('/Users/mishrs39/Downloads/test_breast_file_csv_updated_3.csv')
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


    def page3(uploaded_file = uploaded_file):
        st.markdown("<h3 style='text-align: center; color: grey;'> Document Understanding based on Hypothesis </h3>", unsafe_allow_html=True)
        col1, col2 = st.columns([1,1])
        # page_option = st.sidebar.selectbox(
        # 'Page Selection',
        # (range(0,len(doc))))
        col1.markdown("<h4 style='text-align: center; color: grey;'> Hypothesis: Larger the Fonts, Important the message </h4>", unsafe_allow_html=True)
        dg = pd.read_csv('/Users/mishrs39/Downloads/test_breast_file_csv_updated_3.csv')
        font_dg = pd.DataFrame(dg[['page','font','size']].value_counts()).reset_index().sort_values('size',ascending=False).reset_index(drop=True)
        #font_dg = font_dg[font_dg['page']==page_option]
        col1.dataframe(font_dg)
        
        
        col1.markdown("<h4 style='text-align: center; color: grey;'> All text/messages subject to their font Size/Type </h4>", unsafe_allow_html=True)
        rot = dg[dg['font']=='ConduitITC-Bold'][['page','blocks','text']].replace('Reference:','').replace('References:','').drop_duplicates().groupby(['page']).agg({'text':''.join}).drop_duplicates().reset_index(drop=True)##['text'].to_list())
        col1.dataframe(rot['text'])
        #col1.markdown("<h3 style='text-align: center; color: grey;'> Document Understanding Based on Fonts Size (Larger the Fonts Important the message) </h3>", unsafe_allow_html=True)

        col2.markdown("<h4 style='text-align: center; color: grey;'> Extract Concepts for Highest font size/Type from NLP Based Pipeline (QA,GEN AI, Taxonomy) </h4>", unsafe_allow_html=True)
        dg_g = pd.read_csv('/Users/mishrs39/Downloads/LLM_Based_Summary.csv')
        col2.dataframe(dg_g)
        col2.markdown("<h4 style='text-align: center; color: grey;'> Short Summary based on NLP Model </h4>", unsafe_allow_html=True)

        col2.write("Kadcyla offers a promising solution for patients with residual invasive disease, providing a different direction in treatment. With dual antitumour activity and the ability to adapt therapy based on neoadjuvant response, Kadcyla reduces recurrence risk by 50% compared to trastuzumab. Its efficacy and safety were confirmed in a phase 3 study, although adverse events led to discontinuation in some cases. Considering the entire HER2-positive eBC treatment journey can optimize patient outcomes.")
    def page4():
        st.markdown("<h3 style='text-align: center; color: grey;'> Semantic Search within Document </h3>", unsafe_allow_html=True)
        st.sidebar.markdown("## Semantic Search ")

        try:
            # read hotel reviews dataframe

            data=pd.read_csv('/Users/mishrs39/Downloads/test_breast_file_csv_updated_3.csv')
            # Load a pre-trained model

            model =SentenceTransformer('msmarco-MiniLM-L-12-v3')
            hotel_reviews=data["text"].tolist()

            # embed hotel reviews

            hotel_reviews_embds=model.encode(hotel_reviews)

            # Create an index using FAISS
            index = faiss.IndexFlatL2(hotel_reviews_embds.shape[1])
            index.add(hotel_reviews_embds)
            faiss.write_index(index, 'index_hotel_reviews')
            index = faiss.read_index('index_hotel_reviews')


            def search(query):
    
                t=time.time()
                query_vector = model.encode([query])
                k = 5
                top_k = index.search(query_vector, k)
                print('totaltime: {}'.format(time.time()-t))
                return [hotel_reviews[_id] for _id in top_k[1].tolist()[0]]
            question_asked = st.text_area('Input your query','')
            results=search(question_asked)
            st.write(results)
            



        except:
            open_ai_key = st.sidebar.text_area('Input Your Open AI Key',' ')
            relevant_chunks = st.sidebar.selectbox(
            'Relevant Chunks',
            (range(2,7)))

            chain_type = st.sidebar.selectbox(
            'Chain Type',
            (['stuff', 'map_reduce', "refine", "map_rerank"]))

            
            question_asked = st.text_area('Input your query','')
            
            question_asked_predefined = st.sidebar.selectbox(
            'Pre-Defined Queries',
            (['identify all concepts discussed around quality of life', 'identify all concepts discussed aboutr efficacy']))


            #function to formulate question answer
            def qa(file, query, chain_type, k):
                # load document
                loader = PyPDFLoader(file)
                documents = loader.load()
                # split the documents into chunks
                text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
                texts = text_splitter.split_documents(documents)
                # select which embeddings we want to use
                embeddings = OpenAIEmbeddings()
                # create the vectorestore to use as the index
                db = Chroma.from_documents(texts, embeddings)
                # expose this index in a retriever interface
                retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})
                # create a chain to answer questions 
                qa = RetrievalQA.from_chain_type(
                    llm=OpenAI(), chain_type=chain_type, retriever=retriever, return_source_documents=True)
                result = qa({"query": query})
                print(result['result'])
                return result  
            #convos = []  # store all panel objects in a list

            def qa_result(_):
                os.environ["OPENAI_API_KEY"] = open_ai_key.value
                
                # save pdf file to a temp file 
                if uploaded_file.value is not None:
                    uploaded_file.save("/.cache/temp.pdf")
                    try:
                        prompt_text = question_asked.value
                    except:
                        prompt_text = question_asked_predefined.value

                    if prompt_text:
                        result = qa(file=uploaded_file, query=prompt_text, chain_type=chain_type.value, k=relevant_chunks.value) 
                return result
            
            if open_ai_key is not None:
                temp = qa_result

            st.write(temp)




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

    page_names_to_funcs = {
        "Intelligent Data Parsing": main_page,
        "Non - Contextual Tags, Iterate over pages": page2,
        "Contextual Tags (Based on Hypothesis)": page3,
        "Full Document Summary (Image and Text)": page5,
        "Search across Document": page4,
    }

    selected_page = st.sidebar.selectbox("# Analysis Selection", page_names_to_funcs.keys())
    page_names_to_funcs[selected_page]()