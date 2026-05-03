# One Stop shop creation of a question answering app on top of a set of text
# based on LangChain
# Loads .txt files and .pdf files from a folder
# computes embeddings using OpenAI API (Ada) for overlapping (!) text chunks
# stores those embeddings in an index 
# whcih then, when a user asks a question and an embedding is calculated for that question
# finds the related text chunks and asks OpenAI (GPT) to answer the question based on the candidate chunks

import os
from dotenv import load_dotenv
import joblib
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from pdfminer.high_level import extract_text
import re
import glob
import datetime


# extracts just the "GUID" prefixed string from the .html filename, to use for various links
def extract_guid(filename):
    match = re.match(r'(GUID-[\w-]+\.html)', filename)
    if match:
        return match.group(1)
    else:
        return None

load_dotenv()

st.set_page_config(page_title="Ask your knowledgebase")
st.header("Ask your knowledgebase 💬")

# where to load the data from
pdf_folder_path = "dataset/SLT"        # where can PDFs be found
txt_folder_path = "dataset/SLT"     # where can .TXT files be found

import datetime

# let user choose the file
joblib_files = glob.glob('*.joblib')
selected_file = st.sidebar.selectbox('Select a knowledge base', joblib_files)

# Check if the user selected a file or pressed the "create a new one" button
if st.sidebar.button('Create a new one') or not selected_file:
    with st.spinner('Creating knowledge base...'):
        selected_file = f"knowledge_base_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
        text_dict = {}
        # get all text from PDFs
        for filename in os.listdir(pdf_folder_path):
            if filename.endswith(".pdf"):
                pdf_path = os.path.join(pdf_folder_path, filename)
                pdf_text = extract_text(pdf_path)
                text_dict[filename] = pdf_text

        # get all text from TXTs
        for filename in os.listdir(txt_folder_path):
            if filename.endswith(".txt"):
                txt_path = os.path.join(txt_folder_path, filename)

                # Extract the text
                with open(txt_path, "r", encoding='utf-8') as file:
                    text_dict[filename] = file.read()

        chunks_dict = {}
        # split into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        for filename, text in text_dict.items():
            chunks = text_splitter.split_text(text)
            for i, chunk in enumerate(chunks):
                # Use filename and chunk index as key to allow multiple chunks from same file
                chunks_dict[f"{filename}_{i}"] = chunk

        # create embeddings
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(list(chunks_dict.values()), embeddings)

        # Store knowledge base and chunks_dict to disk, for future reuse
        joblib.dump((knowledge_base, chunks_dict), selected_file)
else:
    with st.spinner('Loading the knowledge base...'):
        knowledge_base, chunks_dict = joblib.load(selected_file)

# show user input
user_question = st.text_input("Ask a question, don't be shy:")
if user_question:
    print('Performing similarity search...')

    docs = knowledge_base.similarity_search(user_question)

    print('Prettyfying the response')

    # ask GPT (using LangChain to process the Q&A better) to answer the user_question based on the identified docs
    llm = OpenAI()
    chain = load_qa_chain(llm, chain_type="stuff")
    with get_openai_callback() as cb:
        response = chain.run(input_documents=docs, question=user_question)
        st.write(f" **:blue[{response}]** ")
        
        # write Open AI API usage info in a pretty way :)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total tokens", cb.total_tokens)
        col2.metric("Prompt", cb.prompt_tokens)
        col3.metric("Completion", cb.completion_tokens)
        col4.metric("Cost (USD)", cb.total_cost)

    # identify which content snippets were used and the documentation source files for them
    # based on identified source files, a web link could also be constructed (based on the GUID)
    # to help the user jump to the official documentation for more details
    st.header('Snippets used for this:')
    for i, doc in enumerate(docs):
        # Get filename of the document
        filename = [file for file, chunk in chunks_dict.items() if chunk == doc.page_content][0]
        guid_html = extract_guid(filename)
        with st.expander(f'Snippet {i+1} from {filename} [link](http://{guid_html})'):
            st.caption(f' ')
            if guid_html:
                file_path = f"dataset/html/{guid_html}"
                if os.path.exists(file_path):
                    with open(file_path, "r") as file:
                        html_content = file.read()
                        st.markdown(html_content, unsafe_allow_html=True)
        st.write(doc.page_content)
