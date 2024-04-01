import os
import pickle
import random
import streamlit as st
import torch

from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.vectorstores.qdrant import Qdrant
from langchain.text_splitter import TokenTextSplitter
from PyPDF2 import PdfReader
from qdrant_client import QdrantClient
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline

from streamlit.web.server import websocket_headers
from streamlit_chat import message

def get_pdf_text():
    """
    Function to load PDF text and split it into chunks.
    """
    st.header("Document Upload")
    uploaded_file = st.file_uploader(
        label="Here, upload your PDF file ",
        type="pdf"
    )
    if uploaded_file:
        pdf_reader = PdfReader(uploaded_file)
        text = "\n\n".join([page.extract_text() for page in pdf_reader.pages])
        text_splitter = TokenTextSplitter(chunk_size=100, chunk_overlap=0)
        return text_splitter.split_text(text)
    else:
        return None



os.environ['SENTENCE_TRANSFORMERS_HOME'] = '/mnt/data/Local-RAG-Development'


prompt_template = """Use the following pieces of context to answer the question enclosed within  3 backticks at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
Please provide an answer which is factually correct and based on the information retrieved from the vector store.
Please also mention any quotes supporting the answer if any present in the context supplied within two double quotes "" .

{context}

QUESTION:```{question}```
ANSWER:
"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context","question"])
#
chain_type_kwargs = {"prompt": PROMPT}

# Initialise session state variables
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]


st.set_page_config(initial_sidebar_state='collapsed')
st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/0/03/Lenovo_Global_Corporate_Logo.png/2560px-Lenovo_Global_Corporate_Logo.png")
clear_button = st.sidebar.button("Clear Conversation", key="clear")

qa_chain = None
doc_store = None

if clear_button:
    st.session_state['generated'] = []
    st.session_state['past'] = []
    st.session_state['messages'] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
    
embedding_model_name = "BAAI/bge-small-en"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}

embeddings = HuggingFaceBgeEmbeddings(model_name=embedding_model_name,
                                      model_kwargs=model_kwargs,
                                      encode_kwargs=encode_kwargs
                                     )

pdf_texts = get_pdf_text()

if pdf_texts:
    doc_store = Qdrant.from_texts(texts=pdf_texts,
                                  embedding=embeddings,
                                  location=":memory:",
                                  collection=f"{embedding_model_name}_press_release"
                                 )

if doc_store:
    chain_type_kwargs = {"prompt": PROMPT}


    ################################################################################
    # bitsandbytes parameters
    ################################################################################

    # Activate 4-bit precision base model loading
    use_4bit = True

    # Compute dtype for 4-bit base models
    bnb_4bit_compute_dtype = "float16"

    # Quantization type (fp4 or nf4)
    bnb_4bit_quant_type = "nf4"

    # Activate nested quantization for 4-bit base models (double quantization)
    use_nested_quant = False

    # Load tokenizer and model with QLoRA configuration
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
    )

    model_id = "NousResearch/Llama-2-7b-chat-hf"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir="/mnt/data/Local-RAG-Development",
        quantization_config=bnb_config,
        device_map='auto'
    )

    # Load LLaMA tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.pad_token_id = model.config.eos_token_id
    
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100)
    rag_llm = HuggingFacePipeline(pipeline=pipe)

    qa_chain = RetrievalQA.from_chain_type(llm=rag_llm,
                                           chain_type="stuff",
                                           chain_type_kwargs={"prompt": PROMPT},
                                           retriever=doc_store.as_retriever(search_kwargs={"k": 5}),
                                           return_source_documents=True
                                          )
# container for chat history
response_container = st.container()
# container for text box
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_area("You:", key='input', height=100)
        submit_button = st.form_submit_button(label='Send')
    if submit_button and user_input and qa_chain:
        answer = None
        with st.spinner("Searching for the answer..."):
            result = qa_chain(user_input)
        if result:
            answer = result["result"]
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(answer)
        
    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, logo='https://freesvg.org/img/1367934593.png', key=str(i) + '_user')
                message(st.session_state["generated"][i], logo='https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQk6e8aarUy37BOHMTSk-TUcs4AyAy3pfAHL-F2K49KHNEbI0QUlqWJFEqXYQvlBdYMMJA&usqp=CAU', key=str(i))