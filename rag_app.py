import streamlit as st
import os
import fitz
import json
from groq import Groq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from PyPDF2 import PdfReader
from dotenv import load_dotenv

load_dotenv()

def get_pdf_text(pdf_docs):
    try:
        text = ""
        for pdf_path in pdf_docs:
            pdf_reader = PdfReader(pdf_path)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text
    except Exception as e:
        print(f"Error extracting text from PDFs: {e}")
        return None

def chunk_text(text):
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=700)
        return text_splitter.split_text(text)
    except Exception as e:
        print(f"Error chunking text: {e}")
        return None
    
def create_documents(pdf_info):
    all_docs = []
    try:
        chunks = chunk_text(pdf_info)
        if chunks is None:
            return all_docs

        for chunk in chunks:
            doc = Document(page_content=chunk)
            all_docs.append(doc)
        return all_docs
    except Exception as e:
        print(f"Error creating documents: {e}")
        return []

def save_vector_store(docs):
    embeddings_obj = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2") 
    vectorstore = FAISS.from_documents(docs, embeddings_obj)
    vectorstore.save_local('faiss_index')
    
def load_and_retrieve(query):
    embeddings_obj = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    try:
        store = FAISS.load_local('faiss_index', embeddings_obj, allow_dangerous_deserialization=True)
        print("Index loaded successfully.")
    except Exception as e:
        print(f"Error loading index: {e}")
        return []

    documents = [Document(page_content=doc.page_content, metadata=doc.metadata)
                 for doc in store.docstore._dict.values()]

    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 3

    faiss_retriever = store.as_retriever(search_kwargs={"k": 3})


    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.7, 0.3]
    )

    results = ensemble_retriever.get_relevant_documents(query)
    return results

def generate_response(relevant_text, query, chat_history):
    client = Groq(api_key=os.getenv('GROQ_API_KEY'))
    background_info = """
    Q. What is Pi RAG?
    A. Pi RAG is a Retrieval-Augmented Generation (RAG) system built using the Pi LLM. It combines retrieval-based information with generative AI, offering versatile question-answering capabilities by leveraging external reference documents and its vast knowledge base.
    Q. Who built you?
    A. I was built by a team of AI researchers and developers at 169Pi AI, an Artificial Intelligence Research Organisation.
    Q. Who created you?
    A. I was created by a team of researchers and engineers at 169Pi AI, an organisation dedicated to establishing new standards in AI research.
    Q. How does Pi RAG retrieve information?
    A. Pi RAG uses a retrieval mechanism that searches through uploaded PDFs, documents to find the most relevant information. It then uses Pi LLM model to synthesize a clear and coherent response based on the retrieved content.
    Q. What can you do?
    A. As part of the pi_rag system, I can:
        - Answer complex questions by retrieving relevant information from documents or internal knowledge.
        - Search and retrieve information from uploaded documents like PDFs.
    Q. Which LLM you are using?
    A. I am using Pi LLM build by 169Pi AI Team.
"""
    formatted_query = f"""
You are an Document Search Agent build by 169pi Team. You know nothing and have no knowledge about the world. 

Your name is Pi RAG.

Here is the user_query: {query}

You have an access to following tools:
chat_history_tool: {chat_history}
document_data_tool: {relevant_text}
background_information_tool: {background_info}

tool description:
chat_history_tool description: This tool has an access to previous message history data. Use this tool only if the user_query is contextualy linked and similar to chat_history data.
document_data_tool description: This tool has an access to extracted document_data based on user_query.

Follow the below Instruction:
* Understand the user_query and decide which tool you need to use (based on tool description) to answer the given user_query.
* ONLY used the tools to answer user_query.
* IF the user wants to know about the Pi RAG, asking questions like "What can you do for me?", "Who are the members involved in building you?", "Which LLM you are using" etc. THEN ONLY use background_information_tool to answer the user_query.
* IF the user_query has open-ended questions like eg:- 'Summarize the pdf/doc' THEN answer should be 'Please specify the name of the document or any specific topic from it to answer'
* IF you don't find the answer in document_data_tool then DO NOT hallucinate just say 'I am unsure about the answer!'.
* ONLY use the above tools to answer the user_query.
* IF you find the answer in chat_history_tool then do not use the answer as it is. It should be identical.
* Don't start your answer with PREAMBLE or Any introductory statements like 'Document_data_tool', 'Here is the answer', 'Answer:', 'Begin!'. start right away.
* Answer the user's query in a professional but warm and conversational tone. Provide clear, concise information while maintaining a human touch.
* makesure you generate the answer in markdown format (never use numbered list)
* Ensure that your answer is primarily based on the tool output.
* DO NOT add any Note at the end of the answer eg: 'Note: As I don't find any infomration about the requestion information in document_data_tool or chat_history_tool and background tool information not have an access to requested data and user request having contextual meaning I don't have any more relevant answer to the request' This is a bad user experience never add something like this.

Answer:
Begin!
"""
    # Generate completion
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": formatted_query,
            }
        ],
        model="llama-3.1-70b-versatile",
        max_tokens=4096,
        stream=True,
    )
    
    for chunk in chat_completion:
        if chunk.choices[0].delta.content is not None:
            yield chunk.choices[0].delta.content

def rephrase_prompt(prompt, chat_history):
    prompt = f""" You are a intelligent and pleasant AI assistant in conversation with your boss. Your job is to provide a rephrased question that captures the essence of the conversation and can take it forward. However as an intelligent AI agent you understand when the context/topic of conversation is changed and keep the new question as the original question.
    Given the following Chat History and a follow up question, rephrase the follow up question into a standalone question that only enriches the follow up question and does not lose the essence of the followup question. DO NOT add any explanation or reasoning about the contextual link.
    Chat History:
    {chat_history}

    You also NEED TO IDENTIFY in WHICH LANGUAGE  the Conclusion is to be provided. (it is the language the Follow Up Question is written in by the user.  IN CASE OF AMBIGUITY IN LANGUAGE USE ENGLISH AS DEFAULT)

    You also understand if there is no significant contextual link or similarity to the chat history, repeating the "Follow Up Question" as the "Standalone question" is the best way. Contextual link means that no information from the chat history is relevant for creating the standalone question. DO NOT exclude any information and statements provided in the "Follow Up Question"
    Follow Up Question : {prompt}
    
    Return a json object with the following structure:
                        
        - question:  <standalone question>
        - language: <LANGUAGE>

    """

    client = Groq(api_key=os.getenv('GROQ_API_KEY'))
    messages = [{'role': 'user', 'content': prompt}]
    response = client.chat.completions.create(
        model='llama-3.1-70b-versatile',
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0.5
    )
    return json.loads(response.choices[0].message.content)

def handle_user_input(user_question, chat_history):
    # Retrieve relevant documents based on the query
    retrieved_docs = load_and_retrieve(user_question)
    relevant_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

    if len(chat_history) > 10:
        rephrased_prompt = rephrase_prompt(user_question, chat_history[-10:])
        with st.chat_message("assistant", avatar = '🤖'):
            response = st.write_stream(generate_response(relevant_text, rephrased_prompt['question'], chat_history))
    else:
        rephrased_prompt = rephrase_prompt(user_question, chat_history)
        with st.chat_message("assistant", avatar = '🤖'):
            response = st.write_stream(generate_response(relevant_text, rephrased_prompt['question'], chat_history))
    
    # Save the full response to session state once streaming is complete
    st.session_state.messages.append({
        "role": "assistant",
        "avatar": '🤖',
        "content": response
    })


def main():
    index_placeholder = None
    st.set_page_config("Pi RAG", layout="centered")
    st.header("Chat with PDF using Pi RAG")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "activate_chat" not in st.session_state:
        st.session_state.activate_chat = False

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar = message['avatar']):
            st.markdown(message["content"])


    with st.sidebar:
        st.subheader('Upload Your PDF File')
        pdf_docs = st.file_uploader("⬆️ Upload your PDF & Click to process", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Extracting Text..."):
                print('Extracting Text...')
                raw_text = get_pdf_text(pdf_docs)
            with st.spinner("Chunking & Creating doc..."):
                print('Chunking & Creating doc...')
                docs = create_documents(raw_text)
            with st.spinner("Saving Vector DB..."):
                print('Saving Vector DB...')
                save_vector_store(docs)
                st.success("Done")
            
            st.session_state.activate_chat = True

    if st.session_state.activate_chat == True:
        if prompt := st.chat_input("Ask your question from the PDF?"):
            with st.chat_message("user", avatar = '👨🏻'):
                st.markdown(prompt)
            st.session_state.messages.append({"role": "user", 
                                              "avatar" :'👨🏻',
                                              "content": prompt})

            if prompt:
                handle_user_input(prompt, st.session_state.messages)
            



if __name__ == "__main__":
    main()