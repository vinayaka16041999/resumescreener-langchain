import time
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Cognine AI")
col1, col2, col3 = st.columns([1, 25, 1])
with col2:
    st.title("Welcome to Cognine AI's Resume Screener")

import os
api_key = "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"
os.environ['GROQ_API_KEY'] = api_key
from dotenv import load_dotenv
from numpy.core.defchararray import endswith
load_dotenv()

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
## LCEL implementation of LangChain ConversationalRetrievalChain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
root = "/app/"
doc_folder = root + "resumes"

vector_database_path = os.path.join(root, "resumes_vector_db")
chatmodel = ChatGroq(model="llama-3.2-90b-vision-preview", temperature=0.15)

uploaded_files = st.file_uploader(
    "Upload resumes", accept_multiple_files=True, type="pdf"
)
for file in uploaded_files:
    save_path = Path(doc_folder, file.name)
    with open(save_path, mode='wb') as w:
        w.write(file.getvalue())

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# resetting the entire conversation
def reset_conversation():
    st.session_state['messages'] = []

## open-source embedding model from HuggingFace - taking the default model only
embedF = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")

## loading the vector database from local
vectorDB = Chroma(embedding_function=embedF, persist_directory=vector_database_path)

kb_retriever = vectorDB.as_retriever(search_type="similarity",search_kwargs={"k": 10})

rephrasing_template = (
    """
        TASK: Convert context-dependent questions into standalone queries.

        INPUT: 
        - chat_history: Previous messages
        - question: Current user query

        RULES:
        1. Use the person's name from the available documents while referring to them
        2. Expand contextual phrases ("the above", "previous")
        3. Return original if already standalone
        4. Give justification for selection

        OUTPUT: Single reformulated question, preserving original intent and style.

        Example:
        History: "Let's shortlist Data Engineering candidates."
        Question: "Select me 2 best candidates"
        Returns: "Select 2 best Data Engineering candidates"
    """
)

rephrasing_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", rephrasing_template),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(
    llm = chatmodel,
    retriever = kb_retriever,
    prompt = rephrasing_prompt
)

system_prompt_template = (
    "As a Resume Sccreening assistant specializing in shortlisting candidates based on resumes, "
    "your primary objective is to shortlist candidates based on the resumes and give a summary of the selected candidates based on user queries. "
    "You will adhere strictly to the instructions provided, referring the resumes of only relevant candidates"
    "short summary of the selected candidates and justification for selection in no more than 10 sentences."
    "Your responses will be a short summary(no more than 4 sentences) of the information you find for each candidate. "
    "If a question falls outside the given context or if the candidate is not relevant, you will simply output that you are sorry and you don't know about this. "
    "The aim is to deliver professional, precise, and contextually relevant information pertaining to the context. "
    "Use ten sentences maximum."
    "\nCONTEXT: {context}"
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt_template),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ]
)

qa_chain = create_stuff_documents_chain(chatmodel, qa_prompt)
## final RAG chain
coversational_rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

## setting-up conversational UI

## printing all (if any) messages in the session_session `message` key
for message in st.session_state.messages:
    with st.chat_message(message.type):
        st.write(message.content)


user_query = st.chat_input("Let's get started! Ask me a question about the candidates you want to shortlist.")

if user_query:
    with st.chat_message("User"):
        st.write(user_query)

    with st.chat_message("Cognine AI"):
        with st.status("Thinking 💡...", expanded=True):
            ## invoking the chain to fetch the result
            result = coversational_rag_chain.invoke({"input": user_query, "chat_history": st.session_state['messages']})

            message_placeholder = st.empty()

            full_response = (
                "⚠️ **This information is a rough approximation.!!!** \n\n\n"
            )
        
        ## displaying the output on the dashboard
        for chunk in result["answer"]:
            full_response += chunk
            time.sleep(0.02) ## <- simulate the output feeling of ChatGPT

            message_placeholder.markdown(full_response + " ▌")
        st.button('Reset Conversation 🗑️', on_click=reset_conversation)
    ## appending conversation turns
    st.session_state.messages.extend(
        [
            HumanMessage(content=user_query),
            AIMessage(content=result['answer'])
        ]
    )