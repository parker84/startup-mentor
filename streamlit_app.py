from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
import pinecone
from dotenv import find_dotenv, load_dotenv
from decouple import config
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from streamlit_chat import message
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import streamlit as st
import coloredlogs, logging
import os
from constants import (
    DEFAULT_CHAT_MODEL,
    EMBEDDING_MODEL
)


# --------------setup
logger = logging.getLogger(__name__)
coloredlogs.install(level=config('LOG_LEVEL', default='INFO'))
st.set_page_config(page_title='Startup Mentor', page_icon='💻', initial_sidebar_state="auto", menu_items=None)
st.title("Startup Mentor 💻")

st.sidebar.subheader("Enter Your API Key 🗝️")
open_api_key = st.sidebar.text_input(
    "Open API Key", 
    value=st.session_state.get('open_api_key', ''),
    help="Get your API key from https://openai.com/",
    type='password'
)
os.environ["OPENAI_API_KEY"] = open_api_key
st.session_state['open_api_key'] = open_api_key
load_dotenv(find_dotenv())

with st.sidebar.expander('Advanced Settings ⚙️', expanded=False):
    open_ai_model = st.text_input('OpenAI Chat Model', DEFAULT_CHAT_MODEL, help='See model options here: https://platform.openai.com/docs/models/overview')   

def get_vectorstore():
    text_field = "text"
    embed = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    pinecone.init(
        api_key=config('PINECONE_API_KEY'),  # find api key in console at app.pinecone.io
        environment=config('PINECONE_ENV')  # find next to api key in console
    )
    index = pinecone.Index(config('PINECONE_INDEX_NAME'))

    vectorstore = Pinecone(
        index, embed.embed_query, text_field
    )

    return vectorstore

@st.cache_data
def get_response_from_question(_vectorstore, question, memory, k=5):
    docs = _vectorstore.similarity_search(question, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    chat = ChatOpenAI(model_name=open_ai_model, temperature=0)

    # Template to use for the system message prompt
    template = """
        You are an assistant to advice founders and employees in startups.

        You have access to the best books on startups / business and draw upon these books to
        give answers / advice / recommendations when asked.
        
        Here is relevant information from those books to  
        help you answer the following questions:
        
        {docs}
        
        Here is relevant information from the current conversation to help you answer the following question: {memory}
        
        Only use the factual information from these books to answer the question.
        
        If you feel like you don't have enough information to answer the question, say "Sorry, but I don't have enough information to give you a confident answer to that question".
        """

    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    # Human question prompt
    human_template = """
        Someone is is coming to you for advice, only advice based on the information from the books that
        have just been directly provided to you, do not make things up.
        Quote and reference these books directly as much as possible. 
        Here is there question: {question}
    """
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )
    chain = LLMChain(llm=chat, prompt=chat_prompt)

    response = chain.run(question=question, docs=docs_page_content, memory=memory)
    return response, docs

if 'questions' not in st.session_state:
    st.session_state['questions'] = []
if 'responses' not in st.session_state:
    st.session_state['responses'] = []

question = st.text_input(
    label="Ask Mentor a question",
    value="How can I secure a competitive moat for my business?"
)

if question != "" and (open_api_key == '' or open_api_key is None):
    st.error("⚠️ Please enter your API key in the sidebar")
else:
    vectorstore = get_vectorstore()
    if len(st.session_state['questions']) > 0:
        memory = '\n\n'.join(
            [
                f'Question: {q}\nAnswer: {a}'
                for q, a in zip(st.session_state['questions'], st.session_state['responses'])
            ]
        )
    else:
        memory = None
    response, docs = get_response_from_question(vectorstore, question=question, memory=memory, k=5)

    st.session_state['questions'].append(question)
    st.session_state['responses'].append(response)

    for i in range(len(st.session_state['questions'])):
        question = st.session_state['questions'][i]
        response = st.session_state['responses'][i]
        message(question, is_user=True)  # align's the message to the left
        message(response, is_user=False)  # align's the message to the right

