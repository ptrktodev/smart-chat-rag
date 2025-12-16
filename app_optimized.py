import streamlit as st
from openai_tts import gerar_audio_openai
from rag import split_text, create_collection, connect_to_collection, gerar_string_aleatoria
from funcoes import get_end_session_id, get_session_history, create_new_session, trimmer
from langchain_groq.chat_models import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

MODEL_70B = "llama-3.3-70b-versatile"
MODEL_8B = "llama-3.1-8b-instant"

llm8b = ChatGroq(model=MODEL_8B, groq_api_key=st.secrets["LLM"]["GROQ_API_KEY"])
llm70b = ChatGroq(model=MODEL_70B, groq_api_key=st.secrets["LLM"]["GROQ_API_KEY"])

chat_template = ChatPromptTemplate.from_messages([
    ('system', "Você é um assistente útil, amigável, casismático e gentil em ajudar o usuário com sua necessidade."),
    MessagesPlaceholder(variable_name="history"),
    ('human', "{prompt}"),
])

rag_template = ChatPromptTemplate([
    ("system", """
    Você é um assistente útil.
    Não traduza a pergunta para o usuário, apenas responda diretamente em PT-BR.

    Utilize o seguinte Contexto para responder a pergunta do usuário: {context}"""),
    MessagesPlaceholder(variable_name="history"),
    ("user", "{prompt}")
])

def get_llm(selection):
    return llm70b if selection == 0 else llm8b

def create_chat_chain(llm):
    return (
        {
            "prompt": lambda x: x["prompt"],
            "history": lambda x: trimmer().invoke(x["history"])
        }
        | chat_template
        | llm
    )

def create_rag_chain(llm):
    return (
        {
            "prompt": lambda x: x["prompt"],
            "history": lambda x: trimmer().invoke(x["history"]),
            "context": lambda x: x["context"]
        }
        |rag_template
        | llm
        | StrOutputParser()
    )

def generate_response(prompt, selection, temperature, context=None, is_rag=False):
    temperature = temperature / 100
    
    if is_rag:
        llm = get_llm(selection)
        chain = create_rag_chain(llm)
        input_data = {'prompt': prompt, 'context': context}
    else:
        llm = get_llm(selection)
        chain = create_chat_chain(llm)
        input_data = {'prompt': prompt}

    runnable_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="prompt",
        history_messages_key="history"
    )

    response = runnable_with_history.invoke(
        input_data,
        config={
            'configurable': {
                'session_id': st.session_state.session_id
            }
        },
        temperature=temperature
    )
    
    return response if is_rag else response.content

def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = create_new_session()
    if 'bool_toggle' not in st.session_state:
        st.session_state.bool_toggle = False
    if 'name_bd_rag' not in st.session_state:
        st.session_state.name_bd_rag = ''
    if 'summary' not in st.session_state:
        st.session_state.summary = False
    if 'audio_state' not in st.session_state:
        st.session_state.audio_state = None

def setup_sidebar():
    uploaded_file_status = 0 
    
    with st.sidebar:
        if st.button("Nova Sessão"):
            st.session_state.messages = []
            st.session_state.session_id = create_new_session()
            st.session_state.bool_toggle = False
            st.session_state.summary = False
            st.session_state.audio_content = None
            st.rerun()

        st.header("Modelo")
        option_map = {0: "70b", 1: "8b"}
        selection = st.segmented_control(
            "LLaMA (8b):",
            options=option_map.keys(),
            format_func=lambda option: option_map[option],
            selection_mode="single",
            default=0 
        )

        temperature = st.slider("Temperatura:", 0, 100, 20)

        uploaded_file = st.file_uploader(
            label="Escolha um arquivo:",
            type=["pdf"],
            help="Apenas arquivos PDF são permitidos"
        )

        if uploaded_file is None:
            uploaded_file_status = 0
            st.session_state.bool_toggle = False
            st.session_state.summary = False
            st.session_state.audio_content = None
        else:
            uploaded_file_status = 1
            col1, col2 = st.columns(2)

            with col1:
                if st.button("Aplicar RAG"):
                    st.session_state.name_bd_rag = gerar_string_aleatoria(10)
                    chunks = split_text(uploaded_file)
                    create_collection(chunks, st.session_state.name_bd_rag)
                    st.session_state.bool_toggle = True
                    st.session_state.summary = True
                    st.session_state.audio_content = None
                    st.rerun()

            with col2:
                if st.session_state.bool_toggle:
                    if st.toggle("Habilitar RAG"):
                        uploaded_file_status = 2

            if st.session_state.summary:
                if st.button("Resumir"):
                    with st.spinner("Wait for it..."):
                        text = "Resuma de forma breve o texto"
                        context = "\n\n".join(doc.page_content for doc in connect_to_collection(st.session_state.name_bd_rag).similarity_search(text, k=4))
                        summary = generate_response(text, selection, temperature, context, is_rag=True)
                        audio = gerar_audio_openai(summary)
                        if audio.status_code == 200:
                            st.session_state.audio_content = audio.content
                        else: 
                            st.write('Erro inesperado ao gerar o áudio.')
            
                if st.session_state.audio_content:
                    st.audio(st.session_state.audio_content, autoplay=True)

    return selection, temperature, uploaded_file_status

def main():
    st.set_page_config(page_title="Chat App") 
    
    st.markdown("""
    <style>
    div[data-testid="stButton"] > button:first-child {
        background-color: #fa5151ff;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

    init_session_state()
    
    selection, temperature, uploaded_file_status = setup_sidebar()
    
    messages_container = st.container(border=False) 
    
    for msg in st.session_state.messages:
        messages_container.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Digite aqui..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        messages_container.chat_message("user").write(prompt)
        
        with st.spinner("Wait for it..."):
            response_content = None
            
            if uploaded_file_status == 2: 
                context = "\n\n".join(doc.page_content for doc in connect_to_collection(st.session_state.name_bd_rag).similarity_search(prompt, k=4))
                response_content = generate_response(prompt, selection, temperature, context=context, is_rag=True)
            else: 
                response_content = generate_response(prompt, selection, temperature, is_rag=False)
            
            if response_content:
                st.session_state.messages.append({"role": "assistant", "content": response_content})
                messages_container.chat_message("assistant").write(response_content)

if __name__ == "__main__":
    main()
