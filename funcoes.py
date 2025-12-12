from supabase import create_client, Client
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.messages import trim_messages
import streamlit as st
print(st.secrets)
supabase: Client = create_client(st.secrets["SUPABASE"]["SUPABASE_URL"], st.secrets["SUPABASE"]["SUPABASE_API_KEY"])

def get_end_session_id():
    response = supabase.table('message_store').select("*").execute()
    ids = [int(data['session_id'])for data in response.data]
    return max(ids)

def create_new_session():
    return str(get_end_session_id() + 1)

def get_session_history(session_id: str):
    # String de conexão PostgreSQL para Supabase
    connection_string = st.secrets["SUPABASE"]["SUPABASE_CONN_STRING"]
    
    return SQLChatMessageHistory(
        session_id=session_id,
        connection=connection_string,
        table_name="message_store"
    )

def trimmer():
    return trim_messages(
    max_tokens=10,  # cada mensagem conta como 1 token
    strategy="last",   # seleciona as últimas mensagens
    token_counter=lambda x: 1,  # Cada mensagem conta como 1 token
    )
