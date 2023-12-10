import streamlit as st
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType

# Defina aqui o caminho do seu arquivo JSON e a chave da API da OpenAI
caminho_do_arquivo_json = '/workspaces/langchain-ask-the-data/pages/alysson_noticias.json'


# Título da página
st.set_page_config(page_title='🦜🔗 Pergunte aos Dados App')
st.title('🦜🔗 Pergunte aos Dados App')

# Carregar arquivo JSON
def carregar_json(arquivo_json):
  df = pd.read_json(arquivo_json)
  with st.expander('Ver DataFrame'):
    st.write(df)
  return df

# Gerar resposta do LLM
def gerar_resposta(consulta_entrada):
  llm = ChatOpenAI(model_name='gpt-3.5-turbo-0613', 
                   temperature=0.2, 
                   openai_api_key='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',
                   streaming=True)
  
  
  df = carregar_json(caminho_do_arquivo_json)
  # Criar Agente DataFrame do Pandas
  agente = create_pandas_dataframe_agent(llm, df, verbose=True, agent_type=AgentType.OPENAI_FUNCTIONS)
  # Realizar consulta usando o Agente
  resposta = agente.run(consulta_entrada)
  return st.success(resposta)

# Widgets de entrada para consulta
question_list = [
  'Quantas linhas existem?',
  'Qual é o intervalo de valores para uma coluna específica?',
  'Quantas linhas têm um valor específico em uma coluna.',
  'Outro']
texto_consulta = st.selectbox('Selecione uma consulta de exemplo:', question_list)

if texto_consulta == 'Outro':
  texto_consulta = st.text_input('Digite sua consulta:', placeholder='Digite a consulta aqui...')

if texto_consulta:
  st.header('Saída')
  gerar_resposta(texto_consulta)
