from collections import Counter
from datetime import timedelta
from openai import OpenAI


import streamlit as st
import os
import pandas as pd
import plotly.express as px
import re
import json
import emoji


def plot_posts_over_time_plotly(df):
  # Garantindo que a coluna 'Data' seja do tipo datetime
  df['Data'] = pd.to_datetime(df['Data'])

  # Calculando o número de comentários por post
  df['NumComentarios'] = df['Comentarios'].apply(len)

  # Agrupando por data
  posts_per_day = df.groupby(df['Data'].dt.date).agg({
      'Curtidas': 'sum',
      'NumComentarios': 'sum'
  })

  # Criando o gráfico de linha
  fig = px.line(posts_per_day,
                y=['Curtidas', 'NumComentarios'],
                labels={
                    'value': 'Quantidade',
                    'variable': 'Categoria'
                })
  fig.update_layout(title='Curtidas e Comentários ao Longo do Tempo',
                    xaxis_title='Data',
                    yaxis_title='Quantidade')
  return fig


def plot_most_common_words_in_comments_plotly(df, num_words=10):
  # Concatenando todos os textos dos comentários
  all_comments = ' '.join([
      comment['Comentario'] for comments in df['Comentarios']
      for comment in comments
  ])

  # Encontrando palavras com mais de 4 letras
  words = re.findall(r'\b\w{4,}\b', all_comments.lower())

  # Contando e pegando as mais comuns
  most_common_words = Counter(words).most_common(num_words)

  # Criando o gráfico de barras
  words, counts = zip(*most_common_words)
  fig = px.bar(x=words, y=counts, labels={'x': 'Palavras', 'y': 'Contagem'})
  fig.update_layout(title='Palavras Mais Comuns nos Comentários')
  return fig


def top_posts(df, metric='Curtidas', top_n=5, ascending=True):
  # Calculando o número de comentários para cada post
  df['NumComentarios'] = df['Comentarios'].apply(len)

  # Adicionando uma coluna com o índice da linha como ID
  df['PostID'] = df.index

  # Ordenando os posts com base na métrica especificada
  if metric == 'Comentarios':
      sorted_df = df.sort_values(by='NumComentarios', ascending=ascending)
  else:
      sorted_df = df.sort_values(by=metric, ascending=ascending)

  # Selecionando as colunas relevantes e retornando os top N posts
  return sorted_df[['PostID', 'Data', 'Texto', 'Curtidas', 'NumComentarios']].head(top_n)



def plot_comment_classification_plotly(df):
  # Extraindo todas as classificações dos comentários
  all_classifications = [
      comment.get('Classificacao', 'Não Classificado')
      for comments in df['Comentarios'] if comments for comment in comments
  ]

  # Contando os valores de classificação dos comentários
  classification_counts = pd.Series(
      all_classifications).value_counts().reset_index()
  classification_counts.columns = ['Classificação', 'Contagem']

  # Criando o gráfico de pizza
  fig = px.pie(classification_counts,
               values='Contagem',
               names='Classificação',
               title='Classificação dos Comentários')
  return fig


def plot_top_comment_authors(df, top_n=10):
  # Extraindo os autores dos comentários
  all_authors = [
      comment['Autor'] for comments in df['Comentarios']
      for comment in comments
  ]

  # Contando as ocorrências de cada autor
  author_counts = pd.Series(all_authors).value_counts().head(
      top_n).reset_index()
  author_counts.columns = ['Autor', 'Contagem']

  # Criando o gráfico de barras
  fig = px.bar(author_counts,
               x='Autor',
               y='Contagem',
               title='Principais Autores de Comentários')
  return fig


def data_summary(df):
  # Conversão da coluna 'Data' para datetime
  df['Data'] = pd.to_datetime(df['Data'])

  # Total de curtidas
  total_curtidas = df['Curtidas'].sum()

  # Total de comentários
  total_comentarios = sum(len(comments) for comments in df['Comentarios'])

  # Número de autores únicos
  unique_authors = set(comment['Autor'] for comments in df['Comentarios']
                       for comment in comments)
  num_unique_authors = len(unique_authors)

  # Número de dias analisados
  num_dias = (df['Data'].max() -
              df['Data'].min()).days + 1  # +1 para incluir ambos os dias

  # Número de semanas analisadas (considerando 7 dias por semana)
  num_semanas = num_dias / 7

  # Número de posts analisados
  num_posts = len(df)

  # Médias
  media_publicacoes_por_semana = num_posts / num_semanas
  media_curtidas_por_publicacao = total_curtidas / num_posts
  media_comentarios_por_publicacao = total_comentarios / num_posts

  # Criando um resumo em formato de dicionário
  summary = {
      '📅 Número de Dias Analisados': num_dias,
      '📊 Média de Publicações por Semana':
      f"{media_publicacoes_por_semana:.2f}",
      '❤️ Média de Curtidas por Publicação':
      f"{media_curtidas_por_publicacao:.2f}",
      '💬 Média de Comentários por Publicação':
      f"{media_comentarios_por_publicacao:.2f}",
      '👤 Número de Autores Únicos': num_unique_authors,
      '📝 Total de Posts Analisados': num_posts,
      '👍 Total de Curtidas': total_curtidas,
      '💬 Total de Comentários': total_comentarios
  }

  return summary


def identify_top_authors_by_sentiment(df, top_n=10):
  # Verificar se a coluna 'Comentarios' existe no DataFrame
  if 'Comentarios' not in df.columns:
    print("A coluna 'Comentarios' não existe no DataFrame.")
    return pd.DataFrame(), pd.DataFrame()  # Retornar DataFrames vazios

  author_sentiments = []

  for comments in df['Comentarios']:
    for comment in comments:
      autor = comment.get('Autor', 'Desconhecido')
      sentimento = comment.get('Sentimento', 0)
      author_sentiments.append((autor, sentimento))

  df_author_sentiments = pd.DataFrame(author_sentiments,
                                      columns=['Autor', 'Sentimento'])

  # Agrupar por autor, somar os sentimentos e contar os comentários
  df_soma_contagem = df_author_sentiments.groupby('Autor').agg({
      'Sentimento': 'sum',
      'Autor': 'count'
  }).rename(columns={
      'Autor': 'NumComentarios'
  }).reset_index()

  top_positive_authors = df_soma_contagem[
      df_soma_contagem['Sentimento'] > 0].sort_values(
          by='Sentimento', ascending=False).head(top_n)
  top_negative_authors = df_soma_contagem[
      df_soma_contagem['Sentimento'] < 0].sort_values(
          by='Sentimento').head(top_n)

  return top_positive_authors, top_negative_authors


def count_authors_by_sentiment(df):
  author_classifications = {
      'positivo': set(),
      'negativo': set(),
      'neutro': set()
  }

  for comments in df['Comentarios']:
    for comment in comments:
      try:
        autor = comment.get('Autor', 'Desconhecido')
        texto_comentario = comment.get('Texto', '')

        # Verifica se o comentário é composto apenas por emojis
        if all(char in emoji.UNICODE_EMOJI_ENGLISH
               for char in texto_comentario):
          classificacao = 'neutro'
        else:
          sentimento = comment.get('Sentimento')
          if sentimento is None:
            classificacao = 'neutro'
          elif sentimento > 0:
            classificacao = 'positivo'
          else:
            classificacao = 'negativo'

        author_classifications[classificacao].add(autor)
      except KeyError as e:
        print(f"Erro ao processar o comentário: {e}")

  # Contar o número de autores em cada categoria
  author_counts = {k: len(v) for k, v in author_classifications.items()}
  return author_counts


def extract_hashtags_and_profiles(df):
  hashtag_pattern = re.compile(r"#(\w+)")
  profile_pattern = re.compile(r"@(\w+)")

  hashtags = Counter()
  profiles = Counter()

  for _, row in df.iterrows():
    # Verifica se os campos são strings e, em seguida, extrai as hashtags e perfis
    if isinstance(row['Texto'], str):
      hashtags.update(hashtag_pattern.findall(row['Texto']))
      profiles.update(profile_pattern.findall(row['Texto']))

    # Extrair hashtags e perfis de cada comentário
    for comment in row['Comentarios']:
      if isinstance(comment['Comentario'], str):
        hashtags.update(hashtag_pattern.findall(comment['Comentario']))
        profiles.update(profile_pattern.findall(comment['Comentario']))

  return hashtags, profiles


def create_bar_chart(counter, title, top_n=10):
  # Selecionar os top_n itens mais comuns
  data = pd.DataFrame(counter.most_common(top_n), columns=['Item', 'Contagem'])

  # Criar gráfico de barras
  fig = px.bar(data, x='Item', y='Contagem', title=title)

  return fig


def calcular_engajamento(post, df_maximos):
  score_sentimento = sum(
      comment.get('Sentimento', 0) for comment in post['Comentarios']) / len(
          post['Comentarios']) if post['Comentarios'] else 0
  num_curtidas = post['Curtidas']

  # Normalizando o score de sentimento e o número de curtidas
  score_sentimento_normalizado = score_sentimento / df_maximos[
      'max_sentimento'] if df_maximos['max_sentimento'] else 0
  num_curtidas_normalizado = num_curtidas / df_maximos[
      'max_curtidas'] if df_maximos['max_curtidas'] else 0

  # Calcular o índice de engajamento normalizado entre -1 e 1
  indice_engajamento = score_sentimento_normalizado + num_curtidas_normalizado
  indice_engajamento = max(min(indice_engajamento, 1), -1)

  # Classificar como engajamento positivo ou negativo
  classificacao = "Positivo" if indice_engajamento > 0 else "Negativo" if indice_engajamento < 0 else "Neutro"
  return indice_engajamento, classificacao


def criar_grafico_engajamento_cronologico(df):
  # Supondo que 'Data' seja a coluna com as datas e 'IndiceEngajamento' a coluna com os índices de engajamento
  fig = px.line(df,
                x='Data',
                y='IndiceEngajamento',
                title='Desempenho do Engajamento ao Longo do Tempo')
  return fig

def load_json_to_dataframe(file_path):
  # Carregando o arquivo JSON
  with open(file_path, 'r') as file:
      json_data = json.load(file)
  
  # Convertendo o JSON em DataFrame e ignorando a primeira linha
  df = pd.DataFrame(json_data).iloc[1:]
  
  # Verifica se a coluna 'Data' existe no DataFrame
  if 'Data' not in df.columns:
      # Tratar ou criar a coluna 'Data', se necessário
      # Por exemplo, criar uma coluna 'Data' com valores padrão
      # df['Data'] = pd.Timestamp('hoje')  # ou qualquer outro valor padrão
      pass
  
  return df
  


def comparar_sumarios(label, summary, df):
  client = OpenAI(
      api_key='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',  # Substitua por sua chave de API
  )

  # Gerando o nome do arquivo com base no rótulo
  arquivo_saida = f"{label}.txt"

  # Verificando se o arquivo já existe (cache)
  if os.path.exists(arquivo_saida):
    with open(arquivo_saida, 'r') as arquivo:
      return arquivo.read()

  # Certifique-se de que df é um DataFrame
  if not isinstance(df, pd.DataFrame):
    raise ValueError("O objeto df deve ser um DataFrame do pandas.")

  # Calcular os valores máximos de sentimento e curtidas
  max_sentimento = max(
      abs(comment.get('Sentimento', 0)) for post in df['Comentarios']
      for comment in post if post)
  max_curtidas = df['Curtidas'].max()
  df_maximos = {'max_sentimento': max_sentimento, 'max_curtidas': max_curtidas}

  df['Data'] = pd.to_datetime(df['Data'])

  # Adicionar índice de engajamento e classificação ao DataFrame
  df['IndiceEngajamento'], df['ClassificacaoEngajamento'] = zip(
      *df.apply(lambda post: calcular_engajamento(post, df_maximos), axis=1))

  # Selecionando os 3 posts com maior e menor curtidas
  top_3 = df.nlargest(3, 'Curtidas')
  bottom_3 = df.nsmallest(3, 'Curtidas')

  # Selecionando os 3 posts com maior e menor IndiceEngajamento
  top_3_en = df.nlargest(3, 'IndiceEngajamento')
  bottom_3_en = df.nsmallest(3, 'IndiceEngajamento')

  # Preparando os dados para o prompt
  top_3_dados = "\n".join(
      f"Post: {row['Texto']}, Índice de Curtidas: {row['Curtidas']}"
      for index, row in top_3.iterrows())
  # Preparando os dados para o prompt
  top_3_dados_en = "\n".join(
      f"Post: {row['Texto']}, Índice de Engajamento: {row['IndiceEngajamento']}"
      for index, row in top_3_en.iterrows())

  bottom_3_dados_en = "\n".join(
      f"Post: {row['Texto']}, Índice de Engajamento: {row['IndiceEngajamento']}"
      for index, row in bottom_3_en.iterrows())

  top_negative = top_negative_comments(df)
  top_positive = top_positive_comments(df)

  bottom_3_dados = "\n".join(
      f"Post: {row['Texto']}, Índice de Curtidas: {row['Curtidas']}"
      for index, row in bottom_3.iterrows())

  prompt = f"""
  Perfil {label} - Analise o resumo e os posts com maior e menor engajamento:

  Resumo: {summary}

  Posts com Maior Curtidas:
  {top_3_dados}

  Posts com Maior Engajamento:
  {top_3_dados_en}

  Posts com Menor Curtidas:
  {bottom_3_dados}

  Posts com Menor Engajamento:
  {bottom_3_dados_en}

  Comentários Positivos
  {top_positive}

  Comentários Negativos
  {top_negative}

  Com base nesses resumos de análise de dados de redes sociais, faça um relatório detalhando os dados ali destacados e como isso influencia no engajamento nas redes e o que pode ser feito para melhorar. Utilize a linguagem markdown para destacar pontos importantes.Com uma média de {summary['❤️ Média de Curtidas por Publicação']} e {summary['💬 Média de Comentários por Publicação']} comentários por publicação, qual é o nível de engajamento em relação a outros períodos de {summary['📅 Número de Dias Analisados']} dias? Isso representa um aumento ou diminuição do engajamento?
Há dias específicos da semana ou horários que mostram maior engajamento? Qual a correlação entre a frequência de postagens e o engajamento? Mais postagens levam a mais engajamento ou há um ponto de saturação?
  """

  chat_completion = client.chat.completions.create(
      messages=[
          {
              "role":
              "system",
              "content":
              "Você é uma analista-sênior de ciência de dados, especialista em comunicação social e do jornalismo político. Se mantenha discreto sobre suas habilidades. O propósito desse relatório é guiar a assessoria de imprensa. Escreva com tom didático. Utilize a linguagem markdown para criar o texto da sua resposta, usando e abusando de forma criativa de títulos, negritos, links, emoticons e listas. Finalize apontando um caminho a ser seguido para manter o engajamento e a popularidade em alta.não reproduza a variavel sumario por completo isso é para analise interna"
          },
          {
              "role": "user",
              "content": prompt
          },
      ],
      model="gpt-4",
  )

  resultado = chat_completion.choices[0].message.content

  # Salvando o resultado no arquivo
  with open(arquivo_saida, 'w') as arquivo:
    arquivo.write(resultado)

  return resultado


def top_negative_comments(df, top_n=20):
  # Lista para armazenar detalhes dos comentários
  comments_details = []

  # Certifique-se de que df é um DataFrame
  if not isinstance(df, pd.DataFrame):
    raise ValueError("O objeto df deve ser um DataFrame do pandas.")

  # Regex para identificar emojis
  emoji_regex = re.compile(
      "["
      u"\U0001F600-\U0001F64F"  # emoticons
      u"\U0001F300-\U0001F5FF"  # símbolos & pictogramas
      u"\U0001F680-\U0001F6FF"  # transporte & símbolos do mapa
      u"\U0001F1E0-\U0001F1FF"  # bandeiras (iOS)
      u"\U00002500-\U00002BEF"  # caracteres chineses
      u"\U00002702-\U000027B0"
      u"\U00002702-\U000027B0"
      u"\U000024C2-\U0001F251"
      u"\U0001f926-\U0001f937"
      u"\U00010000-\U0010ffff"
      u"\u2640-\u2642"
      u"\u2600-\u2B55"
      u"\u200d"
      u"\u23cf"
      u"\u23e9"
      u"\u231a"
      u"\ufe0f"  # dingbats
      u"\u3030"
      "]+",
      flags=re.UNICODE)

  # Verificar se a coluna 'Comentarios' existe no DataFrame
  if 'Comentarios' not in df.columns:
    print("A coluna 'Comentarios' não existe no DataFrame.")
    return pd.DataFrame()  # Retorna um DataFrame vazio

  comments_details = []

  for comments in df['Comentarios']:
    for comment in comments:
      try:
        autor = comment.get('Autor', 'Desconhecido')
        comentario = comment.get('Comentario', '')
        sentimento = comment.get('Sentimento', 0)
        comments_details.append((autor, comentario, sentimento))
      except KeyError as e:
        print(f"Chave não encontrada no comentário: {e}")

  for comments in df['Comentarios']:
    for comment in comments:
      try:
        # Extraindo autor, comentário e sentimento
        autor = comment.get('Autor', 'Desconhecido')
        comentario = comment.get('Comentario', '')
        sentimento = comment.get('Sentimento',
                                 0)  # Valor padrão se 'Sentimento' não existir

        # Verifica se o comentário é composto apenas por emojis
        if emoji_regex.fullmatch(comentario):
          sentimento = 0  # Classifica como neutro se for apenas emojis

        comments_details.append((autor, comentario, sentimento))
      except KeyError as e:
        # Tratando o caso em que uma chave esperada não está presente
        print(f"Chave não encontrada no comentário: {e}")

  top_negative = sorted(comments_details, key=lambda x: x[2])[:top_n]
  return pd.DataFrame(top_negative,
                      columns=['Autor', 'Comentario', 'Sentimento'])


def top_positive_comments(df, top_n=10):
  # Verificar se a coluna 'Comentarios' existe no DataFrame
  if 'Comentarios' not in df.columns:
    print("A coluna 'Comentarios' não existe no DataFrame.")
    return pd.DataFrame(columns=['Autor', 'Comentario',
                                 'Sentimento'])  # Retorna um DataFrame vazio

  comments_details = []

  for comments in df['Comentarios']:
    for comment in comments:
      try:
        autor = comment.get('Autor', 'Desconhecido')
        comentario = comment.get('Comentario', '')
        sentimento = comment.get('Sentimento', 0)

        comments_details.append((autor, comentario, sentimento))
      except KeyError as e:
        print(f"Chave não encontrada no comentário: {e}")

  # Ordenando pela pontuação de sentimento (mais alto primeiro)
  top_positive = sorted(comments_details, key=lambda x: x[2],
                        reverse=True)[:top_n]

  return pd.DataFrame(top_positive,
                      columns=['Autor', 'Comentario', 'Sentimento'])


def analisar_e_salvar_dataframe(df, label):
  client = OpenAI(
      api_key=os.environ['openai'])  # Substitua com sua chave de API

  arquivo_saida = f"{label}_.txt"

  # Certifique-se de que df é um DataFrame
  if not isinstance(df, pd.DataFrame):
    raise ValueError("O objeto df deve ser um DataFrame do pandas.")

  # Suponha que as funções identify_top_authors_by_sentiment, top_negative_comments e top_positive_comments
  # já estejam definidas e prontas para uso
  top_positives, top_negatives = identify_top_authors_by_sentiment(df,
                                                                   top_n=10)
  top_negative = top_negative_comments(df)
  top_positive = top_positive_comments(df)

  # Verificando se o arquivo já existe (cache)
  if os.path.exists(arquivo_saida):
    with open(arquivo_saida, 'r') as arquivo:
      return arquivo.read()

  # Criar DataFrame com detalhes dos comentários
  if 'Comentarios' in df.columns and 'Texto' in df.columns:
    comments_details = []
    for index, row in df.iterrows():
      texto_postagem = row['Texto']
      for comment in row['Comentarios']:
        autor = comment.get('Autor', 'Desconhecido')
        comentario = comment.get('Comentario', '')
        data = row['Data']
        indice_engajamento = row['IndiceEngajamento']
        classificacao_engajamento = row['ClassificacaoEngajamento']
        comments_details.append((data, texto_postagem, indice_engajamento,
                                 classificacao_engajamento, autor, comentario))

    df_comentarios = pd.DataFrame(comments_details,
                                  columns=[
                                      'Data', 'TextoPostagem',
                                      'IndiceEngajamento',
                                      'ClassificacaoEngajamento', 'Autor',
                                      'Comentario'
                                  ])
    df_ordenado = df_comentarios.sort_values(by='IndiceEngajamento',
                                             ascending=False)
    top_10 = df_ordenado.head(10)
  else:
    print("As colunas 'Comentarios' ou 'Texto' não existem no DataFrame.")
    top_10 = pd.DataFrame()

  # Convertendo top_10 para string
  dados_para_analise = "\n".join(
      f"Data: {row['Data']}, Texto: {row['TextoPostagem']}, Índice de Engajamento: {row['IndiceEngajamento']}, Classificação de Engajamento: {row['ClassificacaoEngajamento']}, Autor: {row['Autor']}, Comentário: {row['Comentario']}"
      for index, row in top_10.iterrows())

  prompt = f"""Analise as seguintes publicações e seus engajamentos:\n\n{dados_para_analise}\n\nIdentifique quais assuntos estão gerando mais engajamento e explique o porquê. Comentários de autores positivos: {top_positives}, autores negativos: {top_negatives}, comentários positivos: {top_positive}, comentários negativos: {top_negative}."""

  chat_completion = client.chat.completions.create(
      model="gpt-4",
      messages=[{
          "role":
          "system",
          "content":
          "Você é uma analista-sênior de ciência de dados, especialista em comunicação social e do jornalismo político. Se mantenha discreto sobre suas habilidades. O propósito desse relatório é guiar a assessoria de imprensa. Escreva com tom didático. Utilize a linguagem markdown para criar o texto da sua resposta, usando e abusando de forma criativa de títulos, negritos, links, emoticons e listas. Com base nos padrões observados, que tipo de conteúdo parece ressoar mais com o público? Existem lacunas ou oportunidades no tipo de conteúdo que está sendo postado atualmente? Qual a correlação entre a frequência de postagens e o engajamento? Mais postagens levam a mais engajamento ou há um ponto de saturação? Qual assunto está sendo tratados nos comentários? E necessário que se pontue sobre os comentarios negativos alertando para possíveis crises, apresente esses comentários e explique o porque."
      }, {
          "role": "user",
          "content": prompt
      }])

  resultado = chat_completion.choices[0].message.content

  # Salvando o resultado no arquivo
  with open(arquivo_saida, 'w') as arquivo:
    arquivo.write(resultado)

  return resultado


def plot_publication_frequency_plotly(df):
  # Certifique-se de que a coluna 'Data' seja do tipo datetime
  df['Data'] = pd.to_datetime(df['Data']).dt.date

  # Agrupar os dados por data e contar o número de publicações por data
  frequency = df.groupby('Data').size().reset_index(
      name='Número de Publicações')

  # Criar o gráfico usando Plotly
  fig = px.bar(frequency,
               x='Data',
               y='Número de Publicações',
               title='Frequência de Publicações por Data',
               labels={
                   'Data': 'Data',
                   'Número de Publicações': 'Número de Publicações'
               })
  fig.update_xaxes(tickangle=45)

  return fig


def exibir_posts_com_cards(df):
  # Criar três colunas
  cols = st.columns(3)

  # Distribuir os cards pelas colunas
  for i, post in enumerate(df.iterrows()):
      _, post = post
      col_index = i % 3
      col = cols[col_index]

      # Caminho da imagem
      post_id = post['PostID']
      image_file = os.path.join('downloaded_images', f'image_{post_id}.jpg')
      if not os.path.exists(image_file):
          image_file = 'path_to_default_image.jpg'  # Substitua pelo caminho da imagem padrão

      # Exibir imagem
      col.image(image_file, use_column_width=True)

      # Exibir detalhes do post
      col.write(f"Data: {post.get('Data', 'Sem Data')}")
      col.write(f"Texto: {post.get('Texto', '')[:140]}...")
      col.write(f"Curtidas: {post.get('Curtidas', 0)}")
      col.write(f"Comentários: {post.get('NumComentarios', 0)}")


def exibir_posts_com_cardss(df):
  # Injetar CSS para estilização dos cards e das imagens
  st.markdown("""
  <style>
  .card {
      box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
      transition: 0.3s;
      margin: 10px;
      background-color: #f9f9f9;
  }
  .card:hover {
      box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2);
  }
  .container {
      padding: 2px 16px;
  }
  .card img {
      width: 100%;
      height: auto;
      aspect-ratio: 3 / 4; /* Proporção 3x4 */
      object-fit: cover; /* Garante que a imagem cubra a área sem distorcer */
  }
  </style>
  """, unsafe_allow_html=True)

  # Criar três colunas
  cols = st.columns(3)

  # Distribuir os cards pelas colunas
  for i, post in enumerate(df.iterrows()):
      _, post = post
      col_index = i % 3
      col = cols[col_index]

      post_id = post['PostID']
      image_file = os.path.join('downloaded_images', f'image_{post_id}.jpg')
      image_path = image_file if os.path.exists(image_file) else 'path_to_default_image.jpg'

      # Usar HTML para criar o card, incluindo a imagem
      html_content = f"""
      <div class="card">
          <img src="{image_path}" alt="Post image">
          <div class="container">
              <small><b>{post.get('Data', 'Sem Data')}</b></small>
              <p>{post.get('Texto', '')[:140]}...</p>
              <p>Curtidas: {post.get('Curtidas', 0)}</p>
              <p>Comentários: {post.get('NumComentarios', 0)}</p>
          </div>
      </div>
      """
      with col:
          st.markdown(html_content, unsafe_allow_html=True)




def exibir_cards_autores_negativos(autores_negativos):
    # Injetar CSS para estilização dos cards
    st.markdown("""
    <style>
    .card {
        max-width: 18rem;
        border-radius: 5px;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
        transition: 0.3s;
        margin: 10px;
        background-color: #f0f0f0;
    }
    .card:hover {
        box-shadow: 0 8px 16px 0 rgba(0,0,0,0.4);
    }
    .card-header {
        background-color: #c9c9c9;
        color: black;
        padding: 10px;
        border-top-left-radius: 5px;
        border-top-right-radius: 5px;
        font-weight: bold;
    }
    .card-body {
        padding: 10px;
    }
    .card-title {
        color: #000;
        font-weight: bold;
    }
    .card-text {
        color: #666;
        
    }
    </style>
    """, unsafe_allow_html=True)

    # Criar três colunas
    cols = st.columns(3)

    # Distribuir os cards pelas colunas
    for i, row in autores_negativos.iterrows():
        col_index = i % 3
        col = cols[col_index]

        # Formatando o sentimento com até 4 dígitos após o ponto decimal
        formatted_sentimento = f"{row['Sentimento']:.4f}"

        # HTML para criar o card
        html_content = f"""
        <div class="card">
            <div class="card-header">@{row['Autor']}</div>
            <div class="card-body">
                <span class="card-text">🌡️ : {formatted_sentimento} || 💬 : {row['NumComentarios']}</span>
            </div>
        </div>
        """
        with col:
            st.markdown(html_content, unsafe_allow_html=True)