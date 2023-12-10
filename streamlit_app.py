import streamlit as st
import pandas as pd
import streamlit_extras
import instafunctions

# Função para criar a página com funcionalidades do streamlit-extras

# Caminho para o seu arquivo JSON
label = 'alysson_riobranco'
nomearquivo = f'{label}_processado.json'
file_path = nomearquivo


# Carregando o JSON no DataFrame
df = instafunctions.load_json_to_dataframe(file_path)



# Função para criar a página com funcionalidades do streamlit-extras
def criar_pagina_com_streamlit_extras(df):
  # Cabeçalho da página
  st.title("Análise de Dados do Perfil @alysson_riobranco")

  
  # Menus para navegação
  opcao = st.sidebar.selectbox(
      "Escolha uma opção",
      ("Visão Geral", "Engajamento e Curtidas", "Análise do Conteúdo", "Comentários"))

  # Exibição de posts como cards
  if opcao == "Visão Geral":
    st.subheader("Principais Métricas")
    
    with st.container():
      # No seu script Streamlit
      summary = instafunctions.data_summary(df)

     

      keys = list(summary.keys())
      num_cols = 4

      # Primeira linha
      row1_cols = st.columns(num_cols)
      for i in range(num_cols):
          with row1_cols[i]:
              # Utilizando o HTML para criar um container personalizado
              st.markdown(f"<div class='metric-container'>", unsafe_allow_html=True)
              st.metric(label=keys[i], value=summary[keys[i]])
              st.markdown("</div>", unsafe_allow_html=True)

      # Segunda linha
      if len(keys) > num_cols:
          row2_cols = st.columns(num_cols)
          for i in range(num_cols, min(len(keys), num_cols * 2)):
              with row2_cols[i - num_cols]:
                  st.markdown(f"<div class='metric-container'>", unsafe_allow_html=True)
                  st.metric(label=keys[i], value=summary[keys[i]])
                  st.markdown("</div>", unsafe_allow_html=True)

      st.title("Análise de Frequência de Publicações")

      # Chamando a função para criar o gráfico Plotly
      fig = instafunctions.plot_publication_frequency_plotly(df)

      # Exibindo o gráfico no Streamlit
      st.plotly_chart(fig)
      
      resultado = instafunctions.comparar_sumarios(label, summary, df)
      st.markdown(resultado)

  # Análise
  elif opcao == "Engajamento e Curtidas":
    # Exibindo um gráfico de posts ao longo do tempo
    st.header("Posts, Curtidas e Comentários ao Longo do Tempo")
    # Plotando o gráfico de posts ao longo do tempo
    fig1 = instafunctions.plot_posts_over_time_plotly(df)
    st.plotly_chart(fig1)
  
    # Calcular os valores máximos de sentimento e curtidas
    max_sentimento = max(
        abs(comment.get('Sentimento', 0)) for post in df['Comentarios']
        for comment in post if post)
    max_curtidas = df['Curtidas'].max()
    df_maximos = {'max_sentimento': max_sentimento, 'max_curtidas': max_curtidas}
  
    df['Data'] = pd.to_datetime(df['Data'])
  
    # Adicionar índice de engajamento e classificação ao DataFrame
    df['IndiceEngajamento'], df['ClassificacaoEngajamento'] = zip(*df.apply(
        lambda post: instafunctions.calcular_engajamento(post, df_maximos),
        axis=1))
  
    st.header("Engajamento das Publicações")
    # Criando o gráfico
    fig_engajamento = instafunctions.criar_grafico_engajamento_cronologico(df)
  
    # Exibindo o gráfico no Streamlit
    st.plotly_chart(fig_engajamento)
  
    st.markdown("""
        # Índice de Engajamento 📈
  
        O **índice de engajamento** é uma medida que combina a quantidade de curtidas 👍 e a natureza dos comentários 💬 (positivos 😊 ou negativos 😠) em uma postagem para determinar o quão bem ela foi recebida pelo público.
  
        - **Curtidas**: Adicionam positivamente ao índice. Mais curtidas indicam maior aprovação. 👍
        - **Comentários**: São avaliados por sentimento. Comentários positivos aumentam o índice 😊, enquanto negativos o diminuem 😠.
        - O índice é normalizado para estar entre -1 e 1, onde -1 indica engajamento extremamente negativo 😠, 0 indica neutralidade 😐, e 1 indica engajamento extremamente positivo 😊.
  
        Esse índice ajuda a entender se uma postagem foi bem-recebida (positiva) 😊, mal-recebida (negativa) 😠 ou teve uma recepção mista/neutra 😐 pelo público.
    """)

    # Suponha que df é o seu DataFrame
    hashtags, profiles = instafunctions.extract_hashtags_and_profiles(df)

    # Criar e exibir gráfico de barras para hashtags
    fig_hashtags = instafunctions.create_bar_chart(hashtags, 'Top Hashtags')
    st.plotly_chart(fig_hashtags)

    # Criar e exibir gráfico de barras para perfis
    fig_profiles = instafunctions.create_bar_chart(profiles,
                                                   'Top Perfis Mencionados')
    st.plotly_chart(fig_profiles)
  
  # Análise 
  elif opcao == "Análise do Conteúdo":
    # Calcular os valores máximos de sentimento e curtidas
    max_sentimento = max(
        abs(comment.get('Sentimento', 0)) for post in df['Comentarios']
        for comment in post if post)
    max_curtidas = df['Curtidas'].max()
    df_maximos = {'max_sentimento': max_sentimento, 'max_curtidas': max_curtidas}

    df['Data'] = pd.to_datetime(df['Data'])

    # Adicionar índice de engajamento e classificação ao DataFrame
    df['IndiceEngajamento'], df['ClassificacaoEngajamento'] = zip(*df.apply(
        lambda post: instafunctions.calcular_engajamento(post, df_maximos),
        axis=1))
    # Exibir o DataFrame no Streamlit
    st.dataframe(
        df[['Data', 'Texto', 'IndiceEngajamento', 'ClassificacaoEngajamento']])
    resultado = instafunctions.analisar_e_salvar_dataframe(df, label)
    st.write(resultado)
  
    # Exibindo o gráfico de palavras mais comuns
    st.header("Palavras Mais Comuns nos Comentários")
    # No seu script Streamlit
    fig = instafunctions.plot_most_common_words_in_comments_plotly(df)
    st.plotly_chart(fig)
  
    # Exibindo os top 5 posts com mais curtidas e comentários
    st.header("Top 5 Posts com Mais Curtidas")
    # Exemplo de uso para os posts com mais curtidas
    top_curtidas = instafunctions.top_posts(df, 'Curtidas', 6, False)
    instafunctions.exibir_posts_com_cards(top_curtidas)
          
          
  
    st.header("Top 5 Posts com Menos Curtidas")
    # Exemplo de uso para os posts com mais comentários
    bottom_curtidas = instafunctions.top_posts(df, 'Curtidas', 6, True)
    instafunctions.exibir_posts_com_cards(bottom_curtidas)
  
    st.header("Top 5 Posts com Mais Comentários")
    # Exemplo de uso para os posts com mais comentários
    top_comentarios = instafunctions.top_posts(df, 'Comentarios', 6, False)
    instafunctions.exibir_posts_com_cards(top_comentarios)
  
    st.header("Top 5 Posts com Menos Comentários")
    # Exemplo de uso para os posts com mais comentários
    bottom_comentarios = instafunctions.top_posts(df, 'Comentarios', 6, True)
    instafunctions.exibir_posts_com_cards(bottom_comentarios)
  
    
  # Exibição de autores como metric cards
  # Supondo que você tenha um DataFrame chamado df_autores
  elif opcao == "Comentários":
  # Exibir o DataFrame no Streamlit
    # Exibindo o gráfico de pizza de classificação dos comentários
    st.header("Classificação dos Comentários")
    # No seu script Streamlit
    fig = instafunctions.plot_comment_classification_plotly(df)
    if fig is not None:
      st.plotly_chart(fig)

    # Exibindo o gráfico de pizza de classificação dos comentários
    st.header("Principais autores de comentarios")
    # No seu script Streamlit
    fig = instafunctions.plot_top_comment_authors(df)
    st.plotly_chart(fig)

    # No seu script Streamlit
    top_positive = instafunctions.top_positive_comments(df)
    st.header("Top 10 Comentários Mais Positivos")
    st.dataframe(top_positive)

    top_negative = instafunctions.top_negative_comments(df)
    st.header("Top 10 Comentários Mais Negativos")
    st.dataframe(top_negative)

    # Suponha que df_json é o seu DataFrame
    author_counts = instafunctions.count_authors_by_sentiment(df)

    # Preparando os dados para o gráfico
    data = pd.DataFrame({
        'Classificação': list(author_counts.keys()),
        'Número de Autores': list(author_counts.values())
    })

    # Criar um gráfico de pizza com Plotly
    fig = instafunctions.px.pie(data,
                                values='Número de Autores',
                                names='Classificação',
                                title='Distribuição de Autores por Sentimento')

    # Exibir o gráfico no Streamlit
    st.plotly_chart(fig)

    # Aplicando a função para obter os top 10 autores
    top_positives, top_negatives = instafunctions.identify_top_authors_by_sentiment(
        df, top_n=9)

    # Exibindo os resultados no Streamlit
    st.header("Top 10 Autores com Comentários Mais Positivos")
    instafunctions.exibir_cards_autores_negativos(top_positives)

    st.header("Top 10 Autores com Comentários Mais Negativos")
    instafunctions.exibir_cards_autores_negativos(top_negatives)

# Exemplo de uso
# df = seu_dataframe_com_dados
# df_autores = seu_dataframe_com_autores
criar_pagina_com_streamlit_extras(df)
