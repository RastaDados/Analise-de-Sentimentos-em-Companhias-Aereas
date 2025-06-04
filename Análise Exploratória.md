## Carregamento e Pré-Processamento dos Dados

```python
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

#Verificando e se você não tiver, vai baixar todos os recursos necessários
def download_nltk_resources():
    resources = {
        'punkt': 'tokenizers/punkt',
        'stopwords': 'corpora/stopwords',
        'wordnet': 'corpora/wordnet',
        'omw-1.4': 'corpora/omw-1.4',
        'punkt_tab': 'tokenizers/punkt_tab'
    }
    
    for resource, path in resources.items():
        try:
            nltk.data.find(path)
            print(f"Recurso {resource} já está instalado")
        except LookupError:
            print(f"Baixando recurso {resource}...")
            nltk.download(resource)

#Executando a função de download
download_nltk_resources()

#Carregando os dados
df = pd.read_csv('dados/Tweets.csv')

#Função de limpeza de textoa
def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    #Remoção de URLs, menções e caracteres especiais
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r"[^a-zA-ZÀ-ÿ0-9\s]", '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

#Aplicando limpeza acima
df['clean_text'] = df['text'].apply(clean_text)

#Removendo as duplicatas e textos vazios
df = df[df['clean_text'].str.len() > 0].drop_duplicates(subset=['clean_text'])

#Configurando para a lematização
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

#Função de pré-processamento 
def preprocess_text(text):
    try:
        tokens = word_tokenize(text)
        return ' '.join(
            lemmatizer.lemmatize(
                lemmatizer.lemmatize(word, pos='v'), pos='n'
            ) for word in tokens 
            if word not in stop_words and len(word) > 2
        )
    except Exception as e:
        print(f"Erro no texto: '{text[:50]}...' - {str(e)}")
        return ""

#Pré-processamento 
tqdm.pandas()
print("\nIniciando pré-processamento...")
df['processed_text'] = df['clean_text'].progress_apply(preprocess_text)

#Removendo as linhas com texto vazio
df = df[df['processed_text'].str.len() > 0]

#Colunas auxiliares
df['text_length'] = df['clean_text'].apply(len)
df['word_count'] = df['processed_text'].apply(lambda x: len(x.split()))

#Verificando os dados
print("\nPré-processamento concluído com sucesso!")
print(f"Tamanho final do dataset: {len(df)}")
print("\nAmostra dos resultados:")
print(df[['text', 'processed_text']].head(3).to_markdown())
```

![image](https://github.com/user-attachments/assets/22c2c0ca-4213-4c8c-bc07-05c8f965bdc0)

<hr>

### Análise Inicial

```python
print(f"Total de tweets após limpeza: {len(df)}")
print(f"Distribuição por companhia aérea:\n{df['airline'].value_counts()}")
print(f"Distribuição de sentimentos:\n{df['airline_sentiment'].value_counts()}")
```

![image](https://github.com/user-attachments/assets/f1390b24-1b9a-4670-a862-37bd38df6ec1)

<hr>

### Análise Exploratória

#### Volume de Tweets por Dia

```python
import matplotlib.pyplot as plt
import seaborn as sns

#Convertendo  para datetime
df['tweet_created'] = pd.to_datetime(df['tweet_created'])

#Extraindo o dia e a hora
df['date'] = df['tweet_created'].dt.date
df['hour'] = df['tweet_created'].dt.hour

#Gráfico de Volume de Tweets por Dia
plt.figure(figsize=(12, 6))
df.groupby('date').size().plot(kind='line', title='Volume de Tweets por Dia')
plt.ylabel('Número de Tweets')
plt.show()
```

![image](https://github.com/user-attachments/assets/2c7537d6-1ebb-49a1-a0a3-2283c7888390)

#### Distribuição de Sentimentos

```python
#Distribuição geral
plt.figure(figsize=(8, 6))
sns.countplot(x='airline_sentiment', data=df, order=['negative', 'neutral', 'positive'])
plt.title('Distribuição de Sentimentos')
plt.show()

#Gráfico Por companhia aérea
plt.figure(figsize=(10, 6))
sns.countplot(x='airline', hue='airline_sentiment', data=df, 
              hue_order=['negative', 'neutral', 'positive'])
plt.title('Sentimentos por Companhia Aérea')
plt.xticks(rotation=45)
plt.show()
```

![image](https://github.com/user-attachments/assets/09235a20-b67b-4b2d-b6c5-05f4ad2bb76e)

![image](https://github.com/user-attachments/assets/6966884d-4057-440b-a96d-21cd780114df)

<hr>

#### Nuvem de Palavras mais Usadas

```python
from wordcloud import WordCloud

#Concatenando todos os textos
all_text = ' '.join(df['processed_text'])

#Gerando a nuvem de palavras
wordcloud = WordCloud(width=800, height=500, background_color='white').generate(all_text)

#Criando o gráfico
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Nuvem de Palavras dos Tweets')
plt.show()
```

![image](https://github.com/user-attachments/assets/da29367c-deaa-47e8-9c00-37587d4896c4)

<hr>

#### Sentimentos Negativos

```python
#Filtrando apenas os tweets negativos
negative_tweets = df[df['airline_sentiment'] == 'negative']

#Criando o Gráfico
plt.figure(figsize=(12, 6))
negative_tweets['negativereason'].value_counts().plot(kind='bar')
plt.title('Principais Razões para Tweets Negativos')
plt.xticks(rotation=45)
plt.show()
```

![image](https://github.com/user-attachments/assets/0ab620ac-e9fa-4a3b-ac19-6a395af83f71)

<hr>

## Análise de Sentimentos

```python
from nltk.sentiment.vader import SentimentIntensityAnalyzer

#Iniciando o analisador
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

#Aplicando a análise de sentimentos
df['sentiment_scores'] = df['clean_text'].apply(lambda x: sia.polarity_scores(x))
df['sentiment_compound'] = df['sentiment_scores'].apply(lambda x: x['compound'])

#Classificando com base na coluna compound score
df['vader_sentiment'] = df['sentiment_compound'].apply(lambda x: 'positive' if x >= 0.05 else ('negative' if x <= -0.05 else 'neutral'))

#Comparando com as labels originais
from sklearn.metrics import classification_report
print(classification_report(df['airline_sentiment'], df['vader_sentiment']))
```

![image](https://github.com/user-attachments/assets/45650b41-3b93-4241-b4c8-0f5d50e98d83)

<hr>

### Visualização de Polaridade ao Longo do Tempo

```python
#Média móvel da polaridade
df['date_dt'] = pd.to_datetime(df['date'])
daily_sentiment = df.groupby('date_dt')['sentiment_compound'].mean().rolling(1).mean()

#Criando o Gráfico
plt.figure(figsize=(12, 6))
daily_sentiment.plot(title='Média Móvel de 7 Dias da Polaridade dos Tweets')
plt.ylabel('Polaridade (Compound Score)')
plt.axhline(0, color='gray', linestyle='--')
plt.show()
```

![image](https://github.com/user-attachments/assets/01ab909b-c8c3-4f63-a5f0-d48b8eb9b8a4)

<hr>

### Análise de Tópicos

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

#Vetorização
vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=1000)
X = vectorizer.fit_transform(df['processed_text'])

#Aplicando o LDA
lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(X)

#Mostrando os tópicos
def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = f"Tópico #{topic_idx + 1}: "
        message += ", ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)

print_top_words(lda, vectorizer.get_feature_names_out(), 10)
```

![image](https://github.com/user-attachments/assets/7b7e9729-75b5-4af3-acca-88a46250bac2)







