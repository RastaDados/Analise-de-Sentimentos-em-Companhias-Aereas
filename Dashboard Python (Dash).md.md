## Dashboard em Python Utilizando o Dash

```python
import pandas as pd
from dash import Dash, dcc, html, Input, Output
import plotly.express as px
from wordcloud import WordCloud
import io
import base64
from PIL import Image
import numpy as np


#df = pd.read_csv('Tweets.csv')

#Criando o app
app = Dash(__name__)

#Layout do dashboard
app.layout = html.Div([
    html.H1("Análise de Sentimentos de Companhias Aéreas"),
    
    html.Div([
        dcc.Graph(id='sentiment-pie'),
        dcc.Graph(id='reason-bar')
    ], style={'display': 'flex', 'width': '100%'}),
    
    html.Div([
        dcc.Graph(id='time-series'),
        html.Div(
            html.Img(id='word-cloud'),
            style={'width': '50%', 'display': 'flex', 'justify-content': 'center'}
        )
    ], style={'display': 'flex', 'width': '100%'}),
    
    dcc.Dropdown(
        id='airline-selector',
        options=[{'label': airline, 'value': airline} for airline in df['airline'].unique()],
        value=['Virgin America'],  # Valor inicial como lista
        multi=True,
        style={'width': '50%', 'margin': '20px auto'}
    )
], style={'font-family': 'Arial', 'max-width': '1200px', 'margin': '0 auto'})

#Função para gerar nuvem de palavras como imagem
def generate_wordcloud(text):
    if not text.strip():  #Se não houver texto
        #Retorna uma imagem branca vazia
        img = Image.new('RGB', (800, 400), color='white')
    else:
        wc = WordCloud(width=800, height=400, background_color='white')
        wc.generate(text)
        img = wc.to_image()
    
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    img_str = 'data:image/png;base64,' + base64.b64encode(buf.read()).decode()
    return img_str

#Callback p
@app.callback(
    [Output('sentiment-pie', 'figure'),
     Output('reason-bar', 'figure'),
     Output('time-series', 'figure'),
     Output('word-cloud', 'src')],
    [Input('airline-selector', 'value')]
)
def update_dashboard(selected_airlines):
    #Se nenhuma companhia aérea estiver selecionada, usar todas
    if not selected_airlines:
        filtered_df = df.copy()
    else:
        filtered_df = df[df['airline'].isin(selected_airlines)]
    
    #Gráfico de pizza de sentimentos
    sentiment_counts = filtered_df['airline_sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['sentiment', 'count']
    
    sentiment_pie = px.pie(
        sentiment_counts,
        names='sentiment',
        values='count',
        title='Distribuição de Sentimentos',
        color='sentiment',
        color_discrete_map={
            'positive': '#2ca02c',
            'neutral': '#ff7f0e',
            'negative': '#d62728'
        }
    )
    sentiment_pie.update_traces(textposition='inside', textinfo='percent+label')
    
    #Gráfico de barras de razões
    reason_df = filtered_df[filtered_df['airline_sentiment'] == 'negative']
    if not reason_df.empty:
        reason_counts = reason_df['negativereason'].value_counts().reset_index()
        reason_counts.columns = ['reason', 'count']
        reason_bar = px.bar(
            reason_counts,
            x='reason',
            y='count',
            title='Principais Razões para Reclamações',
            labels={'reason': 'Razão', 'count': 'Número de Reclamações'}
        )
        reason_bar.update_layout(xaxis_tickangle=-45)
    else:
        #DataFrame vazio se não houver reclamações
        reason_bar = px.bar(title='Nenhuma reclamação encontrada')
    
    #Série temporal
    time_series_df = filtered_df.groupby('date_dt')['sentiment_compound'].mean().reset_index()
    time_series = px.line(
        time_series_df,
        x='date_dt',
        y='sentiment_compound',
        title='Sentimento Médio ao Longo do Tempo',
        labels={'date_dt': 'Data', 'sentiment_compound': 'Sentimento Médio'}
    )
    time_series.update_layout(
        yaxis_range=[-1, 1],
        shapes=[{
            'type': 'line',
            'yref': 'paper', 'y0': 0, 'y1': 1,
            'xref': 'paper', 'x0': 0, 'x1': 1,
            'line': {'color': 'gray', 'dash': 'dash'}
        }]
    )
    
    #Nuvem de palavras
    text = ' '.join(filtered_df['processed_text'].dropna().astype(str))
    wordcloud_src = generate_wordcloud(text)
    
    return sentiment_pie, reason_bar, time_series, wordcloud_src

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
```


