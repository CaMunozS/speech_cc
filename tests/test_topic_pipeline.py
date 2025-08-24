import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.cluster import KMeans


def build_dataset():
    data = [
        {"speaker": "interno", "text": "Hola, ¿en qué puedo ayudarte?", "topics": []},
        {"speaker": "externo", "text": "¿Cuál es el precio del plan básico?", "topics": ["precio"]},
        {"speaker": "interno", "text": "El plan básico cuesta 10 dólares.", "topics": []},
        {"speaker": "externo", "text": "Gracias, ¿y si quiero soporte premium?", "topics": ["soporte", "precio"]},
        {"speaker": "externo", "text": "El servicio fue muy malo", "topics": ["reclamo"]},
        {"speaker": "interno", "text": "Lamentamos escuchar eso, mejoraremos.", "topics": []},
        {"speaker": "externo", "text": "Quiero saber la factura pendiente", "topics": ["facturacion"]},
        {"speaker": "externo", "text": "Necesito hablar con ventas", "topics": ["ventas"]},
        {"speaker": "externo", "text": "¿Cuál es el más económico?", "topics": ["ventas"]},
    ]
    return pd.DataFrame(data)


def test_supervised_pipeline_runs():
    df = build_dataset()
    client_msgs = df[df['speaker'] == 'externo']
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(client_msgs['topics'])
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', OneVsRestClassifier(LogisticRegression(solver='liblinear')))
    ])
    pipeline.fit(client_msgs['text'], y)
    preds = pipeline.predict(client_msgs['text'])
    assert preds.shape == y.shape


def test_unsupervised_models_run():
    df = build_dataset()
    texts = df[df['speaker'] == 'externo']['text']
    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(texts)
    lda = LatentDirichletAllocation(n_components=2, random_state=0)
    nmf = NMF(n_components=2, random_state=0)
    km = KMeans(n_clusters=2, random_state=0)
    lda.fit(X)
    nmf.fit(X)
    km.fit(X)
    assert km.cluster_centers_.shape[0] == 2
