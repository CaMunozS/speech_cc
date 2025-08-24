from pathlib import Path

import pandas as pd
import spacy
import nltk
from nltk.corpus import stopwords

# Ensure NLTK uses the local nltk_data directory
nltk.data.path.append(str(Path(__file__).resolve().parent / 'nltk_data'))

# Load Spanish stopwords and spaCy model
nlp = spacy.blank('es')
STOP_WORDS = set(stopwords.words('spanish'))


def tokenize(text: str):
    """Tokenize Spanish text removing stopwords using spaCy and NLTK."""
    doc = nlp(text.lower())
    return [t.text for t in doc if t.is_alpha and t.text not in STOP_WORDS]


def main():
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
        {"speaker": "externo", "text": "Sí, pero sigue igual", "topics": None},
        {"speaker": "externo", "text": "¿Me pueden ayudar?", "topics": None},
    ]

    df = pd.DataFrame(data)

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import MultiLabelBinarizer
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import classification_report
    from sklearn.model_selection import cross_val_score

    client_msgs = df[df['speaker'] == 'externo'].copy()
    labeled = client_msgs[client_msgs['topics'].notnull()]
    unlabeled = client_msgs[client_msgs['topics'].isnull()]

    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(labeled['topics'])

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(tokenizer=tokenize)),
        ('clf', OneVsRestClassifier(LogisticRegression(solver='liblinear'))),
    ])

    scores = cross_val_score(pipeline, labeled['text'], y, cv=3, scoring='f1_macro')
    print('F1 promedio CV:', scores.mean())

    pipeline.fit(labeled['text'], y)
    y_pred = pipeline.predict(labeled['text'])
    print(classification_report(y, y_pred, target_names=mlb.classes_))

    pred = pipeline.predict(unlabeled['text'])
    unlabeled['topics'] = mlb.inverse_transform(pred)

    client_msgs = pd.concat([labeled, unlabeled])
    print(client_msgs)

    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.decomposition import LatentDirichletAllocation

    cv = CountVectorizer(tokenizer=tokenize)
    X_cv = cv.fit_transform(client_msgs['text'])
    lda = LatentDirichletAllocation(n_components=5, random_state=0)
    lda.fit(X_cv)

    def display_topics(model, feature_names, n_top_words=5):
        for idx, topic in enumerate(model.components_):
            terms = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
            print(f'Tópico {idx}: {" ".join(terms)}')

    feature_names = cv.get_feature_names_out()
    display_topics(lda, feature_names)

    from sklearn.decomposition import NMF
    from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF

    tfidf = TFIDF(tokenizer=tokenize)
    X_tfidf = tfidf.fit_transform(client_msgs['text'])
    nmf = NMF(n_components=5, random_state=0, init='nndsvda', max_iter=200)
    nmf.fit(X_tfidf)
    feature_names = tfidf.get_feature_names_out()
    display_topics(nmf, feature_names)

    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=5, random_state=0)
    km.fit(X_tfidf)
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]
    terms = tfidf.get_feature_names_out()
    for i in range(5):
        top_terms = [terms[ind] for ind in order_centroids[i, :5]]
        print(f'Cluster {i}: {" ".join(top_terms)}')

    from bertopic import BERTopic

    bertopic_model = BERTopic(verbose=False)
    ber_topics, _ = bertopic_model.fit_transform(client_msgs['text'].tolist())
    print(bertopic_model.get_topic_info().head())

    all_topics = client_msgs['topics'].explode().dropna()
    print(all_topics.value_counts())


if __name__ == '__main__':
    main()
