# speech_cc

Prueba de concepto para clasificación y descubrimiento de tópicos en conversaciones.

## Contenido
- `topic_modeling.py`: genera datos sintéticos con mensajes internos y externos, aplica clasificación supervisada y múltiples modelos no supervisados (LDA, NMF, K-Means, BERTopic) para encontrar nuevos tópicos. Usa spaCy y NLTK con stopwords en español.
- `nltk_data/`: stopwords de NLTK en inglés y español.
- `requirements.txt`: dependencias del proyecto.
