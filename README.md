# speech_cc

Prueba de concepto para clasificación y descubrimiento de tópicos en conversaciones.

## Requerimientos
- Instalar las dependencias con `pip install -r requirements.txt`.

## Pruebas
- Ejecutar `pytest` para validar el flujo supervisado y los modelos no supervisados.

## Contenido
- `topic_modeling.py`: genera datos sintéticos con mensajes internos y externos, aplica clasificación supervisada y múltiples modelos no supervisados (LDA, NMF, K-Means, BERTopic) para encontrar nuevos tópicos. Usa spaCy y NLTK con stopwords en español.
- `embedding_model.py`: carga `paraphrase-multilingual-MiniLM-L12-v2` de `sentence-transformers` para obtener embeddings multilingües.
- `nltk_data/`: stopwords de NLTK en inglés y español.
- `requirements.txt`: dependencias del proyecto.
