# speech_cc

Prueba de concepto para clasificación y descubrimiento de tópicos en conversaciones.

## Requerimientos
- Instalar las dependencias con `pip install -r requirements.txt`.
- Descargar el modelo `paraphrase-multilingual-MiniLM-L12-v2` y colocarlo en la carpeta `model/` del proyecto. Ejemplo:

  ```bash
  git lfs install
  git clone https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 model/paraphrase-multilingual-MiniLM-L12-v2
  ```

  Para trabajar sin conexión, establecer `TRANSFORMERS_OFFLINE=1` antes de ejecutar los scripts.

## Pruebas
- Ejecutar `pytest` para validar el flujo supervisado y los modelos no supervisados.

## Contenido
- `topic_modeling.py`: genera datos sintéticos con mensajes internos y externos, aplica clasificación supervisada y múltiples modelos no supervisados (LDA, NMF, K-Means, BERTopic) para encontrar nuevos tópicos. Usa spaCy y NLTK con stopwords en español.
- `embedding_model.py`: carga `paraphrase-multilingual-MiniLM-L12-v2` desde `model/` usando `sentence-transformers` para obtener embeddings multilingües.
- `nltk_data/`: stopwords de NLTK en inglés y español.
- `requirements.txt`: dependencias del proyecto.
