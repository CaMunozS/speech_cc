# speech_cc

Prueba de concepto para clasificación y descubrimiento de tópicos en conversaciones.

## Requerimientos
- Instalar las dependencias con `pip install -r requirements.txt`.

## Pruebas
- Ejecutar `pytest` para validar el flujo supervisado y los modelos no supervisados.

## Contenido
- `topic_modeling.ipynb`: genera datos sintéticos con mensajes internos y externos, aplica clasificación supervisada y múltiples modelos no supervisados (LDA, NMF, K-Means, BERTopic) para encontrar nuevos tópicos.
