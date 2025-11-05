# Proyecto de Predicción de Canal Digital

Este proyecto implementa un pipeline completo para predecir si el próximo pedido de un cliente será realizado por canal digital, utilizando modelos de clasificación entrenados con el historial de comportamiento de cada cliente.

## Objetivo

Detectar anticipadamente qué clientes utilizarán el canal DIGITAL en su siguiente pedido, con el fin de:

- Activar campañas de adopción digital.
- Fomentar la retención en el canal digital.
- Optimizar esfuerzos comerciales.

## Enfoque Utilizado

Se exploraron diferentes modelos de clasificacion binaria y multiclase para modelar el problema.

- Enfoque 1: Target binaria con el ultimo canal del cliente (digital o no digital )como objetivo de prediccion
- Enfoque 2: Target multicalse con el cambio de estado del ultimo canal del cliente como objetivo de prediccion
- Enfoque 3: Targte multiclase con el estado de transicion de cada pedido del cliente como objetivo de prediccion

## Principales Hallazgos
- Enfoque 3  obtienne los mejores resultados, aprovechando toda la información y separando adecuadamente las clases
- Segmentacion de madurez tiene alto impacto en los modelos analizados

## Limitaciones y Posibles Mejoras
- Métricas del modelo son adecuadas pero pueden ser optimizadas con mejores features
- Porbar otros enfoques de modelamiento como LSTM
- 
## Estructura del Proyecto

- Project/
- data/              # Datos crudos y procesados
- data_processing/   # Generación de la MDT (Matriz por Pedido)
- eda/               # Análisis exploratorio (EDA)
- experiments/       # Experimentos y pruebas con modelos
- README.md          # Este archivo