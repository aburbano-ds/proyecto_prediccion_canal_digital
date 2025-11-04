# Proyecto de Predicción de Canal Digital

Este proyecto implementa un pipeline completo para predecir si el próximo pedido de un cliente será realizado por canal digital, utilizando modelos de clasificación entrenados con el historial de comportamiento de cada cliente.

## Objetivo

Detectar anticipadamente qué clientes utilizarán el canal DIGITAL en su siguiente pedido, con el fin de:

- Activar campañas de adopción digital.
- Fomentar la retención en el canal digital.
- Optimizar esfuerzos comerciales.

Se modelan las **transiciones entre canales** usando información histórica a nivel de pedido, sin fuga de información.

## Estructura del Proyecto

- Project/
- data/              # Datos crudos y procesados
- data_processing/   # Generación de la MDT (Matriz por Pedido)
- eda/               # Análisis exploratorio (EDA)
- experiments/       # Experimentos y pruebas con modelos
- README.md          # Este archivo