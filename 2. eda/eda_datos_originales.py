# Databricks notebook source
# MAGIC %md
# MAGIC ### Librerias y Funciones

# COMMAND ----------

spark.conf.set("spark.databricks.io.cache.enabled", True)
spark.conf.set('spark.sql.shuffle.partitions', 'auto')

# COMMAND ----------

# MAGIC %run ../../../../../04_utils/commons_functions_de

# COMMAND ----------

# MAGIC %run ../../../../../04_utils/commons_functions_ds

# COMMAND ----------

# MAGIC %run ../../../../../spigot/initial/global_parameter_py

# COMMAND ----------

# MAGIC %md
# MAGIC #### Librerias y Funciones

# COMMAND ----------

# DBTITLE 1,limpiar
from IPython.display import display

# mute warnings
import warnings
warnings.filterwarnings('ignore')

from pyspark.sql.types import IntegerType
import pyspark.sql.functions as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from pyspark.sql import Window
from optbinning import BinningProcess
from optbinning import Scorecard
from optbinning.scorecard import plot_auc_roc, plot_cap, plot_ks
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
import mlflow
import plotly.express as px
pd.set_option('display.max_rows', 520)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
sns.set_theme(style="ticks")
from datetime import datetime
from dateutil.relativedelta import relativedelta
from sklearn import metrics


pd.set_option('display.float_format', lambda x: '%.5f' % x)

from IPython.display import display
import pyspark.sql.functions as F
import pandas as pd
from pyspark.sql import Window
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number
import seaborn as sns

# COMMAND ----------

# MAGIC %md
# MAGIC #### Carga de Fuentes

# COMMAND ----------

tb_pedidos_clientes = spark.read.parquet("/Volumes/dbw_prod_aavanzada/db_tmp/files/pburbano/data/")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Analisis Exploratorio Inicial

# COMMAND ----------

tb_pedidos_clientes.limit(5).display()

# COMMAND ----------

tb_pedidos_clientes.printSchema()

# COMMAND ----------

tb_pedidos_clientes_pd = tb_pedidos_clientes.toPandas()

# COMMAND ----------

df_null = null_function(tb_pedidos_clientes_pd)
df_null.sort_values("per_null", ascending=False).head(10)

# COMMAND ----------

# DBTITLE 1,distribucion numericas
num_cols = df_pedidos_clientes.select_dtypes("number").columns.tolist()

numeric_eda(df_pedidos_clientes, num_cols).round(3)

# COMMAND ----------

# DBTITLE 1,distribucion categoricas
cat_cols = df_pedidos_clientes.select_dtypes("object").columns.tolist()

category_eda(df_pedidos_clientes, cat_cols).round(3)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Analisis de Canales

# COMMAND ----------

df_pedidos_clientes = tb_pedidos_clientes.toPandas()

# COMMAND ----------

# DBTITLE 1,pedidos por canal
sns.countplot(x="canal_pedido_cd", data=df_pedidos_clientes)
plt.title("Distribución de pedidos por canal")

# COMMAND ----------

frecuencia = df.groupby(['cliente_id', 'canal_pedido_cd']).size().reset_index(name='n_pedidos')
sns.boxplot(data=frecuencia, x='canal_pedido_cd', y='n_pedidos')
plt.title("Distribucion de pedidos por canal")


# COMMAND ----------

# DBTITLE 1,pedidos por canal y pais
sns.countplot(data=df_pedidos_clientes, x="pais_cd", hue="canal_pedido_cd")
plt.title("Distribución de pedidos por canal y pais")

# COMMAND ----------

# Crear figura con un subplot por país
paises = df_pedidos_clientes['pais_cd'].unique()
n_paises = len(paises)

fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharey=True); axes = axes.flatten()

# Iterar por país y graficar
for ax, pais in zip(axes, paises):
    subset = df[df['pais_cd'] == pais]
    sns.countplot(data=subset, x='region_comercial_txt', hue="canal_pedido_cd", ax=ax, order=subset['region_comercial_txt'].value_counts().index)
    ax.set_title(f'Pedidos por región - {pais}')
    ax.set_xlabel('Región')
    ax.set_ylabel('Cantidad de pedidos')
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# COMMAND ----------

df_pedidos_clientes.columns

# COMMAND ----------

# DBTITLE 1,pedidos por tipo de cliente y por canal *
sns.countplot(data=df_pedidos_clientes, x="tipo_cliente_cd", hue="canal_pedido_cd")
plt.title("pedidos por tipo de cliente y por canal")

# COMMAND ----------

# DBTITLE 1,pedidos por madurez y canal *
sns.countplot(data=df_pedidos_clientes, x="madurez_digital_cd", hue="canal_pedido_cd")
plt.title("pedidos por madurez_digital_cd y por canal")

# COMMAND ----------

# DBTITLE 1,facturacion por canal
sns.boxplot(data=df_pedidos_clientes, x="canal_pedido_cd", y="facturacion_usd_val")
plt.title("facturacion_usd_val y por canal")

# COMMAND ----------

# DBTITLE 1,materiales por canal
sns.boxplot(data=df_pedidos_clientes, x="canal_pedido_cd", y="materiales_distintos_val")
plt.title("materiales_distintos por canal")

# COMMAND ----------

# DBTITLE 1,cajas fisicas por canal
sns.boxplot(data=df_pedidos_clientes, x="canal_pedido_cd", y="cajas_fisicas")
plt.title("cajas_fisicas por canal")

# COMMAND ----------

# DBTITLE 1,numero de pedidos por canal
# Preparar datos
df = df_pedidos_clientes.copy()
df['mes_anio'] = df['fecha_pedido_dt'].dt.to_period('M').astype(str)

# Agrupar por mes y canal
canales_mensuales = df.groupby(['mes_anio', 'canal_pedido_cd']).size().reset_index(name='pedidos')

# Gráfico con Seaborn
plt.figure(figsize=(10, 6))
sns.lineplot(data=canales_mensuales, x='mes_anio', y='pedidos', hue='canal_pedido_cd', marker='o')

plt.title('Evolución mensual del número de pedidos por canal')
plt.ylabel('Número de pedidos')
plt.xlabel('Mes')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# COMMAND ----------

frecuencia = df.groupby(['cliente_id', 'canal_pedido_cd']).size().reset_index(name='n_pedidos')
sns.boxplot(data=frecuencia, x='canal_pedido_cd', y='n_pedidos')
plt.title("Frecuencia de pedidos por canal (por cliente)")


# COMMAND ----------

# MAGIC %md
# MAGIC #### Analisis por Cliente

# COMMAND ----------

# DBTITLE 1,numero de pedidos por cliente
int_pedidos_clientes.groupBy("cliente_id").agg(F.count("*").alias("pedidos por cliente")).toPandas().hist(bins=20)

# COMMAND ----------

int_pedidos_clientes.groupBy("cliente_id").agg(F.count("*")).toPandas().describe()

# COMMAND ----------

# DBTITLE 1,uso de canales distintos
int_pedidos_clientes.groupBy("cliente_id").agg(F.countDistinct("canal_pedido_cd").alias("canales_distintos_utilizados")).toPandas().hist()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Analisis para Estructurar Target y MDT

# COMMAND ----------

# odena por cliente y fecha
w = Window.partitionBy("cliente_id").orderBy("fecha_pedido_dt")

# canal actual c canal siguiente
df_trans = tb_pedidos_clientes.withColumn("canal_actual", F.col("canal_pedido_cd"))
df_trans = df_trans.withColumn("canal_siguiente", F.lead("canal_pedido_cd").over(w))

# filtra donde hay un canal siguiente
df_trans = df_trans.filter(F.col("canal_siguiente").isNotNull())

# Tabla de transición
transitions = (df_trans.groupBy("canal_actual", "canal_siguiente")
                      .agg(F.count("*").alias("n"))
                      .toPandas()
                      .pivot(index="canal_actual", columns="canal_siguiente", values="n")
                      .fillna(0))

# proporcion por fila
transitions_prop = transitions.div(transitions.sum(axis=1), axis=0)


# COMMAND ----------

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 4))
sns.heatmap(transitions_prop, annot=True, fmt=".2%", cmap="Blues")
plt.title("Transición entre canales")
plt.ylabel("Canal actual")
plt.xlabel("Canal siguiente")
plt.tight_layout()
plt.show()

# COMMAND ----------

madurez_transition = (
    df_trans.groupBy("madurez_digital_cd", "canal_siguiente")
            .agg(F.count("*").alias("n"))
            .groupBy("madurez_digital_cd")
            .pivot("canal_siguiente")
            .sum("n")
            .fillna(0)
)

# Convertir a pandas y normalizar por fila
madurez_transition_pd = madurez_transition.toPandas().set_index("madurez_digital_cd")
madurez_transition_prop = madurez_transition_pd.div(madurez_transition_pd.sum(axis=1), axis=0)

def plot_transition_heatmap(df_prop, title):
    plt.figure(figsize=(6, 4))
    sns.heatmap(df_prop, annot=True, fmt=".2%", cmap="Blues")
    plt.title(title)
    plt.ylabel("Madurez")
    plt.xlabel("Canal siguiente")
    plt.tight_layout()
    plt.show()

# Ejemplo para madurez
plot_transition_heatmap(madurez_transition_prop, "Transición entre canales por Madurez")


# COMMAND ----------

tipo_transition = (
    df_trans.groupBy("tipo_cliente_cd", "canal_siguiente")
            .agg(F.count("*").alias("n"))
            .groupBy("tipo_cliente_cd")
            .pivot("canal_siguiente")
            .sum("n")
            .fillna(0)
)

tipo_transition_pd = tipo_transition.toPandas().set_index("tipo_cliente_cd")
tipo_transition_prop = tipo_transition_pd.div(tipo_transition_pd.sum(axis=1), axis=0)

def plot_transition_heatmap(df_prop, title):
    plt.figure(figsize=(6, 4))
    sns.heatmap(df_prop, annot=True, fmt=".2%", cmap="Blues")
    plt.title(title)
    plt.ylabel("Tipo de Cliente")
    plt.xlabel("Canal siguiente")
    plt.tight_layout()
    plt.show()

plot_transition_heatmap(tipo_transition_prop, "Transición al canal siguiente por Tipo de Cliente")

# COMMAND ----------

pais_transition = (
    df_trans.groupBy("pais_cd", "canal_siguiente")
            .agg(F.count("*").alias("n"))
            .groupBy("pais_cd")
            .pivot("canal_siguiente")
            .sum("n")
            .fillna(0)
)

pais_transition_pd = pais_transition.toPandas().set_index("pais_cd")
pais_transition_prop = pais_transition_pd.div(pais_transition_pd.sum(axis=1), axis=0)

def plot_transition_heatmap(df_prop, title):
    plt.figure(figsize=(6, 4))
    sns.heatmap(df_prop, annot=True, fmt=".2%", cmap="Blues")
    plt.title(title)
    plt.ylabel("Pais")
    plt.xlabel("Canal siguiente")
    plt.tight_layout()
    plt.show()

plot_transition_heatmap(pais_transition_prop, "Transición al canal siguiente por Pais")

# COMMAND ----------

df.groupby("estrellas_txt")["canal_pedido_cd"].value_counts(normalize=True).unstack().plot(kind='bar', stacked=True)

# COMMAND ----------

sns.countplot(data=df, x="frecuencia_visitas_cd", hue="canal_pedido_cd")

# COMMAND ----------

df_target = df_trans.withColumn("target", F.when(F.col("canal_siguiente") == "DIGITAL", 1).otherwise(0))
df_target.groupBy("target").count().toPandas().plot(kind="bar", x="target", y="count", title="Distribución de target potencial")