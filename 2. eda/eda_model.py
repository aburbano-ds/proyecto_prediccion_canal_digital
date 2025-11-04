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

# COMMAND ----------

# MAGIC %md
# MAGIC #### Librerias y Funciones

# COMMAND ----------

spark.conf.set("spark.databricks.io.cache.enabled", True)
spark.conf.set('spark.sql.shuffle.partitions', 'auto')


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
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
import plotly.express as px

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)

sns.set_theme(style="ticks")

from datetime import datetime
from dateutil.relativedelta import relativedelta

pd.set_option('display.float_format', lambda x: '%.5f' % x)

from pyspark.sql import SparkSession

spark = SparkSession \
    .builder \
    .appName("Python Spark") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA

from pyspark.sql.window import Window
from pyspark.sql.functions import row_number
import plotly.express as px
import plotly.graph_objects as go

import mlflow
mlflow.autolog(disable=True)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Carga de Fuente

# COMMAND ----------

int_pedidos_clientes = (spark.read.parquet("/Volumes/dbw_prod_aavanzada/db_tmp/files/pburbano/data/")
                                  .withColumn("fecha_pedido_dt", F.to_date(F.col("fecha_pedido_dt")))
                        )

# COMMAND ----------

from pyspark.sql import Window
import pyspark.sql.functions as F

# === 1. Definición de ventanas ===
w = Window.partitionBy("cliente_id").orderBy(F.asc("fecha_pedido_dt"))
w_prev_all = w.rowsBetween(Window.unboundedPreceding, -1)  # Hasta el anterior
w_recent = w.rowsBetween(-3, -1)  # Últimos 3 anteriores

# === 2. Base con canal_siguiente y target ===
df = int_pedidos_clientes.withColumn("canal_siguiente", F.lead("canal_pedido_cd").over(w))
df = df.filter(F.col("canal_siguiente").isNotNull())
df = df.withColumn("target", F.when(F.col("canal_siguiente") == "DIGITAL", 1).otherwise(0))

# === 3. Variables de canal binarias (DIGITAL vs NO DIGITAL) ===
df = (
    df.withColumn("canal_previo", F.lag("canal_pedido_cd").over(w))
      .withColumn("canal_actual", F.col("canal_pedido_cd"))
      .withColumn("canal_actual_digital", F.when(F.col("canal_actual") == "DIGITAL", 1).otherwise(0))
      .withColumn("canal_previo_digital", F.when(F.col("canal_previo") == "DIGITAL", 1).otherwise(0))
)

# === 4. Variables históricas ===
df = (
    df.withColumn("dias_desde_pedido_anterior", F.datediff("fecha_pedido_dt", F.lag("fecha_pedido_dt").over(w)))
      .withColumn("n_pedidos_previos", F.row_number().over(w) - 1)
      # Facturación
      .withColumn("facturacion_prom_anterior", F.avg("facturacion_usd_val").over(w_prev_all))
      .withColumn("facturacion_total_prev", F.sum("facturacion_usd_val").over(w_prev_all))
      .withColumn("desviacion_facturacion", F.stddev("facturacion_usd_val").over(w_prev_all))
      # Canal digital
      .withColumn("uso_digital_prev", F.sum(F.when(F.col("canal_pedido_cd") == "DIGITAL", 1).otherwise(0)).over(w_prev_all))
      .withColumn("uso_no_digital_prev", F.col("n_pedidos_previos") - F.col("uso_digital_prev"))
      .withColumn("prop_digital_prev", F.col("uso_digital_prev") / F.when(F.col("n_pedidos_previos") > 0, F.col("n_pedidos_previos")).otherwise(1))
      .withColumn("prop_no_digital_prev", F.col("uso_no_digital_prev") / F.when(F.col("n_pedidos_previos") > 0, F.col("n_pedidos_previos")).otherwise(1))
      # Frecuencia
      .withColumn("dias_media_prev", F.avg(F.datediff("fecha_pedido_dt", F.lag("fecha_pedido_dt").over(w))).over(w_prev_all))
      .withColumn("dias_media_std", F.stddev(F.datediff("fecha_pedido_dt", F.lag("fecha_pedido_dt").over(w))).over(w_prev_all))
)

# === 5. Variables recientes (últimos 3 pedidos) ===
df = (
    df.withColumn("facturacion_prom_reciente", F.avg("facturacion_usd_val").over(w_recent))
      .withColumn("uso_digital_reciente", F.avg(F.when(F.col("canal_pedido_cd") == "DIGITAL", 1).otherwise(0)).over(w_recent))
      .withColumn("uso_no_digital_reciente", F.avg(F.when(F.col("canal_pedido_cd") != "DIGITAL", 1).otherwise(0)).over(w_recent))
)

# === 6. Variables de materiales y cajas ===
df = (
    df
      .withColumn("materiales_prom_prev", F.avg("materiales_distintos_val").over(w_prev_all))
      .withColumn("materiales_total_prev", F.sum("materiales_distintos_val").over(w_prev_all))
      .withColumn("cajas_fisicas_prom_prev", F.avg("cajas_fisicas").over(w_prev_all))
      .withColumn("cajas_fisicas_total_prev", F.sum("cajas_fisicas").over(w_prev_all))
      .withColumn("materiales_reciente", F.avg("materiales_distintos_val").over(w_recent))
      .withColumn("cajas_fisicas_reciente", F.avg("cajas_fisicas").over(w_recent))
      .withColumn("cajas_por_material", 
                  F.when(F.col("materiales_distintos_val") > 0, 
                         F.col("cajas_fisicas") / F.col("materiales_distintos_val"))
                   .otherwise(0))
      .withColumn("cajas_por_material_prev", 
                  F.avg(F.when(F.col("materiales_distintos_val") > 0, 
                               F.col("cajas_fisicas") / F.col("materiales_distintos_val"))
                        .otherwise(0)).over(w_prev_all))
)

# === 7. Variables temporales ===
df = (
    df.withColumn("mes", F.month("fecha_pedido_dt"))
      .withColumn("dia_semana", F.dayofweek("fecha_pedido_dt"))
      .withColumn("es_fin_de_semana", F.when(F.col("dia_semana").isin(1, 7), 1).otherwise(0))
      .withColumn("trimestre", F.quarter("fecha_pedido_dt"))
)

# === 8. Antigüedad ===
df = df.withColumn(
    "antiguedad_dias",
    F.datediff("fecha_pedido_dt", F.min("fecha_pedido_dt").over(Window.partitionBy("cliente_id")))
)

# === 9. Selección final de variables ===
mdt = (
    df.filter(F.col("n_pedidos_previos") > 0)
      .filter(F.col("target").isNotNull())
      .select(
        # Identificadores
        "cliente_id", "pais_cd", "region_comercial_txt", "agencia_id", "ruta_id",
        "tipo_cliente_cd", "madurez_digital_cd", "estrellas_txt", "frecuencia_visitas_cd",
        # Target
        "target",
        # Canal binario
        "canal_actual_digital", "canal_previo_digital",
        # Comportamiento
        "facturacion_usd_val", "dias_desde_pedido_anterior", "n_pedidos_previos",
        "facturacion_prom_anterior", "facturacion_total_prev", "desviacion_facturacion",
        "uso_digital_prev", "uso_no_digital_prev", "prop_digital_prev", "prop_no_digital_prev",
        "facturacion_prom_reciente", "uso_digital_reciente", "uso_no_digital_reciente",
        "dias_media_prev", "dias_media_std",
        # Materiales y cajas
        "materiales_distintos_val", "materiales_prom_prev", "materiales_total_prev", "materiales_reciente",
        "cajas_fisicas", "cajas_fisicas_prom_prev", "cajas_fisicas_total_prev", "cajas_fisicas_reciente",
        "cajas_por_material", "cajas_por_material_prev",
        # Temporalidad
        "mes", "dia_semana", "es_fin_de_semana", "trimestre",
        "antiguedad_dias", "fecha_pedido_dt"
      )
)

# === 10. Etiquetar periodo y limpiar ===
fecha_corte = "2024-03-01"
mdt = mdt.withColumn("periodo", F.when(F.col("fecha_pedido_dt") < fecha_corte, "TRAIN").otherwise("TEST"))
mdt = mdt.fillna(0)

# === 11. Exportar a pandas (opcional) ===
mdt_pd = mdt.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Analisis de Nulos

# COMMAND ----------

df_null = null_function(mdt_pd)
df_null.sort_values("per_null", ascending=False).head(10)

# COMMAND ----------

df_binaria_pd = mdt_pd.copy()
#df_binaria_pd = mdt_pd.drop(columns=["madurez_digital_cd"]).copy()

top_20 = df_binaria_pd['ruta_id'].value_counts().nlargest(20).index
df_binaria_pd['ruta_id'] = df_binaria_pd['ruta_id'].where(df_binaria_pd['ruta_id'].isin(top_20), 'otra_ruta')

# Dividir en Train/Test según columna 'periodo'
data_train = df_binaria_pd[df_binaria_pd["periodo"] == "TRAIN"].copy()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Análisis Numericas

# COMMAND ----------

data_train.select_dtypes("number").columns.to_list()

# COMMAND ----------

num_list = ['canal_actual_digital',
 'canal_previo_digital',
 'facturacion_usd_val',
 'dias_desde_pedido_anterior',
 'n_pedidos_previos',
 'facturacion_prom_anterior',
 'facturacion_total_prev',
 'desviacion_facturacion',
 'uso_digital_prev',
 'uso_no_digital_prev',
 'prop_digital_prev',
 'prop_no_digital_prev',
 'facturacion_prom_reciente',
 'uso_digital_reciente',
 'uso_no_digital_reciente',
 'dias_media_prev',
 'dias_media_std',
 'materiales_distintos_val',
 'materiales_prom_prev',
 'materiales_total_prev',
 'materiales_reciente',
 'cajas_fisicas',
 'cajas_fisicas_prom_prev',
 'cajas_fisicas_total_prev',
 'cajas_fisicas_reciente',
 'cajas_por_material',
 'cajas_por_material_prev',
 'mes',
 'dia_semana',
 'es_fin_de_semana',
 'trimestre',
 'antiguedad_dias']

# COMMAND ----------

df_iv_num = iv_df_opt(data_train, [], num_list, "target")

# COMMAND ----------

df_iv_num.head(20)

# COMMAND ----------

seleccion_numeric_iv(data_train, df_iv_num, corr_umbral=0.5, iv_umbral=0.01)

# COMMAND ----------

num_list_final = seleccion_numeric_iv(data_train, df_iv_num, corr_umbral=0.4, iv_umbral=0.01)

# COMMAND ----------

pearson_matrix(data_train, num_list_final).round(3)

# COMMAND ----------

numeric_eda(data_train, num_list_final).round(3)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Categoricas

# COMMAND ----------

data_train.select_dtypes("object").columns.to_list()

# COMMAND ----------

cat_list = ['pais_cd',
 'region_comercial_txt',
 'agencia_id',
 'ruta_id',
 'tipo_cliente_cd',
 'madurez_digital_cd',
 'estrellas_txt',
 'frecuencia_visitas_cd',
 ]

# COMMAND ----------

df_iv_cat = iv_df_opt(data_train, cat_list, [], "target")

# COMMAND ----------

df_iv_cat

# COMMAND ----------

seleccion_categoric_iv(data_train, df_iv_cat, corr_umbral=0.5, iv_umbral=0.01)

# COMMAND ----------

cat_list_final = seleccion_categoric_iv(data_train, df_iv_cat, corr_umbral=0.5, iv_umbral=0.02)

# COMMAND ----------

cramers_v_matrix(data_train, cat_list_final)

# COMMAND ----------

category_eda(data_train, cat_list_final)

# COMMAND ----------

var_final = num_list_final + cat_list_final
var_final

# COMMAND ----------

# MAGIC %md
# MAGIC #### VIF

# COMMAND ----------

var_vif = ['prop_digital_prev',
'canal_actual_digital']

# COMMAND ----------

calc_vif(data_train[var_vif].dropna()).sort_values("VIF", ascending = False)