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
from pyspark.sql.functions import regexp_extract

from datetime import datetime, date



# COMMAND ----------

# MAGIC %md
# MAGIC #### Carga de Fuente

# COMMAND ----------

int_pedidos_clientes = (spark.read.parquet("/Volumes/dbw_prod_aavanzada/db_tmp/files/pburbano/data/")
                                  .withColumn("fecha_pedido_dt", F.to_date(F.col("fecha_pedido_dt")))
                        )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Prediccion del Canal del Ultimo Pedido del Cliente

# COMMAND ----------

# MAGIC %md
# MAGIC #### Creacion de MDT 

# COMMAND ----------

# Definir ventana ordenada por fecha por cliente
window_cliente = Window.partitionBy("cliente_id")

# Obtener la última fecha por cliente
df = int_pedidos_clientes.withColumn("fecha_ultimo_pedido", F.max("fecha_pedido_dt").over(window_cliente))

# Marcar si es el último pedido
df = df.withColumn("es_ultimo", F.when(F.col("fecha_pedido_dt") == F.col("fecha_ultimo_pedido"), 1).otherwise(0))

# Crear el target 
df_target = df.filter("es_ultimo = 1").select(
    "cliente_id",
    F.when(F.col("canal_pedido_cd") == "DIGITAL", 1).otherwise(0).alias("target")
)

# eliminar clientes con mas de un pedido en la ultima fecha para evitar ambiguedades (es slo el 0.76% de los clientes)
df_target_unica = (df_target.groupBy("cliente_id").agg(F.count("*").alias("n_pedidos_ultima_fecha"))
                            .filter("n_pedidos_ultima_fecha = 1")
                            .drop("n_pedidos_ultima_fecha")      
                  )  
            
df_target = df_target.join(df_target_unica, on="cliente_id", how="inner")

# Crear historico del cliente antes de su ultimo pedido
df_historico = (df.filter(F.col("es_ultimo") == 0)
                  .join(df_target.select("cliente_id"), "cliente_id", "inner")   
                  .withColumn("dias_antes_ultimo",F.datediff(F.col("fecha_ultimo_pedido"), F.col("fecha_pedido_dt")))   
                  .withColumn("canal_pedido_cd", F.when(F.col("canal_pedido_cd") == "DIGITAL", "DIGITAL").otherwise("NO_DIGITAL"))
                )

df_historico = df_historico.repartition("cliente_id").persist(StorageLevel.MEMORY_AND_DISK)

# Número total de pedidos anteriores
f_pedidos = df_historico.groupBy("cliente_id").agg(
    F.count("*").alias("n_pedidos_previos"),
    F.countDistinct("canal_pedido_cd").alias("n_canales_utilizados")
)

# calcular días entre pedidos
w_orden = Window.partitionBy("cliente_id").orderBy("fecha_pedido_dt")
df_historico = df_historico.withColumn("dias_entre_pedidos", F.datediff("fecha_pedido_dt", F.lag("fecha_pedido_dt").over(w_orden)))

# agregar canal previo y cambio de canal
df_historico = df_historico.withColumn("canal_previo", F.lag("canal_pedido_cd").over(w_orden))
df_historico = df_historico.withColumn("cambio_canal", F.when(F.col("canal_previo") != F.col("canal_pedido_cd"), 1).otherwise(0))

# cambios hacia/desde digital
df_historico = df_historico.withColumn(
    "cambio_a_digital",
    F.when((F.col("canal_previo") != "DIGITAL") & (F.col("canal_pedido_cd") == "DIGITAL"), 1).otherwise(0)
)
df_historico = df_historico.withColumn(
    "cambio_desde_digital",
    F.when((F.col("canal_previo") == "DIGITAL") & (F.col("canal_pedido_cd") != "DIGITAL"), 1).otherwise(0)
)

f_frecuencia = df_historico.groupBy("cliente_id").agg(
    F.mean("dias_entre_pedidos").alias("dias_entre_pedidos_mean"),
    F.expr("percentile(dias_entre_pedidos, 0.5)").alias("dias_entre_pedidos_median"),
    F.min("dias_entre_pedidos").alias("dias_entre_pedidos_min"),
    F.max("dias_entre_pedidos").alias("dias_entre_pedidos_max")
)

# Conteo de cada canal
f_canales = df_historico.groupBy("cliente_id").pivot("canal_pedido_cd").count().fillna(0)
f_canales = (f_canales.withColumnRenamed("DIGITAL", "n_digital")
                     .withColumnRenamed("NO_DIGITAL", "n_no_digital")
            )

# Proporción de pedidos digitales
f_canales = f_canales.withColumn("prop_digital", F.col("n_digital")/(F.col("n_digital") + F.col("n_no_digital")))
            
# variables numericas
f_valores = (df_historico.groupBy("cliente_id").agg(
    F.sum("facturacion_usd_val").alias("facturacion_total"),
    F.avg("facturacion_usd_val").alias("facturacion_prom"),
    F.stddev("facturacion_usd_val").alias("facturacion_std"),
    F.sum("materiales_distintos_val").alias("materiales_distintos_total"),
    F.avg("materiales_distintos_val").alias("materiales_prom"),
    F.stddev("materiales_distintos_val").alias("materiales_std"),
    F.sum("cajas_fisicas").alias("cajas_fisicas_total"),
    F.avg("cajas_fisicas").alias("cajas_fisicas_prom"),
    F.stddev("cajas_fisicas").alias("cajas_fisicas_std"),
)
.fillna(0, subset=["facturacion_std", "materiales_std", "cajas_fisicas_std"])
)

# Canal mas reciente utilizado
f_ultimo_canal = df_historico.groupBy("cliente_id").agg(
    F.first("canal_pedido_cd", ignorenulls=True).alias("canal_mas_reciente")
)

# agregados de cambio de canal
f_cambio_canal = df_historico.groupBy("cliente_id").agg(
    F.sum("cambio_canal").alias("n_cambios_canal"),
    F.mean("cambio_canal").alias("prop_cambios_canal"),
    F.sum("cambio_a_digital").alias("n_cambios_a_digital"),
    F.sum("cambio_desde_digital").alias("n_cambios_desde_digital"),
    (F.sum("cambio_a_digital") / (F.sum("cambio_canal") + F.lit(1))).alias("prop_cambios_a_digital")
)

# variables temporales
df_historico = df_historico.withColumn("en_30d", F.when(F.col("dias_antes_ultimo") <= 30, 1).otherwise(0))
df_historico = df_historico.withColumn("en_60d", F.when(F.col("dias_antes_ultimo") <= 60, 1).otherwise(0))
df_historico = df_historico.withColumn("en_90d", F.when(F.col("dias_antes_ultimo") <= 90, 1).otherwise(0))

# Crear agregados por cliente
f_ventanas = (df_historico.groupBy("cliente_id").agg(
    F.min("dias_antes_ultimo").alias("dias_desde_ultimo_pedido"),

    F.sum("en_30d").alias("n_pedidos_ult_30d"),
    F.sum("en_60d").alias("n_pedidos_ult_60d"),
    F.sum("en_90d").alias("n_pedidos_ult_90d"),

    F.sum(F.when((F.col("en_30d") == 1), F.col("facturacion_usd_val"))).alias("facturacion_ult_30d"),
    F.sum(F.when((F.col("en_60d") == 1), F.col("facturacion_usd_val"))).alias("facturacion_ult_60d"),
    F.sum(F.when((F.col("en_90d") == 1), F.col("facturacion_usd_val"))).alias("facturacion_ult_90d"),
    
    F.max(F.when((F.col("en_30d") == 1) & (F.col("canal_pedido_cd") == "DIGITAL"), 1).otherwise(0)).alias("uso_digital_ult_30d"),
    F.max(F.when((F.col("en_60d") == 1) & (F.col("canal_pedido_cd") == "DIGITAL"), 1).otherwise(0)).alias("uso_digital_ult_60d"),
    F.max(F.when((F.col("en_90d") == 1) & (F.col("canal_pedido_cd") == "DIGITAL"), 1).otherwise(0)).alias("uso_digital_ult_90d")
)
.fillna(0, subset=["facturacion_ult_30d", "facturacion_ult_60d", "facturacion_ult_90d"])
)

w = Window.partitionBy("cliente_id").orderBy(F.desc("fecha_pedido_dt"))
df_last3 = (
    df_historico.withColumn("rn", F.row_number().over(w))
    .filter(F.col("rn") <= 3)
    .groupBy("cliente_id")
    .agg(
        F.avg("facturacion_usd_val").alias("facturacion_last3"),
        F.avg("materiales_distintos_val").alias("materiales_last3"),
        F.avg("cajas_fisicas").alias("cajas_last3")
    )
)

f_tendencias = df_last3.join(
    f_valores.select("cliente_id", "facturacion_prom", "materiales_prom", "cajas_fisicas_prom"),"cliente_id","left"
)
f_tendencias = (f_tendencias.withColumn("tendencia_facturacion", F.col("facturacion_last3") / (F.col("facturacion_prom") + F.lit(1)))
                           .withColumn("tendencia_materiales", F.col("materiales_last3") / (F.col("materiales_prom") + F.lit(1)))
                           .withColumn("tendencia_cajas", F.col("cajas_last3") / (F.col("cajas_fisicas_prom") + F.lit(1)))
                )

# variables que no cambian en el tiempo
f_variables_fijas = (df_historico.groupBy("cliente_id").agg(
    F.first("pais_cd").alias("pais_cd"),
    F.first("region_comercial_txt").alias("region_comercial_txt"),
    F.first("tipo_cliente_cd").alias("tipo_cliente_cd"),
    F.first("madurez_digital_cd").alias("madurez_digital_val"),
    F.first("estrellas_txt").cast("int").alias("estrellas_val"), 
    F.length(F.first("frecuencia_visitas_cd")).alias("frecuencia_visitas_val"),
    F.first("fecha_ultimo_pedido").cast("timestamp").alias("fecha_ultimo_pedido")
)) 

# tendencia en uso digital
df_historico = df_historico.withColumn("canal_digital_bin", F.when(F.col("canal_pedido_cd") == "DIGITAL", 1).otherwise(0))

df_last3_digital = (
    df_historico.withColumn("rn", F.row_number().over(w))
    .filter(F.col("rn") <= 3)
    .groupBy("cliente_id")
    .agg(F.avg("canal_digital_bin").alias("uso_digital_last3"))
)

f_tendencia_digital = df_last3_digital.join(
    f_canales.select("cliente_id", F.col("prop_digital").alias("prop_digital_ref")),
    "cliente_id",
    "left"
)
f_tendencia_digital = f_tendencia_digital.fillna({"uso_digital_last3": 0, "prop_digital_ref": 0})
f_tendencia_digital = f_tendencia_digital.withColumn(
    "tendencia_digital", F.when(F.col("uso_digital_last3") > F.col("prop_digital_ref"), 1).otherwise(0)
)

# informacion de agencia y ruta
df_historico = (df_historico.withColumn("agencia_num", regexp_extract(F.col("agencia_id"), r"A(\d+)$", 1).cast("int"))
                            .withColumn("ruta_num", regexp_extract(F.col("ruta_id"), r"R(\d+)$", 1).cast("int"))
)

f_ruta_agencia = (df_historico.groupBy("cliente_id").agg(
    F.first("agencia_num").alias("agencia_num"),
    F.first("ruta_num").alias("ruta_num")
))                                          

# mdt final
df_mdt = (f_pedidos
          .join(f_canales, "cliente_id", "left")
          .join(f_valores, "cliente_id", "left")
          .join(f_frecuencia, "cliente_id", "left")
          .join(f_ventanas, "cliente_id", "left")
          .join(f_tendencias.select("cliente_id", "tendencia_facturacion", "tendencia_materiales", "tendencia_cajas"), "cliente_id", "left")
          .join(f_tendencia_digital.select("cliente_id", "uso_digital_last3", "tendencia_digital"), "cliente_id", "left")
          .join(f_ultimo_canal, "cliente_id", "left")
          .join(f_cambio_canal, "cliente_id", "left")  
          .join(f_ruta_agencia, "cliente_id", "left")
          .join(f_variables_fijas, "cliente_id", "left")
          .join(df_target, "cliente_id", "inner")
)


# COMMAND ----------

# MAGIC %md
# MAGIC #### Modelamiento

# COMMAND ----------

df_mdt_pd = df_mdt.toPandas()

# COMMAND ----------


df_features_pd = df_mdt_pd.copy()

df_features_pd = df_features_pd.fillna(0)

# df_features_pd = df_features_pd[df_features_pd["madurez_digital_val"] == 'BAJA']
# df_features_pd["madurez_digital_baja"] = (df_features_pd["madurez_digital_val"] == "BAJA").astype(int)
# df_features_pd = df_features_pd.drop(columns=["madurez_digital_val"])

#df_features_pd["madurez_digital_no_alta"] = (df_features_pd["madurez_digital_val"].isin(["BAJA", "MEDIA"])).astype(int)
df_features_pd = df_features_pd.drop(columns=["madurez_digital_val"])

df_features_pd["fecha_ultimo_pedido"] = df_features_pd["fecha_ultimo_pedido"].dt.date

df_train = df_features_pd[df_features_pd["fecha_ultimo_pedido"] < date(2024,8,1)]
df_test = df_features_pd[df_features_pd["fecha_ultimo_pedido"] >= date(2024,8,1)]

# COMMAND ----------

df_test.shape[0] / (df_test.shape[0] + df_train.shape[0])

# COMMAND ----------

df_train["target"].value_counts(normalize=True)

# COMMAND ----------

df_test["target"].value_counts(normalize=True)

# COMMAND ----------

data_train = df_train.copy()
data_test = df_test.copy()

cols_exclude = ["cliente_id", "fecha_ultimo_pedido", "target"]

X_train = data_train.drop(cols_exclude, axis=1)
X_test = data_test.drop(cols_exclude, axis=1)

y_train = data_train["target"]
y_test = data_test["target"]

categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

# Asegúrate de que las columnas categóricas estén como category en ambos datasets
for col in categorical_cols:
    X_train[col] = X_train[col].astype("category")
    X_test[col] = X_test[col].astype("category")

import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, roc_auc_score

# Crear modelo
model = LGBMClassifier(
    objective="binary",
    boosting_type="gbdt",
    n_estimators=2000,
    learning_rate=0.01,
    max_depth=6,
    num_leaves=63,
    subsample=0.8,
    colsample_bytree=0.8,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

# Callbacks para early stopping y logging
callbacks = [
    lgb.early_stopping(stopping_rounds=10),
    lgb.log_evaluation(period=0)  # silencia el output
]

# Entrenar (sin early_stopping_rounds directo)
model.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    eval_metric="auc",
    callbacks=callbacks,
    categorical_feature=categorical_cols  # puedes pasar aquí tus columnas categóricas
)

# COMMAND ----------

# Evaluación Train
y_pred_train = model.predict(X_train)
y_proba_train = model.predict_proba(X_train)[:, 1]

print("ROC AUC:", roc_auc_score(y_train, y_proba_train))
print("Classification report:\n", classification_report(y_train, y_pred_train))

# COMMAND ----------

# Evaluación Test
y_pred_test = model.predict(X_test)
y_proba_test = model.predict_proba(X_test)[:, 1]

print("ROC AUC:", roc_auc_score(y_test, y_proba_test))
print("Classification report:\n", classification_report(y_test, y_pred_test))

# COMMAND ----------

# Suponiendo que ya tienes tu modelo entrenado
lgb.plot_importance(model, max_num_features=30, importance_type='gain')  # o 'split'
plt.title("Importancia de variables")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Prediccion Del Canal del Siguiente Pedido del Cliente

# COMMAND ----------

# DBTITLE 1,mdt final
df = int_pedidos_clientes

# Transformaciones

# Convertir madurez_digital_cd a ordinal numérica
df = df.withColumn(
    "madurez_digital_val",
    F.when(F.col("madurez_digital_cd") == "BAJA", 1)
     .when(F.col("madurez_digital_cd") == "MEDIA", 2)
     .when(F.col("madurez_digital_cd") == "ALTA", 3)
     .otherwise(None)  
)
# Convertir estrellas_txt a numérica
df = df.withColumn("estrellas_val", F.col("estrellas_txt").cast("int"))

# Convertir frecuencia_visitas_cd a cantidad de letras (número de días)
df = df.withColumn("frecuencia_visitas_val", F.length(F.col("frecuencia_visitas_cd")))

# === 1. Definición de ventanas ===
w = Window.partitionBy("cliente_id").orderBy(F.asc("fecha_pedido_dt"))
w_prev_all = w.rowsBetween(Window.unboundedPreceding, -1)  # Hasta el anterior
w_recent = w.rowsBetween(-3, -1)  # Últimos 3 anteriores

# === 2. Base con canal_siguiente y target ===
df = df.withColumn("canal_siguiente", F.lead("canal_pedido_cd").over(w))
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
df = (#
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
        "antiguedad_dias", "fecha_pedido_dt",
        # Transformaciones 
        "madurez_digital_val", "estrellas_val", "frecuencia_visitas_val"
      )
)

# === 10. Etiquetar periodo y limpiar ===
fecha_corte = "2024-03-01"
mdt = mdt.withColumn("periodo", F.when(F.col("fecha_pedido_dt") < fecha_corte, "TRAIN").otherwise("TEST"))
mdt = mdt.fillna(0)

# 4. Proporción de pedidos digitales por agencia_id (solo con df_train)
prop_agencia = mdt.filter("periodo == 'TRAIN'").groupBy("agencia_id").agg(F.avg(F.col("target")).alias("prop_digital_agencia"))

# 5. Proporción de pedidos digitales por ruta_id
prop_ruta = mdt.filter("periodo == 'TRAIN'").groupBy("ruta_id").agg(F.avg(F.col("target")).alias("prop_digital_ruta"))

# 6. Join a df_train y df_test
mdt = mdt.join(prop_agencia, on="agencia_id", how="left")
mdt = mdt.join(prop_ruta, on="ruta_id", how="left")

# === 11. Exportar a pandas (opcional) ===
mdt_pd = mdt.toPandas()


# COMMAND ----------

# === 0. Base ===
df = int_pedidos_clientes

# === 1. Transformaciones iniciales ===
df = (
    df.withColumn("madurez_digital_val", 
                  F.when(F.col("madurez_digital_cd") == "BAJA", 1)
                   .when(F.col("madurez_digital_cd") == "MEDIA", 2)
                   .when(F.col("madurez_digital_cd") == "ALTA", 3))
      .withColumn("estrellas_val", F.col("estrellas_txt").cast("int"))
      .withColumn("frecuencia_visitas_val", F.length(F.col("frecuencia_visitas_cd")))
)

# === 2. Definición de ventanas ===
w = Window.partitionBy("cliente_id").orderBy(F.asc("fecha_pedido_dt"))
w_prev_all = w.rowsBetween(Window.unboundedPreceding, -1)
w_recent = w.rowsBetween(-3, -1)

# === 3. Target: canal siguiente ===
df = (
    df.withColumn("canal_siguiente", F.lead("canal_pedido_cd").over(w))
      .filter(F.col("canal_siguiente").isNotNull())
      .withColumn("target", F.when(F.col("canal_siguiente") == "DIGITAL", 1).otherwise(0))
)

# === 4. Variables del canal actual y anterior ===
df = (
    df.withColumn("canal_previo", F.lag("canal_pedido_cd").over(w))
      .withColumn("canal_actual", F.col("canal_pedido_cd"))
      .withColumn("canal_actual_digital", F.when(F.col("canal_actual") == "DIGITAL", 1).otherwise(0))
      .withColumn("canal_previo_digital", F.when(F.col("canal_previo") == "DIGITAL", 1).otherwise(0))
)

# === 5. Variables históricas (hasta pedido anterior) ===
df = (
    df.withColumn("dias_desde_pedido_anterior", F.datediff("fecha_pedido_dt", F.lag("fecha_pedido_dt").over(w)))
      .withColumn("n_pedidos_previos", F.row_number().over(w) - 1)
      .withColumn("facturacion_prom_anterior", F.avg("facturacion_usd_val").over(w_prev_all))
      .withColumn("facturacion_total_prev", F.sum("facturacion_usd_val").over(w_prev_all))
      .withColumn("desviacion_facturacion", F.stddev("facturacion_usd_val").over(w_prev_all))
      .withColumn("uso_digital_prev", F.sum(F.when(F.col("canal_pedido_cd") == "DIGITAL", 1).otherwise(0)).over(w_prev_all))
      .withColumn("prop_digital_prev", F.col("uso_digital_prev") / F.when(F.col("n_pedidos_previos") > 0, F.col("n_pedidos_previos")).otherwise(1))
      .withColumn("dias_media_prev", F.avg(F.datediff("fecha_pedido_dt", F.lag("fecha_pedido_dt").over(w))).over(w_prev_all))
      .withColumn("dias_media_std", F.stddev(F.datediff("fecha_pedido_dt", F.lag("fecha_pedido_dt").over(w))).over(w_prev_all))
)

# === 6. Variables recientes (últimos 3 pedidos) ===
df = (
    df.withColumn("facturacion_prom_reciente", F.avg("facturacion_usd_val").over(w_recent))
      .withColumn("uso_digital_reciente", F.avg(F.when(F.col("canal_pedido_cd") == "DIGITAL", 1).otherwise(0)).over(w_recent))
)

# === 7. Materiales y cajas ===
df = (
    df.withColumn("materiales_prom_prev", F.avg("materiales_distintos_val").over(w_prev_all))
      .withColumn("materiales_total_prev", F.sum("materiales_distintos_val").over(w_prev_all))
      .withColumn("cajas_fisicas_prom_prev", F.avg("cajas_fisicas").over(w_prev_all))
      .withColumn("cajas_fisicas_total_prev", F.sum("cajas_fisicas").over(w_prev_all))
      .withColumn("materiales_reciente", F.avg("materiales_distintos_val").over(w_recent))
      .withColumn("cajas_fisicas_reciente", F.avg("cajas_fisicas").over(w_recent))
      .withColumn("cajas_por_material", 
                  F.when(F.col("materiales_distintos_val") > 0, F.col("cajas_fisicas") / F.col("materiales_distintos_val")).otherwise(0))
      .withColumn("cajas_por_material_prev", 
                  F.avg(F.when(F.col("materiales_distintos_val") > 0, 
                               F.col("cajas_fisicas") / F.col("materiales_distintos_val"))
                        .otherwise(0)).over(w_prev_all))
)

# === 8. Temporalidad ===
df = (
    df.withColumn("mes", F.month("fecha_pedido_dt"))
      .withColumn("dia_semana", F.dayofweek("fecha_pedido_dt"))
      .withColumn("es_fin_de_semana", F.when(F.col("dia_semana").isin(1, 7), 1).otherwise(0))
      .withColumn("trimestre", F.quarter("fecha_pedido_dt"))
)

# === 9. Antigüedad del cliente ===
df = df.withColumn("antiguedad_dias", F.datediff("fecha_pedido_dt", F.min("fecha_pedido_dt").over(Window.partitionBy("cliente_id"))))

# === 10. Construcción de MDT (selección de variables) ===
mdt = (
    df.filter(F.col("n_pedidos_previos") > 0)
      .filter(F.col("target").isNotNull())
      .select(
        # IDs
        "cliente_id", "pais_cd", "region_comercial_txt", "agencia_id", "ruta_id",
        "tipo_cliente_cd", 
        #"madurez_digital_cd", "estrellas_txt", "frecuencia_visitas_cd",
        # Target
        "target",
        # Canal
        "canal_actual_digital", "canal_previo_digital",
        # Comportamiento histórico
        "facturacion_usd_val", "dias_desde_pedido_anterior", "n_pedidos_previos",
        "facturacion_prom_anterior", "facturacion_total_prev", "desviacion_facturacion",
        "uso_digital_prev", "prop_digital_prev", 
        "facturacion_prom_reciente", "uso_digital_reciente",
        "dias_media_prev", "dias_media_std",
        # Materiales y cajas
        "materiales_distintos_val", "materiales_prom_prev", "materiales_total_prev", "materiales_reciente",
        "cajas_fisicas", "cajas_fisicas_prom_prev", "cajas_fisicas_total_prev", "cajas_fisicas_reciente",
        "cajas_por_material", "cajas_por_material_prev",
        # Temporalidad
        "mes", "dia_semana", "es_fin_de_semana", "trimestre",
        # Antigüedad y transformaciones
        "antiguedad_dias", "fecha_pedido_dt",
        "madurez_digital_val", "estrellas_val", "frecuencia_visitas_val"
      )
)

# === 11. Etiquetado de periodo ===
fecha_corte = "2024-03-01"
mdt = mdt.withColumn("periodo", F.when(F.col("fecha_pedido_dt") < fecha_corte, "TRAIN").otherwise("TEST"))

# === 12. Cálculo de proporciones por agencia y ruta ===
prop_agencia = (
    mdt.filter(F.col("periodo") == "TRAIN")
       .groupBy("agencia_id")
       .agg(F.avg("target").alias("prop_digital_agencia"))
)

prop_ruta = (
    mdt.filter(F.col("periodo") == "TRAIN")
       .groupBy("ruta_id")
       .agg(F.avg("target").alias("prop_digital_ruta"))
)

# === 13. Merge proporciones a toda la MDT ===
mdt = mdt.join(prop_agencia, on="agencia_id", how="left").drop("agencia_id")
mdt = mdt.join(prop_ruta, on="ruta_id", how="left").drop("ruta_id")

# === 14. Rellenar nulos y exportar (opcional) ===
mdt = mdt.fillna(0)
mdt_pd = mdt.toPandas()

# COMMAND ----------

mdt_pd["periodo"].value_counts(normalize=True)

# COMMAND ----------

mdt_pd.groupby("periodo")["target"].value_counts(normalize=True)

# COMMAND ----------

mdt_pd.groupby("periodo")["madurez_digital_val"].value_counts(normalize=True)

# COMMAND ----------

from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, roc_auc_score
import lightgbm as lgb

# Convertir a pandas
df_binaria_pd = mdt_pd.copy()

# top_20 = df_binaria_pd['ruta_id'].value_counts().nlargest(20).index
# df_binaria_pd['ruta_id'] = df_binaria_pd['ruta_id'].where(df_binaria_pd['ruta_id'].isin(top_20), 'otra_ruta')

#df_binaria_pd["madurez_x_prop_digital"] = df_binaria_pd["prop_digital_prev"] * df_binaria_pd["madurez_digital_val"]
df_binaria_pd["madurez_x_uso_digital"] = df_binaria_pd["uso_digital_prev"] * df_binaria_pd["madurez_digital_val"]
df_binaria_pd["estrellas_val_x_uso_digital"] = df_binaria_pd["estrellas_val"] * df_binaria_pd["uso_digital_reciente"]


df_binaria_pd = df_binaria_pd.drop(columns=["madurez_digital_val", "estrellas_val"]).copy()

#df_binaria_pd["estrellas_x_prop_digital"] = df_binaria_pd["estrellas_val"] * df_binaria_pd["prop_digital_prev"]
#df_binaria_pd = df_binaria_pd.drop(columns=["estrellas_txt"]).copy()

# Multiplicaciones que capturan no linealidades importantes
# df_binaria_pd["estrellas_x_prop_digital"] = df_binaria_pd["estrellas_val"] * df_binaria_pd["prop_digital_prev"]
# df_binaria_pd["madurez_x_uso_digital"] = df_binaria_pd["madurez_digital_val"] * df_binaria_pd["uso_digital_prev"]
# df_binaria_pd["n_pedidos_x_prop_digital"] = df_binaria_pd["n_pedidos_previos"] * df_binaria_pd["prop_digital_prev"]

#df_binaria_pd = df_binaria_pd.drop(["madurez_digital_val", "estrellas_val"], axis=1)

# Dividir en Train/Test según columna 'periodo'
df_train = df_binaria_pd[df_binaria_pd["periodo"] == "TRAIN"].copy()
df_test  = df_binaria_pd[df_binaria_pd["periodo"] == "TEST"].copy()

# Separar X e y

X_train = df_train.drop(columns=["target", "periodo", "fecha_pedido_dt", "cliente_id"])
y_train = df_train["target"]

X_test = df_test.drop(columns=["target", "periodo", "fecha_pedido_dt", "cliente_id"])
y_test = df_test["target"]

# Identificar columnas categóricas
cat_cols = X_train.select_dtypes(include="object").columns.tolist()

# Asegurar que estén en formato 'category'
for col in cat_cols:
    X_train[col] = X_train[col].astype("category")
    X_test[col] = X_test[col].astype("category")

# Modelo LightGBM binario
model = LGBMClassifier(
    objective='binary',
    learning_rate=0.03,         # más suave que 0.05 para mejor generalización
    n_estimators=1500,          # más iteraciones con early stopping
    max_depth=8,                # un poco más profundo para capturar interacciones
    num_leaves=63,              # 2^6 o 2^7-1, consistente con max_depth
    min_child_samples=100,      # evita splits sobre pocas filas
    subsample=0.8,              # por robustez (bagging)
    colsample_bytree=0.8,       # por robustez (feature sampling)
    feature_fraction=0.8,       # evita dominancia de una sola variable
    reg_alpha=1.5,              # penalización L1 (sparse model)
    reg_lambda=1.5,             # penalización L2 (evita sobreajuste)
    class_weight='balanced',    # tus clases están relativamente balanceadas pero aún ayuda
    importance_type='gain',     # mide importancia por ganancia, no frecuencia
    random_state=42,
    verbosity=-1
)

# Entrenamiento con early stopping
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric="auc",
    callbacks=[
        lgb.early_stopping(stopping_rounds=20),
        lgb.log_evaluation(20)
    ],
    categorical_feature=cat_cols
)

# COMMAND ----------

# Evaluación Train
y_pred_train = model.predict(X_train)
y_proba_train = model.predict_proba(X_train)[:, 1]

print("ROC AUC:", roc_auc_score(y_train, y_proba_train))
print("Classification report:\n", classification_report(y_train, y_pred_train))

# COMMAND ----------

# Evaluación Test
y_pred_test = model.predict(X_test)
y_proba_test = model.predict_proba(X_test)[:, 1]

print("ROC AUC:", roc_auc_score(y_test, y_proba_test))
print("Classification report:\n", classification_report(y_test, y_pred_test))


# COMMAND ----------

import matplotlib.pyplot as plt
import pandas as pd

feature_imp = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
}).sort_values(by="importance", ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(feature_imp['feature'][:20][::-1], feature_imp['importance'][:20][::-1])
plt.title("Top 20 Feature Importances")
plt.xlabel("Importance (gain)")
plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,data drift
import seaborn as sns
import matplotlib.pyplot as plt

for col in numericas:
    plt.figure()
    sns.kdeplot(data=mdt_pd, x=col, hue="periodo", common_norm=False)
    plt.title(f"Distribución de {col} por periodo")
    plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Modelos Por Nivel de Madurez

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Baja

# COMMAND ----------

df_mdt_baja = mdt_pd[mdt_pd["madurez_digital_val"] == 1].copy()

# COMMAND ----------

df_mdt_baja["periodo"].value_counts(normalize=True)

# COMMAND ----------

df_mdt_baja.groupby("periodo")["target"].value_counts(normalize=True)

# COMMAND ----------

# Dividir en Train/Test según columna 'periodo'
df_train = df_mdt_baja[df_mdt_baja["periodo"] == "TRAIN"].copy()
df_test  = df_mdt_baja[df_mdt_baja["periodo"] == "TEST"].copy()

# Separar X e y

X_train = df_train.drop(columns=["target", "periodo", "fecha_pedido_dt", "cliente_id", "prop_digital_ruta"])
y_train = df_train["target"]

X_test = df_test.drop(columns=["target", "periodo", "fecha_pedido_dt", "cliente_id", "prop_digital_ruta"])
y_test = df_test["target"]

# Identificar columnas categóricas
cat_cols = X_train.select_dtypes(include="object").columns.tolist()

# Asegurar que estén en formato 'category'
for col in cat_cols:
    X_train[col] = X_train[col].astype("category")
    X_test[col] = X_test[col].astype("category")

# Modelo LightGBM binario
model = LGBMClassifier(
    objective='binary',
    learning_rate=0.03,         # más suave que 0.05 para mejor generalización
    n_estimators=1500,          # más iteraciones con early stopping
    max_depth=8,                # un poco más profundo para capturar interacciones
    num_leaves=63,              # 2^6 o 2^7-1, consistente con max_depth
    min_child_samples=100,      # evita splits sobre pocas filas
    subsample=0.8,              # por robustez (bagging)
    colsample_bytree=0.8,       # por robustez (feature sampling)
    feature_fraction=0.8,       # evita dominancia de una sola variable
    reg_alpha=1.5,              # penalización L1 (sparse model)
    reg_lambda=1.5,             # penalización L2 (evita sobreajuste)
    class_weight='balanced',    # tus clases están relativamente balanceadas pero aún ayuda
    importance_type='gain',     # mide importancia por ganancia, no frecuencia
    random_state=42,
    verbosity=-1
)

# Entrenamiento con early stopping
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric="auc",
    callbacks=[
        lgb.early_stopping(stopping_rounds=20),
        lgb.log_evaluation(20)
    ],
    categorical_feature=cat_cols
)

# COMMAND ----------

# Evaluación Train
y_pred_train = model.predict(X_train)
y_proba_train = model.predict_proba(X_train)[:, 1]

print("ROC AUC:", roc_auc_score(y_train, y_proba_train))
print("Classification report:\n", classification_report(y_train, y_pred_train))

# COMMAND ----------

# Evaluación Test
y_pred_test = model.predict(X_test)
y_proba_test = model.predict_proba(X_test)[:, 1]

print("ROC AUC:", roc_auc_score(y_test, y_proba_test))
print("Classification report:\n", classification_report(y_test, y_pred_test))

# COMMAND ----------

import matplotlib.pyplot as plt
import pandas as pd

feature_imp = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
}).sort_values(by="importance", ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(feature_imp['feature'][:20][::-1], feature_imp['importance'][:20][::-1])
plt.title("Top 20 Feature Importances")
plt.xlabel("Importance (gain)")
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Random Search de Hiperpaarmetros

# COMMAND ----------

from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score, classification_report
from scipy.stats import randint, uniform
from sklearn.model_selection import train_test_split
import time

# 🔹 1. (Opcional) Submuestreo para acelerar búsqueda
X_sample, _, y_sample, _ = train_test_split(X_train, y_train, 
                                            train_size=0.3, 
                                            stratify=y_train, 
                                            random_state=42)

# 🔹 2. Hiperparámetros ajustados para búsquedas más rápidas
param_dist = {
    'learning_rate': uniform(0.02, 0.05),
    'n_estimators': randint(300, 800),
    'max_depth': randint(5, 9),
    'num_leaves': randint(31, 100),
    'min_child_samples': randint(20, 100),
    'subsample': uniform(0.7, 0.3),
    'colsample_bytree': uniform(0.7, 0.3),
    'reg_alpha': uniform(0, 2),
    'reg_lambda': uniform(0, 2),
    'feature_fraction': uniform(0.7, 0.3)
}

# 🔹 3. Modelo base
model = LGBMClassifier(
    objective='binary',
    class_weight='balanced',
    importance_type='gain',
    random_state=42,
    verbosity=-1
)

# 🔹 4. RandomizedSearchCV con parámetros optimizados
start = time.time()
search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=10,           # 🔹 Menos combinaciones
    scoring='roc_auc',
    cv=2,                # 🔹 Menos folds
    verbose=1,
    n_jobs=-1,
    random_state=42
)

search.fit(X_sample, y_sample, categorical_feature=cat_cols)
end = time.time()

# 🔹 5. Resultados de búsqueda
print(f"\n🔍 RandomizedSearchCV terminado en {end - start:.2f} segundos")
print("🧠 Mejor combinación encontrada:")
print(search.best_params_)
print(f"🏆 Mejor AUC promedio en CV: {search.best_score_:.4f}")

# 🔹 6. Entrenar modelo final con todos los datos
best_model = search.best_estimator_
best_model.fit(X_train, y_train, categorical_feature=cat_cols)

# 🔹 7. Evaluar en test
y_proba_test = best_model.predict_proba(X_test)[:, 1]
y_pred_test = best_model.predict(X_test)

print("\n✅ ROC AUC test:", roc_auc_score(y_test, y_proba_test))
print("📊 Classification report:\n", classification_report(y_test, y_pred_test))


# COMMAND ----------

# MAGIC %md
# MAGIC #### Puebas Catboost

# COMMAND ----------

from catboost import CatBoostClassifier, Pool
from sklearn.metrics import classification_report, roc_auc_score

# Convertir a pandas
df_binaria_pd = mdt_pd.copy()

# Agrupar 'ruta_id' fuera del top 20 como 'otra_ruta'
top_20 = df_binaria_pd['ruta_id'].value_counts().nlargest(20).index
df_binaria_pd['ruta_id'] = df_binaria_pd['ruta_id'].where(df_binaria_pd['ruta_id'].isin(top_20), 'otra_ruta')

# Split temporal
df_train = df_binaria_pd[df_binaria_pd["periodo"] == "TRAIN"].copy()
df_test  = df_binaria_pd[df_binaria_pd["periodo"] == "TEST"].copy()

# Features y target
#vars = ['prop_digital_prev', 'uso_telefono_prev', 'madurez_digital_cd', 'estrellas_txt', 'canal_previo']

X_train = df_train.drop(columns=["target", "periodo", "fecha_pedido_dt"])
y_train = df_train["target"]
X_test  = df_test.drop(columns=["target", "periodo", "fecha_pedido_dt"])
y_test  = df_test["target"]

# Identificar columnas categóricas
cat_cols = X_train.select_dtypes(include="object").columns.tolist()

# Entrenar modelo CatBoost
model = CatBoostClassifier(class_weights=None,
    iterations=500,
    learning_rate=0.05,
    depth=7,
    random_seed=42,
    eval_metric='AUC',
    early_stopping_rounds=20,
    verbose=50
)

model.fit(X_train, y_train, cat_features=cat_cols, eval_set=(X_test, y_test))

# Evaluación
y_proba = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

print("ROC AUC:", roc_auc_score(y_test, y_proba))
print("Classification report:\n", classification_report(y_test, y_pred))


# COMMAND ----------

from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import OrdinalEncoder

# Convertir a pandas
df_binaria_pd = mdt_pd.copy()

# Agrupar 'ruta_id'
top_20 = df_binaria_pd['ruta_id'].value_counts().nlargest(20).index
df_binaria_pd['ruta_id'] = df_binaria_pd['ruta_id'].where(df_binaria_pd['ruta_id'].isin(top_20), 'otra_ruta')

# Split temporal
df_train = df_binaria_pd[df_binaria_pd["periodo"] == "TRAIN"].copy()
df_test  = df_binaria_pd[df_binaria_pd["periodo"] == "TEST"].copy()

# Features y target
#vars = ['prop_digital_prev', 'uso_telefono_prev', 'madurez_digital_cd', 'estrellas_txt', 'canal_previo']

X_train = df_train.drop(columns=["target", "periodo", "fecha_pedido_dt"])
y_train = df_train["target"]
X_test  = df_test.drop(columns=["target", "periodo", "fecha_pedido_dt"])
y_test  = df_test["target"]

# Identificar columnas categóricas y codificarlas
cat_cols = X_train.select_dtypes(include="object").columns.tolist()
encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X_train[cat_cols] = encoder.fit_transform(X_train[cat_cols])
X_test[cat_cols] = encoder.transform(X_test[cat_cols])

# Entrenar modelo XGBoost
model = XGBClassifier(
    objective='binary:logistic',
    learning_rate=0.05,
    n_estimators=500,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.5,
    reg_lambda=0.5,
    #scale_pos_weight=2,  # puedes ajustar según proporción
    use_label_encoder=False,
    eval_metric='auc',
    random_state=42
)

model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=20, verbose=20)

# Evaluación
y_proba = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

print("ROC AUC:", roc_auc_score(y_test, y_proba))
print("Classification report:\n", classification_report(y_test, y_pred))


# COMMAND ----------

# MAGIC %md
# MAGIC #### Multiclase

# COMMAND ----------

# === 0. Base ===
df = int_pedidos_clientes

# === 1. Transformaciones iniciales ===
df = (
    df.withColumn("madurez_digital_val", 
                  F.when(F.col("madurez_digital_cd") == "BAJA", 1)
                   .when(F.col("madurez_digital_cd") == "MEDIA", 2)
                   .when(F.col("madurez_digital_cd") == "ALTA", 3))
      .withColumn("estrellas_val", F.col("estrellas_txt").cast("int"))
      .withColumn("frecuencia_visitas_val", F.length(F.col("frecuencia_visitas_cd")))
)

# === 2. Definición de ventanas ===
w = Window.partitionBy("cliente_id").orderBy(F.asc("fecha_pedido_dt"))
w_prev_all = w.rowsBetween(Window.unboundedPreceding, -1)
w_recent = w.rowsBetween(-3, -1)

# === 3. Target multiclase: canal siguiente ===
df = (
    df.withColumn("canal_siguiente", F.lead("canal_pedido_cd").over(w))
      .filter(F.col("canal_siguiente").isNotNull())
      .withColumn("target", F.col("canal_siguiente"))  # multiclase: DIGITAL, TELEFONO, VENDEDOR
)

# === 4. Variables del canal actual y anterior ===
df = (
    df.withColumn("canal_previo", F.lag("canal_pedido_cd").over(w))
      .withColumn("canal_actual", F.col("canal_pedido_cd"))
      .withColumn("canal_actual_digital", F.when(F.col("canal_actual") == "DIGITAL", 1).otherwise(0))
      .withColumn("canal_previo_digital", F.when(F.col("canal_previo") == "DIGITAL", 1).otherwise(0))
)

# === 5. Variables históricas ===
df = (
    df.withColumn("dias_desde_pedido_anterior", F.datediff("fecha_pedido_dt", F.lag("fecha_pedido_dt").over(w)))
      .withColumn("n_pedidos_previos", F.row_number().over(w) - 1)
      .withColumn("facturacion_prom_anterior", F.avg("facturacion_usd_val").over(w_prev_all))
      .withColumn("facturacion_total_prev", F.sum("facturacion_usd_val").over(w_prev_all))
      .withColumn("desviacion_facturacion", F.stddev("facturacion_usd_val").over(w_prev_all))
      .withColumn("uso_digital_prev", F.sum(F.when(F.col("canal_pedido_cd") == "DIGITAL", 1).otherwise(0)).over(w_prev_all))
      .withColumn("prop_digital_prev", F.col("uso_digital_prev") / F.when(F.col("n_pedidos_previos") > 0, F.col("n_pedidos_previos")).otherwise(1))
      .withColumn("dias_media_prev", F.avg(F.datediff("fecha_pedido_dt", F.lag("fecha_pedido_dt").over(w))).over(w_prev_all))
      .withColumn("dias_media_std", F.stddev(F.datediff("fecha_pedido_dt", F.lag("fecha_pedido_dt").over(w))).over(w_prev_all))
)

# === 6. Variables recientes ===
df = (
    df.withColumn("facturacion_prom_reciente", F.avg("facturacion_usd_val").over(w_recent))
      .withColumn("uso_digital_reciente", F.avg(F.when(F.col("canal_pedido_cd") == "DIGITAL", 1).otherwise(0)).over(w_recent))
)

# === 7. Materiales y cajas ===
df = (
    df.withColumn("materiales_prom_prev", F.avg("materiales_distintos_val").over(w_prev_all))
      .withColumn("materiales_total_prev", F.sum("materiales_distintos_val").over(w_prev_all))
      .withColumn("cajas_fisicas_prom_prev", F.avg("cajas_fisicas").over(w_prev_all))
      .withColumn("cajas_fisicas_total_prev", F.sum("cajas_fisicas").over(w_prev_all))
      .withColumn("materiales_reciente", F.avg("materiales_distintos_val").over(w_recent))
      .withColumn("cajas_fisicas_reciente", F.avg("cajas_fisicas").over(w_recent))
      .withColumn("cajas_por_material", F.when(F.col("materiales_distintos_val") > 0, F.col("cajas_fisicas") / F.col("materiales_distintos_val")).otherwise(0))
      .withColumn("cajas_por_material_prev", F.avg(F.when(F.col("materiales_distintos_val") > 0, F.col("cajas_fisicas") / F.col("materiales_distintos_val")).otherwise(0)).over(w_prev_all))
)

# === 8. Temporalidad ===
df = (
    df.withColumn("mes", F.month("fecha_pedido_dt"))
      .withColumn("dia_semana", F.dayofweek("fecha_pedido_dt"))
      .withColumn("es_fin_de_semana", F.when(F.col("dia_semana").isin(1, 7), 1).otherwise(0))
      .withColumn("trimestre", F.quarter("fecha_pedido_dt"))
)

# === 9. Antigüedad ===
df = df.withColumn("antiguedad_dias", F.datediff("fecha_pedido_dt", F.min("fecha_pedido_dt").over(Window.partitionBy("cliente_id"))))

# === 10. MDT final ===
mdt = (
    df.filter(F.col("n_pedidos_previos") > 0)
      .filter(F.col("target").isNotNull())
      .select(
        "cliente_id", "pais_cd", "region_comercial_txt", "agencia_id", "ruta_id", "tipo_cliente_cd",
        "target", "canal_actual_digital", "canal_previo_digital",
        "facturacion_usd_val", "dias_desde_pedido_anterior", "n_pedidos_previos",
        "facturacion_prom_anterior", "facturacion_total_prev", "desviacion_facturacion",
        "uso_digital_prev", "prop_digital_prev", "facturacion_prom_reciente", "uso_digital_reciente",
        "dias_media_prev", "dias_media_std",
        "materiales_distintos_val", "materiales_prom_prev", "materiales_total_prev", "materiales_reciente",
        "cajas_fisicas", "cajas_fisicas_prom_prev", "cajas_fisicas_total_prev", "cajas_fisicas_reciente",
        "cajas_por_material", "cajas_por_material_prev",
        "mes", "dia_semana", "es_fin_de_semana", "trimestre",
        "antiguedad_dias", "fecha_pedido_dt", "madurez_digital_val", "estrellas_val", "frecuencia_visitas_val"
      )
)

# === 11. Periodo TRAIN/TEST ===
fecha_corte = "2024-03-01"
mdt = mdt.withColumn("periodo", F.when(F.col("fecha_pedido_dt") < fecha_corte, "TRAIN").otherwise("TEST"))

# === 12. Proporciones por agencia y ruta ===
prop_agencia = mdt.filter(F.col("periodo") == "TRAIN").groupBy("agencia_id").agg(F.expr("percentile_approx(target, 0.5)").alias("moda_agencia"))
prop_ruta = mdt.filter(F.col("periodo") == "TRAIN").groupBy("ruta_id").agg(F.expr("percentile_approx(target, 0.5)").alias("moda_ruta"))
mdt = mdt.join(prop_agencia, on="agencia_id", how="left").drop("agencia_id")
mdt = mdt.join(prop_ruta, on="ruta_id", how="left").drop("ruta_id")

# === 13. Rellenar y exportar ===
mdt = mdt.fillna(0)
mdt_pd = mdt.toPandas()

# COMMAND ----------

df_train.head(5).display()

# COMMAND ----------

df_train["target"].value_counts(normalize=True)

# COMMAND ----------

df_test["target"].value_counts(normalize=True)

# COMMAND ----------

mdt_pd["periodo"].value_counts(normalize=True)

# COMMAND ----------

from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder

# Encode target multiclase (DIGITAL, TELEFONO, VENDEDOR)
le = LabelEncoder()
mdt_pd["target_encoded"] = le.fit_transform(mdt_pd["target"])

mdt_pd = mdt_pd[mdt_pd["madurez_digital_val"] == 3]

# División Train/Test
df_train = mdt_pd[mdt_pd["periodo"] == "TRAIN"].copy()
df_test = mdt_pd[mdt_pd["periodo"] == "TEST"].copy()

X_train = df_train.drop(columns=["target", "target_encoded", "periodo", "fecha_pedido_dt", "cliente_id"])
y_train = df_train["target_encoded"]

X_test = df_test.drop(columns=["target", "target_encoded", "periodo", "fecha_pedido_dt", "cliente_id"])
y_test = df_test["target_encoded"]

# Columnas categóricas
cat_cols = X_train.select_dtypes(include="object").columns.tolist()
for col in cat_cols:
    X_train[col] = X_train[col].astype("category")
    X_test[col] = X_test[col].astype("category")

# Modelo multiclase
model = LGBMClassifier(
    objective="multiclass",
    num_class=3,
    learning_rate=0.03,
    n_estimators=1000,
    max_depth=8,
    num_leaves=63,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    class_weight="balanced"
)

# Entrenamiento
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric="multi_logloss",
    callbacks=[lgb.early_stopping(stopping_rounds=20)]
)

# Predicción
y_pred = model.predict(X_test)

# Evaluación
print("Classification report:\n", classification_report(y_test, y_pred, target_names=le.classes_))


# COMMAND ----------

import matplotlib.pyplot as plt
import pandas as pd

feature_imp = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
}).sort_values(by="importance", ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(feature_imp['feature'][:20][::-1], feature_imp['importance'][:20][::-1])
plt.title("Top 20 Feature Importances")
plt.xlabel("Importance (gain)")
plt.tight_layout()
plt.show()


# COMMAND ----------

df = int_pedidos_clientes.toPandas()

# COMMAND ----------

import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

def prepare_lgb_features(df):
    """Preparar features específicos para LightGBM"""
    
    # Agrupar por cliente y calcular features
    client_features = df.groupby('cliente_id').agg({
        'fecha_pedido_dt': ['min', 'max', 'count'],
        'canal_pedido_cd': lambda x: (x == 'DIGITAL').mean(),
        'facturacion_usd_val': ['mean', 'std', 'sum'],
        'materiales_distintos_val': 'mean',
        'cajas_fisicas': 'mean'
    }).reset_index()
    
    client_features.columns = ['cliente_id', 'primera_compra', 'ultima_compra', 
                              'total_pedidos', 'ratio_digital_historico', 
                              'ticket_promedio', 'ticket_std', 'facturacion_total',
                              'materiales_promedio', 'cajas_promedio']
    
    # Features temporales avanzados
    client_features['antiguedad_dias'] = (client_features['ultima_compra'] - client_features['primera_compra']).dt.days
    client_features['frecuencia_pedidos'] = client_features['antiguedad_dias'] / client_features['total_pedidos']
    
    # Ratio digital últimos 3 meses
    cutoff_date = df['fecha_pedido_dt'].max() - pd.Timedelta(days=90)
    recent_digital = df[df['fecha_pedido_dt'] > cutoff_date].groupby('cliente_id').agg({
        'canal_pedido_cd': lambda x: (x == 'DIGITAL').mean()
    }).reset_index()
    recent_digital.columns = ['cliente_id', 'ratio_digital_3m']
    
    client_features = client_features.merge(recent_digital, on='cliente_id', how='left')
    client_features['ratio_digital_3m'] = client_features['ratio_digital_3m'].fillna(0)
    
    return client_features

# Preparar datos
features_df = prepare_lgb_features(df)

# Separar features y target (último pedido por cliente como target)
latest_orders = df.sort_values('fecha_pedido_dt').groupby('cliente_id').last().reset_index()
latest_orders['target'] = (latest_orders['canal_pedido_cd'] == 'DIGITAL').astype(int)

train_data = features_df.merge(latest_orders[['cliente_id', 'target']], on='cliente_id')

# Features para modelo
feature_cols = [col for col in train_data.columns if col not in ['cliente_id', 'target', 'primera_compra', 'ultima_compra']]
X = train_data[feature_cols]
y = train_data['target']

# Entrenar LightGBM
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

lgb_model = lgb.LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    n_jobs=-1
)

lgb_model.fit(X_train, y_train, 
              eval_set=[(X_test, y_test)],
              eval_metric='auc',
              #early_stopping_rounds=50,
              #verbose=50
              )

y_pred_proba = lgb_model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred_proba)
print(f"LightGBM AUC: {auc}")

# COMMAND ----------

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from datetime import timedelta

# =========================
# 1️⃣ Función para preparar features
# =========================
def prepare_lgb_features(df):
    """Preparar features históricas por cliente para LightGBM"""
    
    client_features = df.groupby('cliente_id').agg({
        'fecha_pedido_dt': ['min', 'max', 'count'],
        'canal_pedido_cd': lambda x: (x == 'DIGITAL').mean(),
        'facturacion_usd_val': ['mean', 'std', 'sum'],
        'materiales_distintos_val': 'mean',
        'cajas_fisicas': 'mean'
    }).reset_index()
    
    client_features.columns = [
        'cliente_id', 'primera_compra', 'ultima_compra', 'total_pedidos',
        'ratio_digital_historico', 'ticket_promedio', 'ticket_std', 
        'facturacion_total', 'materiales_promedio', 'cajas_promedio'
    ]
    
    # Features temporales
    client_features['antiguedad_dias'] = (client_features['ultima_compra'] - client_features['primera_compra']).dt.days
    client_features['frecuencia_pedidos'] = client_features['antiguedad_dias'] / client_features['total_pedidos']
    
    # Ratio digital últimos 3 meses
    cutoff_date = df['fecha_pedido_dt'].max() - pd.Timedelta(days=90)
    recent_digital = df[df['fecha_pedido_dt'] > cutoff_date].groupby('cliente_id').agg({
        'canal_pedido_cd': lambda x: (x == 'DIGITAL').mean()
    }).reset_index()
    recent_digital.columns = ['cliente_id', 'ratio_digital_3m']
    
    client_features = client_features.merge(recent_digital, on='cliente_id', how='left')
    client_features['ratio_digital_3m'] = client_features['ratio_digital_3m'].fillna(0)
    
    # Tendencia digital
    client_features['tendencia_digital'] = (
        client_features['ratio_digital_3m'] - client_features['ratio_digital_historico']
    )
    
    return client_features


# =========================
# 2️⃣ Preparar dataset de entrenamiento
# =========================

# Asegurar tipo datetime
df['fecha_pedido_dt'] = pd.to_datetime(df['fecha_pedido_dt'])

# Último pedido (target)
latest_orders = (
    df.sort_values('fecha_pedido_dt')
      .groupby('cliente_id')
      .last()
      .reset_index()
)
latest_orders['target'] = (latest_orders['canal_pedido_cd'] == 'DIGITAL').astype(int)

# Excluir último pedido del histórico (para evitar fuga temporal)
df_hist = df.merge(
    latest_orders[['cliente_id', 'fecha_pedido_dt']],
    on='cliente_id', 
    suffixes=('', '_target')
)
df_hist = df_hist[df_hist['fecha_pedido_dt'] < df_hist['fecha_pedido_dt_target']]

# Preparar features
features_df = prepare_lgb_features(df_hist)

# Combinar con el target
train_data = features_df.merge(latest_orders[['cliente_id', 'target']], on='cliente_id')

# =========================
# 3️⃣ Entrenamiento del modelo
# =========================

feature_cols = [
    c for c in train_data.columns 
    if c not in ['cliente_id', 'target', 'primera_compra', 'ultima_compra']
]

X = train_data[feature_cols]
y = train_data['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

lgb_model = lgb.LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    n_jobs=-1
)

lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric='auc',
    #verbose=False
)

# =========================
# 4️⃣ Evaluación
# =========================

y_pred_proba = lgb_model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred_proba)

print(f"✅ LightGBM AUC: {auc:.4f}")

# Importancia de variables
importances = pd.DataFrame({
    'feature': feature_cols,
    'importance': lgb_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n📊 Top variables más importantes:")
print(importances.head(10))



# COMMAND ----------

