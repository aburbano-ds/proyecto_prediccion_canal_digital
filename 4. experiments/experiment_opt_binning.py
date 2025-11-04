# Databricks notebook source
# MAGIC %md
# MAGIC #### Librerias

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

spark.conf.set("spark.databricks.io.cache.enabled", True)
spark.conf.set('spark.sql.shuffle.partitions', 'auto')

from IPython.display import display
import mlflow
#mute warnings
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

try:    import mlflow
except ImportError:
    print('mlflow package is not installed or has issues.')

mlflow.autolog(disable=True)

from optbinning import BinningProcess
from optbinning import Scorecard
from optbinning.scorecard import plot_auc_roc, plot_cap, plot_ks
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

# COMMAND ----------

int_pedidos_clientes = (spark.read.parquet("/Volumes/dbw_prod_aavanzada/db_tmp/files/pburbano/data/")
                                  .withColumn("fecha_pedido_dt", F.to_date(F.col("fecha_pedido_dt")))
                        )

# Crear ventana ordenada por cliente y fecha
w = Window.partitionBy("cliente_id").orderBy(F.asc("fecha_pedido_dt"))
w_prev_all = Window.partitionBy("cliente_id").rowsBetween(Window.unboundedPreceding, -1)

# Agregar canal del siguiente pedido
df = int_pedidos_clientes.withColumn("canal_siguiente", F.lead("canal_pedido_cd").over(w))

# Crear target
df = df.withColumn("target", F.when(F.col("canal_siguiente") == "DIGITAL", 1).otherwise(0))

# Variables de comportamiento
df = (df.withColumn("canal_previo", F.lag("canal_pedido_cd").over(w))
        .withColumn("dias_desde_pedido_anterior", F.datediff("fecha_pedido_dt", F.lag("fecha_pedido_dt").over(w)))
        .withColumn("n_pedidos_previos", F.row_number().over(w) - 1)
        .withColumn("facturacion_prom_anterior", F.avg("facturacion_usd_val").over(w_prev_all))
        .withColumn("uso_digital_prev", F.sum(F.when(F.col("canal_pedido_cd") == "DIGITAL", 1).otherwise(0)).over(w_prev_all))
        .withColumn("prop_digital_prev", F.col("uso_digital_prev") / F.when(F.col("n_pedidos_previos") > 0, F.col("n_pedidos_previos")).otherwise(1))
)

# 3. Nuevas features adicionales útiles
df = (
    df.withColumn("uso_telefono_prev", F.sum(F.when(F.col("canal_pedido_cd") == "TELEFONO", 1).otherwise(0)).over(w_prev_all))
      .withColumn("uso_vendedor_prev", F.sum(F.when(F.col("canal_pedido_cd") == "VENDEDOR", 1).otherwise(0)).over(w_prev_all))
      .withColumn("prop_telefono_prev", F.col("uso_telefono_prev") / F.col("n_pedidos_previos"))
      .withColumn("prop_vendedor_prev", F.col("uso_vendedor_prev") / F.col("n_pedidos_previos"))
      .withColumn("dias_media_prev", F.avg(F.datediff("fecha_pedido_dt", F.lag("fecha_pedido_dt").over(w))).over(w_prev_all))
      .withColumn("dias_media_std", F.stddev(F.datediff("fecha_pedido_dt", F.lag("fecha_pedido_dt").over(w))).over(w_prev_all))
      .withColumn("facturacion_total_prev", F.sum("facturacion_usd_val").over(w_prev_all))
      .withColumn("desviacion_facturacion", F.stddev("facturacion_usd_val").over(w_prev_all))
)

mdt = (df.filter(F.col("n_pedidos_previos") > 0)
         .filter(F.col("target").isNotNull())
         .select("pais_cd", 
                 "region_comercial_txt", 
                 "agencia_id", 
                 "ruta_id", 
                 "estrellas_txt",
                 "target", 
                 "canal_previo", 
                 "facturacion_usd_val", 
                 "dias_desde_pedido_anterior",
                 "n_pedidos_previos", 
                 "facturacion_prom_anterior", 
                 "uso_digital_prev",
                 "prop_digital_prev",
                 "tipo_cliente_cd",
                 "frecuencia_visitas_cd", 
                 "madurez_digital_cd",
                 "uso_telefono_prev",
                 "uso_vendedor_prev",
                 "prop_telefono_prev",
                 "prop_vendedor_prev",
                 "dias_media_prev",
                 "facturacion_total_prev",
                 "desviacion_facturacion",
                 "fecha_pedido_dt"
                )
)


#mdt = mdt.withColumn("madurez_digital_cd", F.when(F.col("madurez_digital_cd") != "ALTA", F.lit("OTRA")).otherwise(F.col("madurez_digital_cd")))

fecha_corte = "2024-04-15"
mdt = mdt.withColumn("periodo", F.when(F.col("fecha_pedido_dt") < fecha_corte, F.lit("TRAIN")).otherwise(F.lit("TEST")))

#mdt = mdt.filter("madurez_digital_cd == 'ALTA'")

mdt = mdt.fillna(0)

mdt_pd = mdt.toPandas()

# COMMAND ----------

var = ['prop_digital_prev',
 'uso_telefono_prev',
 'madurez_digital_cd',
 'estrellas_txt',
 'canal_previo']

# COMMAND ----------

data_train = mdt_pd[mdt_pd.periodo == "TRAIN"]
x_train = data_train[var]
y_train = data_train["target"].values

data_test = mdt_pd[mdt_pd.periodo == "TEST"]
x_test = data_test[var]
y_test = data_test["target"].values

# COMMAND ----------

selection_criteria = {
   "iv": {"min": 0.01, "max": 10},
   "quality_score": {"min": 0.0001},
}

binning_fit_params = {

}



binning_process = BinningProcess(var, max_n_bins=5, selection_criteria=selection_criteria, binning_fit_params=binning_fit_params)

estimator = LogisticRegression(solver="lbfgs")

scorecard = Scorecard(binning_process=binning_process, intercept_based=True,
                      estimator=estimator, scaling_method="min_max",
                      scaling_method_params={"min": 0, "max": 1000}, rounding=True)


binning_process = BinningProcess(var, max_n_bins=5, selection_criteria=selection_criteria, binning_fit_params=binning_fit_params)

estimator = LogisticRegression(solver="lbfgs")

scorecard = Scorecard(binning_process=binning_process, intercept_based=True,
                      estimator=estimator, scaling_method="min_max",
                      scaling_method_params={"min": 0, "max": 1000}, rounding=True)

# COMMAND ----------

scorecard.fit(x_train, y_train, show_digits=4)

# COMMAND ----------

score_card = scorecard.table(style="detailed")
score_card

# COMMAND ----------

y_pred_train = scorecard.predict_proba(x_train)[:, 1]
y_pred_test = scorecard.predict_proba(x_test)[:, 1]
data_train["score_train"] = scorecard.score(x_train)
data_test["score_test"] = scorecard.score(x_test)

# COMMAND ----------

plot_ks(y_train, y_pred_train)

# COMMAND ----------

plot_ks(y_test, y_pred_test)

# COMMAND ----------

plot_cap(y_train, y_pred_train)

# COMMAND ----------

plot_cap(y_test, y_pred_test) 

# COMMAND ----------

probability_train = y_pred_train
df_train = adjusted_prediction(data_train.target, probability_train, 0.52, 0.52)
model_metrics(df_train)
#plot_model(df_train)

# COMMAND ----------

probability_test = y_pred_test
df_test = adjusted_prediction(data_test.target, probability_test, 0.52, 0.52)
model_metrics(df_test)
#plot_model(df_test)