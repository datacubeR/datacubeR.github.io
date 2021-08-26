---
title: "Bitcoin Price Prediction"
subheadline: "Predecir el comportamiento del BitCoin"
teaser: "Creando LSTM en Pytorch Lightning"
# layout: page-fullwidth
usemathjax: true
category: quick
header: no
image:
    thumb: bitcoin/bitcoin.jpg
tags:
- python
- tutorial
published: true
---


![picture of me]({{ site.urlimg }}bitcoin/bitcoin.jpg){: .left .show-for-large-up .hide-for-print width="300"}
![picture of me]({{ site.urlimg }}bitcoin/bitcoin.jpg){: .center .hide-for-large-up width="250"}

Las criptomonedas son un tema apasionante. En el último tiempo se han convertido en el dolor de cabezas de los gamers (también de los modeladores de Deep Learning y de Nvidia que no sabe como producir más GPUs), ya que han provocado una escasez en el stock de GPUs, pero también han ayudado a ganar (y también a perder) dinero a mucha gente. Una de las catacterísticas más llamativas de este mercado es lo impredecibles que son y cómo comentarios de gente importante (<mark>tío Elon</mark>) pueden generar variaciones muy importantes de manera súbita. Hoy intentaremos predecir el comportamiento del precio del Bitcoin utilizando Pytorch Lightning<!--more-->, aplicando una de las que fueron las redes Neuronales más famosas antes de la era de los Transformers, las LSTM (Long Short-Term Memory).

Las LSTM son más que un tipo de red, se podría decir que es una configuración de varias redes neuronales cada una con un propósito en particular. Podríamos <q>casi</q> llamarla una arquitectura. La principal característica que tienen es que poseen memoria, una característica que las hace ideal para trabajar con datos secuenciales, ya que puede utilizar datos del pasado para poder <q>predecir el futuro</q>. Las LSTM Son la evolución de las RNN convencionales ya que introduciendo un **cell state** son capaces de solucionar el problema del vanishing gradient que éstas suelen sufrir.

Entender las redes recurrentes puede ser complicado y no es mi intención explicar en detalle la matemática de fondo de este tipo de redes. Pero para quienes les interese, pueden visitar el [Colah's Blog](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) que es el lugar donde mejor se explica esta arquitectura (y de donde robé las siguientes imágenes).

Básicamente, una LSTM tiene distintas etapas (cada una una red neuronal) que se encargan de aprender o desaprender (espero que esta palabra exista) distintas partes. La primera es la *forget gate*. Esta red neuronal se encarga de olvidar, a medida que se entrena decide qué cosas ya no vale la pena recordar en futuras iteraciones.

![picture of me]({{ site.urlimg }}bitcoin/forget_gate.png){: .center}

La *input gate* es la información que llega. El propósito de esta red neuronal es aprender qué aspectos son importantes de los nuevos datos de la secuencia

![picture of me]({{ site.urlimg }}bitcoin/input_gate.png){: .center}

Esta información nueva debe combinarse con el output del *forget gate* dando paso al *cell state*. El *cell state* será la memoria y es la información nueva menos lo que tiene que olvidarse, es decir, toda la información que debe ser pasada a la siguiente iteración.

![picture of me]({{ site.urlimg }}bitcoin/cell_state.png){: .center}

Finalmente el output será una red que decide la salida de la red considerando tanto la entrada (input gate) como la memoria (cell state). La gracia de una LSTM es que cada capa asociada (neurona LSTM) tendrá un output, pero además ese output pasará a la siguiente iteración como el *hidden state*.

![picture of me]({{ site.urlimg }}bitcoin/output.png){: .center}

Para poder resolver este problema predictivo se ha utilizado data descargada desde Binance que corresponde a la información por minuto del precio del Bitcoin, la cual se puede descargar del siguiente [sitio](https://www.cryptodatadownload.com/data/binance).

Al menos en mi caso descargué datos que van desde el 2019 hasta el 5 de Agosto de 2021 (sí, me demoré en poder generar un modelo decente para mostrar).


```python
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib

import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
from multiprocessing import cpu_count

from sklearn.preprocessing import MinMaxScaler
```
{: title="Librerías comunes"}


```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import torchmetrics
```
{: title="Ecosistema Pytorch"}

```python
%config InlineBackend.figure_format = 'retina'
sns.set(style = 'whitegrid', palette = 'muted', font_scale = 1.2)
HAPPY_COLORS_PALETTE = ['#01BEFE', '#FFDD00','#FF7D00','#FF006D','#ADFF02','#8F00FF']
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
rcParams['figure.figsize'] = 6,4
tqdm.pandas()
```
{: title="Configuración Visual"}


```python
RANDOM_SEED = 42
pl.seed_everything(RANDOM_SEED)
```
{: title="Asegurar reproducibilidad"}

    Global seed set to 42

Lo primero que haremos será entonces importar la data de `Binance`. Sólo destacar que se hace un parseo de la fecha y se ordena la data por fecha dado su caracter secuencial. Luego se reinicia el índice para evitar cualquier inconveniente posterior.

```python
df = pd.read_csv('Binance_BTCUSDT_minute.csv', parse_dates = ['date']).sort_values(by = 'date').reset_index(drop = True)
df.shape
```

    (997650, 10)

La cantidad de data extraída es bastante. Las redes recurrentes en general son redes lentas de entrenar debido precisamente a su caracter presencial. Para ayudar en el proceso de entrenamiento, evitar ruidos del pasado (aunque podría impactar en la estacionalidad) y permitir que otras personas sin GPU puedan reproducirlo, decidí entrenar sólo con la información de este año.

Debido a que la información está detallada al minuto, la estacionalidad debiera también revisarse a ese nivel. La LSTM debiera fijarse en minutos anteriores para predecir el minuto siguiente.

```python
df.query('date > "2021-01-01"', inplace = True)
print(df.shape)
df
```
{: title="Filtro considerando sólo el 2021"}
    (306437, 10)


<div class='table-overflow'>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>unix</th>
      <th>date</th>
      <th>symbol</th>
      <th>open</th>
      <th>high</th>
      <th>low</th>
      <th>close</th>
      <th>Volume BTC</th>
      <th>Volume USDT</th>
      <th>tradecount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>691213</th>
      <td>1609459260000</td>
      <td>2021-01-01 00:01:00</td>
      <td>BTC/USDT</td>
      <td>28961.67</td>
      <td>29017.50</td>
      <td>28961.01</td>
      <td>29009.91</td>
      <td>58.477501</td>
      <td>1.695803e+06</td>
      <td>1651</td>
    </tr>
    <tr>
      <th>691214</th>
      <td>1609459320000</td>
      <td>2021-01-01 00:02:00</td>
      <td>BTC/USDT</td>
      <td>29009.54</td>
      <td>29016.71</td>
      <td>28973.58</td>
      <td>28989.30</td>
      <td>42.470329</td>
      <td>1.231359e+06</td>
      <td>986</td>
    </tr>
    <tr>
      <th>691215</th>
      <td>1609459380000</td>
      <td>2021-01-01 00:03:00</td>
      <td>BTC/USDT</td>
      <td>28989.68</td>
      <td>28999.85</td>
      <td>28972.33</td>
      <td>28982.69</td>
      <td>30.360677</td>
      <td>8.800168e+05</td>
      <td>959</td>
    </tr>
    <tr>
      <th>691216</th>
      <td>1609459440000</td>
      <td>2021-01-01 00:04:00</td>
      <td>BTC/USDT</td>
      <td>28982.67</td>
      <td>28995.93</td>
      <td>28971.80</td>
      <td>28975.65</td>
      <td>24.124339</td>
      <td>6.992262e+05</td>
      <td>726</td>
    </tr>
    <tr>
      <th>691217</th>
      <td>1609459500000</td>
      <td>2021-01-01 00:05:00</td>
      <td>BTC/USDT</td>
      <td>28975.65</td>
      <td>28979.53</td>
      <td>28933.16</td>
      <td>28937.11</td>
      <td>22.396014</td>
      <td>6.483227e+05</td>
      <td>952</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>997645</th>
      <td>1628125260000</td>
      <td>2021-08-05 01:01:00</td>
      <td>BTC/USDT</td>
      <td>39592.99</td>
      <td>39600.00</td>
      <td>39559.77</td>
      <td>39570.02</td>
      <td>33.148678</td>
      <td>1.312017e+06</td>
      <td>1462</td>
    </tr>
    <tr>
      <th>997646</th>
      <td>1628125320000</td>
      <td>2021-08-05 01:02:00</td>
      <td>BTC/USDT</td>
      <td>39570.01</td>
      <td>39570.01</td>
      <td>39388.00</td>
      <td>39396.36</td>
      <td>207.697394</td>
      <td>8.198434e+06</td>
      <td>4960</td>
    </tr>
    <tr>
      <th>997647</th>
      <td>1628125380000</td>
      <td>2021-08-05 01:03:00</td>
      <td>BTC/USDT</td>
      <td>39396.36</td>
      <td>39440.69</td>
      <td>39363.55</td>
      <td>39434.95</td>
      <td>121.355510</td>
      <td>4.780575e+06</td>
      <td>3329</td>
    </tr>
    <tr>
      <th>997648</th>
      <td>1628125440000</td>
      <td>2021-08-05 01:04:00</td>
      <td>BTC/USDT</td>
      <td>39434.95</td>
      <td>39436.77</td>
      <td>39364.06</td>
      <td>39369.67</td>
      <td>46.405522</td>
      <td>1.827913e+06</td>
      <td>1604</td>
    </tr>
    <tr>
      <th>997649</th>
      <td>1628125500000</td>
      <td>2021-08-05 01:05:00</td>
      <td>BTC/USDT</td>
      <td>39369.67</td>
      <td>39394.00</td>
      <td>39338.53</td>
      <td>39361.02</td>
      <td>38.950155</td>
      <td>1.533056e+06</td>
      <td>2341</td>
    </tr>
  </tbody>
</table>
<p>306437 rows × 10 columns</p>
</div>

Como se puede apreciar hay 16 variables. Se escogeran las que que ami parecer son más relevantes y crearemos algunas que nos permitan captar información temporal:

{% include alert warning='No soy para nada experto en temas de Trading. Mi única intención con este post es mostrar las capacidades que puede tener una red Neuronal para predecir el comportamiento del Bitcoin. Si este tema le interesa desde el punto de vista financiero o de trading le recomiendo seguir a [Lautaro Parada](https://www.linkedin.com/in/lautaro-parada-opazo-a85615b2/?originalSubdomain=cl), este cabro sabe y ha desarrollado algunas herramientas en Python para facilitar el tema financiero (ver más detalles [acá](https://github.com/LautaroParada/eod-data))'%}

## Creación del Close Change

Esta es una variable que creo que puede ser significativa. Corresponde a cuánto cambió el precio del Bitcoin respecto al periodo anterior (en este caso al minuto anterior).

```python
%%time
df['prev_close'] = df.shift(1)['close']
df['close_change'] = (df.close-df.prev_close).fillna(0)
df.head()
```


<div class='table-overflow'>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>unix</th>
      <th>date</th>
      <th>symbol</th>
      <th>open</th>
      <th>high</th>
      <th>low</th>
      <th>close</th>
      <th>Volume BTC</th>
      <th>Volume USDT</th>
      <th>tradecount</th>
      <th>prev_close</th>
      <th>close_change</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>691213</th>
      <td>1609459260000</td>
      <td>2021-01-01 00:01:00</td>
      <td>BTC/USDT</td>
      <td>28961.67</td>
      <td>29017.50</td>
      <td>28961.01</td>
      <td>29009.91</td>
      <td>58.477501</td>
      <td>1.695803e+06</td>
      <td>1651</td>
      <td>NaN</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>691214</th>
      <td>1609459320000</td>
      <td>2021-01-01 00:02:00</td>
      <td>BTC/USDT</td>
      <td>29009.54</td>
      <td>29016.71</td>
      <td>28973.58</td>
      <td>28989.30</td>
      <td>42.470329</td>
      <td>1.231359e+06</td>
      <td>986</td>
      <td>29009.91</td>
      <td>-20.61</td>
    </tr>
    <tr>
      <th>691215</th>
      <td>1609459380000</td>
      <td>2021-01-01 00:03:00</td>
      <td>BTC/USDT</td>
      <td>28989.68</td>
      <td>28999.85</td>
      <td>28972.33</td>
      <td>28982.69</td>
      <td>30.360677</td>
      <td>8.800168e+05</td>
      <td>959</td>
      <td>28989.30</td>
      <td>-6.61</td>
    </tr>
    <tr>
      <th>691216</th>
      <td>1609459440000</td>
      <td>2021-01-01 00:04:00</td>
      <td>BTC/USDT</td>
      <td>28982.67</td>
      <td>28995.93</td>
      <td>28971.80</td>
      <td>28975.65</td>
      <td>24.124339</td>
      <td>6.992262e+05</td>
      <td>726</td>
      <td>28982.69</td>
      <td>-7.04</td>
    </tr>
    <tr>
      <th>691217</th>
      <td>1609459500000</td>
      <td>2021-01-01 00:05:00</td>
      <td>BTC/USDT</td>
      <td>28975.65</td>
      <td>28979.53</td>
      <td>28933.16</td>
      <td>28937.11</td>
      <td>22.396014</td>
      <td>6.483227e+05</td>
      <td>952</td>
      <td>28975.65</td>
      <td>-38.54</td>
    </tr>
  </tbody>
</table>
</div>

# Creación de Variables temporales.

Estas variables son las que nos permitirán poder determinar posibles estacionalidades. Si bien es posible determinar muchísimas subdivisiones de tiempo, decidí que el día de la semana, el día del mes, la semana del año y el mes serían suficiente.

```python
df = df.assign(
    day_of_week = lambda x: x.date.dt.dayofweek,
    day_of_month = lambda x: x.date.dt.day,
    week_of_year = lambda x: x.date.dt.isocalendar().week,
    month = lambda x: x.date.dt.month
        )
df.head()
```

{% include alert tip='Es muy probable que aquí esté cometiendo un error, ya que estoy entrenando con menos de un año y puede que esté desaprovechando algunas de estas variables temporales. Inicialmente entrené con toda la data pero se demoraba demasiado (aunque sólo probé en el notebook). Es importante destacar que este proceso de entrenamiento es largo a pesar de que la data es muy poquita.'%}


<div class='table-overflow'>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>unix</th>
      <th>date</th>
      <th>symbol</th>
      <th>open</th>
      <th>high</th>
      <th>low</th>
      <th>close</th>
      <th>Volume BTC</th>
      <th>Volume USDT</th>
      <th>tradecount</th>
      <th>prev_close</th>
      <th>close_change</th>
      <th>day_of_week</th>
      <th>day_of_month</th>
      <th>week_of_year</th>
      <th>month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>691213</th>
      <td>1609459260000</td>
      <td>2021-01-01 00:01:00</td>
      <td>BTC/USDT</td>
      <td>28961.67</td>
      <td>29017.50</td>
      <td>28961.01</td>
      <td>29009.91</td>
      <td>58.477501</td>
      <td>1.695803e+06</td>
      <td>1651</td>
      <td>NaN</td>
      <td>0.00</td>
      <td>4</td>
      <td>1</td>
      <td>53</td>
      <td>1</td>
    </tr>
    <tr>
      <th>691214</th>
      <td>1609459320000</td>
      <td>2021-01-01 00:02:00</td>
      <td>BTC/USDT</td>
      <td>29009.54</td>
      <td>29016.71</td>
      <td>28973.58</td>
      <td>28989.30</td>
      <td>42.470329</td>
      <td>1.231359e+06</td>
      <td>986</td>
      <td>29009.91</td>
      <td>-20.61</td>
      <td>4</td>
      <td>1</td>
      <td>53</td>
      <td>1</td>
    </tr>
    <tr>
      <th>691215</th>
      <td>1609459380000</td>
      <td>2021-01-01 00:03:00</td>
      <td>BTC/USDT</td>
      <td>28989.68</td>
      <td>28999.85</td>
      <td>28972.33</td>
      <td>28982.69</td>
      <td>30.360677</td>
      <td>8.800168e+05</td>
      <td>959</td>
      <td>28989.30</td>
      <td>-6.61</td>
      <td>4</td>
      <td>1</td>
      <td>53</td>
      <td>1</td>
    </tr>
    <tr>
      <th>691216</th>
      <td>1609459440000</td>
      <td>2021-01-01 00:04:00</td>
      <td>BTC/USDT</td>
      <td>28982.67</td>
      <td>28995.93</td>
      <td>28971.80</td>
      <td>28975.65</td>
      <td>24.124339</td>
      <td>6.992262e+05</td>
      <td>726</td>
      <td>28982.69</td>
      <td>-7.04</td>
      <td>4</td>
      <td>1</td>
      <td>53</td>
      <td>1</td>
    </tr>
    <tr>
      <th>691217</th>
      <td>1609459500000</td>
      <td>2021-01-01 00:05:00</td>
      <td>BTC/USDT</td>
      <td>28975.65</td>
      <td>28979.53</td>
      <td>28933.16</td>
      <td>28937.11</td>
      <td>22.396014</td>
      <td>6.483227e+05</td>
      <td>952</td>
      <td>28975.65</td>
      <td>-38.54</td>
      <td>4</td>
      <td>1</td>
      <td>53</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>

Finalmente me quedé con las siguientes variables. La única lógica que utilicé para escoger estas variables es mi sentido común y mi vago conocimiento de finanzas. Si alguien considera que algo más debió ser considerado (o que de frentón me qequivoqué) por favor no dude en notificarlo en mi [github](https://github.com/datacubeR/datacubeR.github.io/issues).

```python
features_df = df[['day_of_week','day_of_month','week_of_year','month', 'open','high','low','close_change','close']].copy()
features_df
```

<div class='table-overflow'>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>day_of_week</th>
      <th>day_of_month</th>
      <th>week_of_year</th>
      <th>month</th>
      <th>open</th>
      <th>high</th>
      <th>low</th>
      <th>close_change</th>
      <th>close</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>691213</th>
      <td>4</td>
      <td>1</td>
      <td>53</td>
      <td>1</td>
      <td>28961.67</td>
      <td>29017.50</td>
      <td>28961.01</td>
      <td>0.00</td>
      <td>29009.91</td>
    </tr>
    <tr>
      <th>691214</th>
      <td>4</td>
      <td>1</td>
      <td>53</td>
      <td>1</td>
      <td>29009.54</td>
      <td>29016.71</td>
      <td>28973.58</td>
      <td>-20.61</td>
      <td>28989.30</td>
    </tr>
    <tr>
      <th>691215</th>
      <td>4</td>
      <td>1</td>
      <td>53</td>
      <td>1</td>
      <td>28989.68</td>
      <td>28999.85</td>
      <td>28972.33</td>
      <td>-6.61</td>
      <td>28982.69</td>
    </tr>
    <tr>
      <th>691216</th>
      <td>4</td>
      <td>1</td>
      <td>53</td>
      <td>1</td>
      <td>28982.67</td>
      <td>28995.93</td>
      <td>28971.80</td>
      <td>-7.04</td>
      <td>28975.65</td>
    </tr>
    <tr>
      <th>691217</th>
      <td>4</td>
      <td>1</td>
      <td>53</td>
      <td>1</td>
      <td>28975.65</td>
      <td>28979.53</td>
      <td>28933.16</td>
      <td>-38.54</td>
      <td>28937.11</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>997645</th>
      <td>3</td>
      <td>5</td>
      <td>31</td>
      <td>8</td>
      <td>39592.99</td>
      <td>39600.00</td>
      <td>39559.77</td>
      <td>-22.97</td>
      <td>39570.02</td>
    </tr>
    <tr>
      <th>997646</th>
      <td>3</td>
      <td>5</td>
      <td>31</td>
      <td>8</td>
      <td>39570.01</td>
      <td>39570.01</td>
      <td>39388.00</td>
      <td>-173.66</td>
      <td>39396.36</td>
    </tr>
    <tr>
      <th>997647</th>
      <td>3</td>
      <td>5</td>
      <td>31</td>
      <td>8</td>
      <td>39396.36</td>
      <td>39440.69</td>
      <td>39363.55</td>
      <td>38.59</td>
      <td>39434.95</td>
    </tr>
    <tr>
      <th>997648</th>
      <td>3</td>
      <td>5</td>
      <td>31</td>
      <td>8</td>
      <td>39434.95</td>
      <td>39436.77</td>
      <td>39364.06</td>
      <td>-65.28</td>
      <td>39369.67</td>
    </tr>
    <tr>
      <th>997649</th>
      <td>3</td>
      <td>5</td>
      <td>31</td>
      <td>8</td>
      <td>39369.67</td>
      <td>39394.00</td>
      <td>39338.53</td>
      <td>-8.65</td>
      <td>39361.02</td>
    </tr>
  </tbody>
</table>
<p>306437 rows × 9 columns</p>
</div>

# Data Split

Como se trata de data secuencial el split de datos de validación no se puede (ni se debe) realizar de manera aleatoria. Si no más bien, toda la data, que llamaremos test, tiene que ser data posterior al proceso de entrenamiento siguiendo la secuencia lógica temporal.

Es por eso que decidí entrenar con el 90% de la data disponible y validar con el 10% restante:


```python
train_size = int(len(features_df)* 0.9)
print('Number of Train Sequences: ', train_size)
train_df, test_df = features_df[:train_size], features_df[train_size + 1:] # + 1 is not necessary
train_df.shape, test_df.shape
```
{: title="Split de la Data"}

    Number of Train Sequences:  275793

    ((275793, 9), (30643, 9))


# Estandarización 
Finalmente para lograr una mejor convergencia de los datos generaremos una estadarización. Llevando todos los datos al rango -1, 1. 


```python
scaler = MinMaxScaler(feature_range=(-1,1))
train_df = pd.DataFrame(
    scaler.fit_transform(train_df),
    index = train_df.index,
    columns = train_df.columns
    )

test_df = pd.DataFrame(
    scaler.transform(test_df),
    index = test_df.index,
    columns = test_df.columns
    )
```

## Creación de Secuencias

El formato inicial de la data no viene listo para ser usado en redes recurrentes. Como dije anteriormente, la aplicación de este tipo de Redes Neuronales se justifica siempre y cuando tengamos data secuencial. Esta data se construirá de la siguiente manera:

```python
def create_sequences(input_data: pd.DataFrame, target_column, sequence_length):
    sequences = []
    data_size = len(input_data)
    
    for i in tqdm(range(data_size - sequence_length)):
        sequence = input_data[i: i+sequence_length] # No considero el último valor
        label_position = i + sequence_length # Este es el último valor, usado de Label
        label = input_data.iloc[label_position][target_column]
        
        sequences.append((sequence, label))
    
    return sequences
```
{: title="Función para crear secuencias"}

Esta función puede ser un poco dificil de explicar por lo que lo haré mediante un ejemplo. Supongamos el siguiente DataFrame:

```python
sample_data = pd.DataFrame(dict(
    feature_1 = [1,2,3,4,5],
    label = [6,7,8,9,10]
))
sample_data
```

<div class='table-overflow'>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feature_1</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>8</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>9</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>10</td>
    </tr>
  </tbody>
</table>
</div>


Ejecutaremos la función de manera de crear secuencias de largo 3:
```python
sample_sequences = create_sequences(sample_data, 'label', sequence_length=3)
```

Como se puede ver, esta función dejará 3 filas (largo de secuencia 3) para cada una de las variables contenidas en el DataFrame. Esos 3 registros tendrán como objetivo predecir el valor siguiente como se muestra en los siguientes ejemplos:

```python
print()
print(f'label: {sample_sequences[0][1]}')
sample_sequences[0][0]
```
{: title="Primera secuencia"}
    
    label: 9


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feature_1</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>

Debido a la definición de nuestra función sólo se pueden crear secuencias hasta que el último registro del dataframe exista. Por ejemplo la secuencia para los registros 2, 3, 4 no existe porque no es posible asociarle un label.


```python
print()
print(f'label: {sample_sequences[1][1]}')
sample_sequences[1][0]
```
{: title="Segunda Secuencia"}

    
    label: 10


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feature_1</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>8</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
</div>

{% include alert info='El modelo que queremos plantear acá es un modelo de Serie de Tiempo Multivariada. Una de las ventajas que poseen las redes neuronales es que es posible utilizar más de una variable (o lo que se llama también usar variables exógenas) para predecir el comportamiento de mi objetivo en el tiempo.'%}

Por lo tanto para poder crear este modelo predictivo usaremos secuencias de largo 120, es decir, dos horas. 

```python
SEQUENCE_LENGTH = 120
train_sequences = create_sequences(train_df, 'close', SEQUENCE_LENGTH)
test_sequences = create_sequences(test_df, 'close', SEQUENCE_LENGTH)
```

```python
print('Secuencias de Entrenamiento', len(train_sequences))
print('Diferencia entre Registros y Secuencias creadas: ', len(train_df) - len(train_sequences))

```

    Secuencias de Entrenamiento 275673
    Diferencia entre Registros y Secuencias creadas:  120

{% include alert tip='La diferencia entre el número de registros de un dataset y el número de secuencias creadas debe ser igual al largo de la secuencia. Estos valores corresponderán a la secuencia que no se puede crear debido a no poder crear el target.'%}


# Modelo 

El modelo será construido en Pytorch Lightning debido a la facilidad de uso. Ya hemos vistos en post anteriores que cualquier modelo puede ser transformado en Lightning. La ventaja de Lightning es que el código quedará mucho más organizado y es posible configurar opciones avanzadas de manera mucho más sencilla. 

Si quieres aprender la equivalencia entre un modelo en Pytorch Nativo y uno en Pytorch Lightning puedes ver este [post]({{ site.baseurl }}/lightning).

## Pytorch Dataset y Pytorch Data Module

A continuación crearemos el Pytorch Dataset, que se encargará de generar las secuencias, las cuales serán el DataFrame como se explicó anteriormente y el label que será el valor a predecir.
Adicionalmente se crearán los Dataloaders en el Lightning Data Module. Los 3 dataloaders creados serán el de entrenamiento, el de validación y el de predicción. El dataloader de predicción se utilizará con la data de testeo que es con la que verificaremos si nuestro modelo responde como esperamos.


```python
class BTCDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences
        
    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, idx):
        sequence, label = self.sequences[idx]
        
        return dict(
            sequence = torch.tensor(sequence.to_numpy(), dtype = torch.float32),
            label = torch.tensor(label, dtype = torch.float32)
        )
    
class BTCPriceDataModule(pl.LightningDataModule):
    def __init__(self, train_sequences, test_sequences, batch_size = 8):
        super().__init__()
        self.train_sequences = train_sequences
        self.test_sequences = test_sequences
        self.batch_size = batch_size
        
    def setup(self, stage = None):
        self.train_dataset = BTCDataset(self.train_sequences)
        self.test_dataset = BTCDataset(self.test_sequences)
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            pin_memory=True, 
            num_workers=cpu_count(),
            shuffle = False
        )
    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            pin_memory=True, 
            num_workers=cpu_count(),
            shuffle = False
        )
    def predict_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            pin_memory=True, 
            num_workers=cpu_count(),
            shuffle = False
        )
```
Es importante destacar que todos los dataloader están paralelizados utilizando todos los cores disponibles y haciendo un `pin_memory` para aprovechar mejor la GPU. El modelo fue entrenado en [Jarvis]({{ site.baseurl}}/lightning) por lo que utilizó una `RTX 3090` como GPU, la cual redujo el tiempo de entrenamiento casi 4 veces (de 8 minutos a menos de 2) y se paralelizó en 16 hilos.


# Pytorch Model y Lightning Module

El modelo utilizado para este problema es una red neuronal LSTM que tendrá 9 neuronas (una para cada variable a utilizar). Contará con 128 celdas LSTM y 2 capas. Además a modo de regularización usarmos un dropout de un 20%.

```python
class PricePredictionModel(nn.Module):
    def __init__(self, n_features, n_hidden = 128, n_layers = 2):
        super().__init__()
        
        self.n_hidden = n_hidden
        
        self.lstm = nn.LSTM(
            input_size = n_features, 
            hidden_size = n_hidden,
            batch_first = True,
            num_layers = n_layers,
            dropout = 0.2
            
        )
        
        self.regressor = nn.Linear(n_hidden, 1)
        
    def forward(self, x):
        #self.lstm.flatten_parameters() # for distributed training
        
        _, (hidden, _) = self.lstm(x)
        #print("Shape of LSTM output: ", hidden.shape)
        out = hidden[-1]
        #print('Shape of out: ', out.shape)
        x = self.regressor(out)
        #print(x.shape)
        return x
    
class BTCPricePredictor(pl.LightningModule):
    def __init__(self, n_features: int):
        super().__init__()
        self.model = PricePredictionModel(n_features)
        self.criterion = nn.MSELoss()
        
    def forward(self, x, labels = None):
        output = self.model(x)
        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels.unsqueeze(dim = 1))
            return loss, output
        return output
    
    def training_step(self, batch, batch_idx):
        sequences = batch['sequence']
        labels = batch['label']
        
        loss, output = self(sequences, labels) 
        self.log('train_loss', loss, prog_bar = True, logger = True)
        return loss
    def validation_step(self, batch, batch_idx):
        sequences = batch['sequence']
        labels = batch['label']
        
        loss, output = self(sequences, labels) 
        self.log('val_loss', loss, prog_bar = True, logger = True)

    def predict_step(self, batch, batch_idx):
        sequences = batch['sequence']
        labels = batch['label']
        
        loss, output = self(sequences, labels) 
        return labels, output
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr = 1e-4)
```

Para entrenar definimos `training_step`, `validation_step` y `predict_step` para ser consistente con los dataloaders creados anteriormente. El modelo está siendo optimizado mediante Adam, con un learning rate de <var>1e-4</var> usando el `MSELoss` como función de costo.

Cabe destacar que para efectos de la gráfica que queremos construir al final del artículo, el `predict_step()` devolverá tanto la etiqueta real como la etiqueta predicha (esta es una de las flexibilidades que entrega Lightning para evitar duplicación de loops).

# Entrenamiento del Modelo

Finalmente para el entrenamiento del modelo se utilizará:

* 8 Epochs con EarlyStopping, eso quiere decir que si en 2 epochs no hay mejora en el score de validación el entrenamiento se termina.
* Usaremos un Batch Size de 64 secuencias que se irán pasando por la red a la vez.
* Se utilizará un Model Checkpoint, almacenando el mejor modelo de las epochs ejecutadas.
* Se forzará a que el modelo entrene al menos por 3 epochs antes de poder activar el Early Stopping.

```python
N_EPOCHS = 8
BATCH_SIZE = 64
```

```python
model = BTCPricePredictor(n_features = 9)
dm = BTCPriceDataModule(train_sequences, test_sequences, batch_size=BATCH_SIZE)
#model(item['sequence']).shape

mc = ModelCheckpoint(
    dirpath='checkpoints',
    filename = 'best-checkpoint',
    save_top_k=1,
    verbose = True,
    monitor='val_loss',
    mode = 'min'
)

early_stop = EarlyStopping(monitor = 'val_loss', patience = 2)

trainer = pl.Trainer(
    deterministic = True
    min_epochs = 3, 
    callbacks = [mc, early_stop],
    max_epochs = N_EPOCHS,
    gpus = 1,
    progress_bar_refresh_rate = 30    
)
```

    GPU available: True, used: True
    TPU available: False, using: 0 TPU cores
    IPU available: False, using: 0 IPUs

```python
trainer.fit(model, dm)
```

    LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
    
      | Name      | Type                 | Params
    ---------------------------------------------------
    0 | model     | PricePredictionModel | 203 K 
    1 | criterion | MSELoss              | 0     
    ---------------------------------------------------
    203 K     Trainable params
    0         Non-trainable params
    203 K     Total params
    0.814     Total estimated model params size (MB)

    Global seed set to 42

    Epoch 0, global step 4307: val_loss reached 0.01444 (best 0.01444), saving model to "/home/alfonso/Documents/binance/checkpoints/best-checkpoint-v1.ckpt" as top 1

    Epoch 1, global step 8615: val_loss was not in top 1

    Epoch 2, global step 12923: val_loss was not in top 1

El modelo se entrenó por 3 epochs, en las cuales tuvo dos casos sin mejoría por lo que se detuvo.

```python
trained_model = BTCPricePredictor.load_from_checkpoint(
    'checkpoints/best-checkpoint-v1.ckpt',
    n_features = 9
)
```
{: title="Carga del Mejor Checkpoint"}


```python
trained_model
```

    BTCPricePredictor(
      (model): PricePredictionModel(
        (lstm): LSTM(9, 128, num_layers=2, batch_first=True, dropout=0.2)
        (regressor): Linear(in_features=128, out_features=1, bias=True)
      )
      (criterion): MSELoss()
    )


```python
trained_model.freeze()
```
{: title="Se congelan las capas para evitar que se modifiquen gradientes."}


```python
trainer.validate(model = model)
```
{: title="Inferencia de Validación"}

    --------------------------------------------------------------------------------
    DATALOADER:0 VALIDATE RESULTS
    {'val_loss': 0.014435895718634129}
    --------------------------------------------------------------------------------

## Verificación en datos de validación

```python
preds = trainer.predict(model = trained_model, dataloaders = dm.val_dataloader())
```

```python
labels = torch.tensor(preds)[:,0]
predictions = torch.tensor(preds)[:,1]
```

Es importante destacar acá que devolvimos etiquetas y labels. Transformando el output en tensor es posible obtener dos tensores independientes de cada set de etiquetas.


## Creación de un Descaler 

Es importante recordar que para el proceso de entrenamiento toda la data fue escalada. Por lo tanto para ver efectivamente nuestro resultado se necesita traerla a la escala real. Para ello se hace el siguiente truco para poder aprovechar el `inverse_transform` de `Scikit-Learn`.

```python
descaler = MinMaxScaler()
descaler.min_, descaler.scale_ = scaler.min_[-1], scaler.scale_[-1]
```
{% include alert todo='`scaler.min_` y `scaler.scale_` son los parámetros aprendidos por MinMaxScaler. Basicamente acá estamos simulando un entrenamiento artificial utilizando sólo los parámetros de nuestro label.'%}

```python
def descale(values, descaler = descaler):
    values_2d = np.array(values)[:, np.newaxis]
    return descaler.inverse_transform(values_2d).flatten()
```

```python
predictions_descaled = descale(predictions)
labels_descaled = descale(labels)
```
{: title="Se traen los valores a escala real."}


```python
test_data = df[train_size + 1:]
len(test_data), len(test_df)
```

    (30643, 30643)

Luego verificamos que los datos de test sean consistentes con lo que generamos, ambos deben entregarnos el mismo número de registros.

Finalmente, tenemos que notar que no es posible generar predicciones para las secuencias menores al largo de Secuencia ya que no cuentan con secuencias necesarias para predecir. Por lo tanto, generamos el siguiente set de secuencias para luego extraer la fecha:

```python
test_sequences_data = test_data.iloc[SEQUENCE_LENGTH:]
len(test_sequences_data), len(test_sequences)
```
    (30523, 30523)

```python
test_sequences_data.head()
```

<div class='table-overflow'>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>unix</th>
      <th>date</th>
      <th>symbol</th>
      <th>open</th>
      <th>high</th>
      <th>low</th>
      <th>close</th>
      <th>Volume BTC</th>
      <th>Volume USDT</th>
      <th>tradecount</th>
      <th>prev_close</th>
      <th>close_change</th>
      <th>day_of_week</th>
      <th>day_of_month</th>
      <th>week_of_year</th>
      <th>month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>967127</th>
      <td>1626294180000</td>
      <td>2021-07-14 20:23:00</td>
      <td>BTC/USDT</td>
      <td>32794.15</td>
      <td>32800.00</td>
      <td>32772.74</td>
      <td>32779.19</td>
      <td>6.164449</td>
      <td>202108.117635</td>
      <td>278</td>
      <td>32794.15</td>
      <td>-14.96</td>
      <td>2</td>
      <td>14</td>
      <td>28</td>
      <td>7</td>
    </tr>
    <tr>
      <th>967128</th>
      <td>1626294240000</td>
      <td>2021-07-14 20:24:00</td>
      <td>BTC/USDT</td>
      <td>32780.65</td>
      <td>32808.76</td>
      <td>32780.64</td>
      <td>32804.94</td>
      <td>24.302589</td>
      <td>797120.586022</td>
      <td>388</td>
      <td>32779.19</td>
      <td>25.75</td>
      <td>2</td>
      <td>14</td>
      <td>28</td>
      <td>7</td>
    </tr>
    <tr>
      <th>967129</th>
      <td>1626294300000</td>
      <td>2021-07-14 20:25:00</td>
      <td>BTC/USDT</td>
      <td>32804.93</td>
      <td>32823.71</td>
      <td>32804.93</td>
      <td>32817.51</td>
      <td>10.519932</td>
      <td>345226.325132</td>
      <td>359</td>
      <td>32804.94</td>
      <td>12.57</td>
      <td>2</td>
      <td>14</td>
      <td>28</td>
      <td>7</td>
    </tr>
    <tr>
      <th>967130</th>
      <td>1626294360000</td>
      <td>2021-07-14 20:26:00</td>
      <td>BTC/USDT</td>
      <td>32814.92</td>
      <td>32823.42</td>
      <td>32812.00</td>
      <td>32820.00</td>
      <td>5.496652</td>
      <td>180388.987189</td>
      <td>284</td>
      <td>32817.51</td>
      <td>2.49</td>
      <td>2</td>
      <td>14</td>
      <td>28</td>
      <td>7</td>
    </tr>
    <tr>
      <th>967131</th>
      <td>1626294420000</td>
      <td>2021-07-14 20:27:00</td>
      <td>BTC/USDT</td>
      <td>32819.99</td>
      <td>32843.55</td>
      <td>32819.96</td>
      <td>32821.31</td>
      <td>11.457531</td>
      <td>376199.073946</td>
      <td>351</td>
      <td>32820.00</td>
      <td>1.31</td>
      <td>2</td>
      <td>14</td>
      <td>28</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>

Finalmente graficamos los valores reales del periodo de Test y los obtenidos por nuestra predicción:

```python
dates = matplotlib.dates.date2num(test_sequences_data.date.tolist())
plt.figure(figsize = (10,6))
plt.plot_date(dates, predictions_descaled, '-', label = 'predicted')
plt.plot_date(dates, labels_descaled, '-', label = 'real')
plt.xticks(rotation = 45)
plt.title('Predicción BitCoin 2021')
plt.legend();
```

![picture of me]({{ site.urlimg }}bitcoin/output_47_0.png){: .center }

## Disclaimer

No soy experto en temas de Finanzas ni Criptomonedas por lo que si se quiere aplicar modelos como estos para sus inversiones asegúrese de asesorarse. No creo que sea un modelo malo, pero si tengo las siguientes consideraciones:

* Tanto el entrenamiento como la validación está medido en datos del pasado, y funciona bien. Pero este modelo no puede predecir que Elon Musk de algún comunicado y el precio se del BitCoin se caiga de un día para otro.
* Si bien logra predecir de muy buena manera las tendencias, no predice la magnitud correcta. 
* Probablemente entrenando por más epochs y/o utilizando más data puede ayudar.
* Quizás una buena idea sea eliminar el Early Stopping y dejar que el modelo entrene las 8 epochs completas.
* Un mejor afinamiento de hiperparámetros puede ser de gran utilidad.

Espero que este tutorial haya sido de ayuda y se pueda entender el gran potencial que se tiene al utilizar redes neuronales.

## UPDATE TL;DR

Luego de subir este post recibí algunos comentarios acerca de la metodología utilizada. Quiero agradecer particularmente a [Mario Leni](https://www.linkedin.com/in/mario-jos%C3%A9-leni-rodr%C3%ADguez-baa616150/), quien me compartió el siguiente video en el cual se critican algunas de las prácticas utilizadas para la predicción de Stocks y me gustaría hacer algunos comentarios al respecto:

<div class='embed-youtube'>
{% include youtubePlayer.html id="Vfx1L2jh2Ng" %}
</div>

<br>

* Primero, el video está subido por un tipo que se denomina LazyProgrammer. He tenido la oportunidad algunos cursos de él en Udemy y la verdad es que explica super bien. En su sitio web menciona que tiene 2 Masters:
  * Uno en Computer engineering con Especialización en Machine Learning y Reconocimiento de Patrones (largo el nombre)
  * Y uno en estadísticas.
Que el tipo tiene credenciales tiene...

Ahora, es muy extraño como parte su video porque no me gusta mucho su tono. Tiene un tono agradable de voz pero igual usa palabras que encuentro pesadas. Es raro porque siento que su video es medio *pataleta* (como un berrinche) porque la gente está equivocada y reclama varias cosas sin una estructura lógica (pero igual hace propaganda a sus cursos):
  * Que nadie escribe su código.
  * Que uno copia y pega arrastrando errores.
  * Que está mal utilizar el `MinMaxScaler` (algo que yo hice en mi implementación y se supone es su reclamo principal).
  * Que no se debe usar el precio, sino que los returns.

### Que nadie escribe su código

Eso es cierto, una de las ventajas del código es poder reutilizarlo, y poder reproducir algo que alguien más hizo sin que tenga que demostrar cómo lo hizo. En mi caso yo sí tomé el código de varios otros posteos, pero no estaba en Pytorch Lightning. De hecho, yo lo adapté y sufrí harto para hacerlo funcionar, y entrenar el modelo costó, por eso la diferencia entre la data que saqué hasta que lo publiqué.

### Que uno copia y pega arrastrando errores

Es posible, en mi caso, siempre me cuestiono lo que hago con mi código para entenderlo. No suelo comprar lo que todos dicen. De hecho, especialmente en el mundo Pytorch hay mucho código boilerplate que uno solamente copia y pega sin saber para qué sirve. Yo no puedo con eso, me quita el sueño colocar código que no entiendo y siempre trato de estar super conciente de por qué se aplica cada parte. Una crítica que él reclamaba en particular es el hecho de usar secuencia de largo 1. Es muy probable que sea un error, y que mucha gente no se lo cuestione (básicamente no es una secuencia). En mi caso usé secuencias de 120. 

Es interesante que en un punto dice que no se puede predecir stocks como secuencias, porque son aleatorios (~2:40). Creo que esa es precisamente la idea de las redes recurrentes: ver si es que hay algún patrón difícil de encontrar al ojo humano y aprender de él. Pero aprender no significa repetir, el punto que él menciona me da la impresión que se refiere a redes que hacen overfitting, toman el patrón y lo repiten por que no saben generalizar. Si fuera porque nada realmente es una secuencia porque es aleatorio, no podríamos predecir nada.

### Que está mal utilizar el `MinMaxScaler`

Es raro, reclama todo el rato que está mal hacerlo pero al final dice que no está *tan* mal (~10:00). Su punto es que a diferencia de las ondas, el stock price no está acotado por lo tanto no puedo asumir el máximo como 1 y el mínimo como -1. Yo personalmente creo que esto está mal. El `MinMaxScaler` sólo mueve la escala, en ningún momento obliga a la red neuronal a predecir en el rango -1 a 1. De hecho, la implementación tiene como capa de salida una red densamente conectada sin función de activación (esto por ser un problema de regresión). Por lo tanto, al no tener una función sigmoide no está forzada a predecir en este rango. 

Entiendo que las redes neuronales no se caractarizan por su capacidad de extrapolar, y aquí puede ser que implícitamente estamos colocando un techo a la red. Pero de nuevo el objetivo de este tipo de post es mostrar cómo se entrena un modelo de serie de tiempos, no volverse rico invirtiendo en BitCoin.

### Que no se debe usar el precio, sino que los returns

Este punto no lo entiendo. Si me interesa investigar el precio, tratar de entender hasta cuanto va a llegar el precio, ¿por qué debería usar el return? Su razón es que no tiene tendencia. Bueno si la tendencia es el problema de la serie de tiempo de Precios, se puede quitar la tendencia aplicando diferenciación. Es muy sencillo de aplicar y me permite no tener que cambiar la variable que quiero monitorear. Probablemente si quiero ganar plata el return es una mejor opción, pero aún así no me calza la crítica.

Yo creo que está bien criticar cuando uno ve un proceso incorrecto. Pero en particular en el uso de redes neuronales, donde ni los Kaggle GrandMasters están 100% seguros de lo que hacen, no me parece la manera. El inglés no es mi idioma nativo, pero estoy casi seguro que <q>idiotic</q> (~3:00) no es una palabra buena onda. Y la conclusión que saco es que hace una crítica sólo para vender sus cursos.

Creo que una de las características que tienen que primar en este campo es la humildad. He visto código horriblemente malo de Phds, Kaggle Grandmasters, incluso he visto errores de concepto por parte de LazyProgrammer (tomé uno de sus cursos y aplicaba un Softmax a la salida de un problema multiclase utilizando como Loss Function CrossEntropy Loss, un concepto incorrecto y que se desaconseja/desaprueba en los foros de Pytorch), pero este es un campo en desarrollo, nadie sabe realmente cómo funcionan las Redes Neuronales. Y usando su mismo argumento, si él sí sabe como hacer Stock Prediction, por qué no es millonario invirtiendo en BitCoin (o quizás ya lo es 🤔).

Sorry por lo latero, 

Nos vemos la próxima,

[**Alfonso**]({{ site.baseurl }}/contact/)

*[Binance]: Plataforma de Intercambio de Criptomonedas.