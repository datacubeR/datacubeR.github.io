---
title: "Intro a Optuna y Wandb"
subheadline: "Herramientas complementarias para crear un Modelo"
teaser: "Optuna + Weights & Biases"
# layout: page-fullwidth
usemathjax: true
category: quick
header: no
image:
    thumb: wandb-optuna/wb.jpg
tags:
- python
- ML
- dl
- tutorial
published: false
---

# Optimización Bayesiana

![picture of me]({{ site.urlimg }}wandb-optuna/wb.jpg){: .left .show-for-large-up .hide-for-print width="500"}
![picture of me]({{ site.urlimg }}wandb-optuna/wb.jpg){: .center .hide-for-large-up width="250"}
Este va a ser un tutorial muy rápido para utilizar Optuna, una librería de optimización Bayesiana y Weights & Biases.<!--more--> El cuál espero pueda llegar a ser mi plataforma para almacenar logs de entrenamientos de modelos.

La verdad he probado otras plataformas como **MlFlow** (puedes ver un tutorial [acá]({{ site.baseurl }}/mlflow/)), el cuál fue extremadamente difícil de aprender y sumamente poco intuitivo. Si bien no creo que sea una mala herramienta la verdad no me terminó de convencer.
También vi Neptune, pero la verdad es que es una librería relativamente nueva con poquitas estrellas en github.

Por otro lado **Weights and Biases** se está volviendo la herramienta más utilizada en investigación y la verdad es que es bastante más sencilla que el resto, pero tiene una ligera curva de aprendizaje. En mi opinión, la plataforma es intuitiva, pero siento que debiera de haber un mayor esfuerzo en la documentación.

Para probar esto rápidamente veremos si es posible resolver un proceso de optimización. 

Supongamos el siguiente ejemplo:

$$y = x^2 + 1$$

Si queremos encontrar el mínimo de esta función es muy sencillo, basta con derivar e igualar a cero:
    
$$y' = 2x = 0 \rightarrow x = 0$$
$$ y = 2 \cdot 0 + 1 = 1 $$

Luego el mínimo de esta parabola se encuentra en la coordenada `(0,1)`.

## Usando Optuna

**Optuna** es una librería que está pensada para procesos de optimización en general, pero es normalmente utilizada para resolver problemas de Búsqueda de Hiperparámetros. Es decir, se buscan los hiperparámetros que permitan encontrar la métrica óptima, la cual puede ser la máxima (Accuracy o R²) o la mínima (Logloss o MSE).

Por lo tanto uno debe de entregar un rango para probar los hiperparámetros y el número de muestras a sacar de ahí. Dado que el proceso es de optimización Bayesiana, utilizará los resultados anteriores para acercarse de mejor manera al óptimo real sin la necesidad de hacer un GridSearch (revisar todas y cada una de las combinaciones posibles).

Dado que este es un proceso aleatorio, es importante destacar que podria ocurrir que el proceso no converja, por lo que es necesario dar un número de ensayos apropiado para que el proceso funcione como se espera.


```python
import wandb
import optuna
```

Lo primero es crearse una cuenta en `wandb.ai/login`. Mi recomendación es utilizar tu cuenta de github ya que permitirá linkear de mejor manera el código de tus repos con la plataforma.


```python
wandb.login()
```

    [34m[1mwandb[0m: Currently logged in as: [33mdatacuber[0m (use `wandb login --relogin` to force relogin)
    True



`wandb.login()` permitirá loguearse a la plataforma. La primera vez te solicitará el ingreso de tu contraseña pero de ahí en adelante quedará almacenado en tu sistema. Me gustó mucho el sistema ya que no tuve que configurar nada. Otras plataformas como neptune te piden almacenar tú el TOKEN de autenticación dentro de una variable de entorno y en OS como Windows puede ser más complicado.

### Optimizando en Optuna

Vamos a definir un rango en el que sospechamos que está el óptimo que nos interesa, en este caso el mínimo global. Se define como un diccionario de la siguiente manera:


```python
rango = {'min': -10,
        'max': 10}
```

Luego se define una función, la cual puede llamarse como queramos. Esta función debe tener como parámetro `trial`, que es un ensayo. La funcion debe escoger un valor del rango dado, en este caso mediante `trial.suggest_uniform()` y se evaluará en la expresión a optimizar, la cual va en el `return`. 

> Existen diversas maneras de muestrar valores, dependiendo si del tipo de variable o de la manera que queramos hacerlo. Para más información ir [acá](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial).


```python
def optimize_squared(trial):
    x = trial.suggest_uniform('x', rango['min'], rango['max'])
    return  x** 2 + 1 # nuestra función a optimizar
```

Finalmente para ejecutar la optimización ejecutaremos un estudio, en el cual queremos minimizar.
Elegimos la funcion y el número de ensatos y listo.


```python
study = optuna.create_study(direction = 'minimize')
study.optimize(optimize_squared, n_trials=50)
```

    [32m[I 2021-06-28 23:09:18,931][0m A new study created in memory with name: no-name-8cdd525d-2017-462c-b030-427b6c38ec32[0m
    [32m[I 2021-06-28 23:09:18,937][0m Trial 0 finished with value: 23.902017490831042 and parameters: {'x': -4.785605237671724}. Best is trial 0 with value: 23.902017490831042.[0m
    [32m[I 2021-06-28 23:09:18,941][0m Trial 1 finished with value: 2.597698699418838 and parameters: {'x': -1.2640010678076337}. Best is trial 1 with value: 2.597698699418838.[0m
    [32m[I 2021-06-28 23:09:18,943][0m Trial 2 finished with value: 31.022405959147648 and parameters: {'x': -5.479270568163946}. Best is trial 1 with value: 2.597698699418838.[0m
    [32m[I 2021-06-28 23:09:18,946][0m Trial 3 finished with value: 67.73334832123625 and parameters: {'x': 8.16904818943041}. Best is trial 1 with value: 2.597698699418838.[0m
    [32m[I 2021-06-28 23:09:18,948][0m Trial 4 finished with value: 24.48057869420729 and parameters: {'x': 4.8456762886316795}. Best is trial 1 with value: 2.597698699418838.[0m
    [32m[I 2021-06-28 23:09:18,950][0m Trial 5 finished with value: 21.09260385213723 and parameters: {'x': 4.4824774234944265}. Best is trial 1 with value: 2.597698699418838.[0m
    [32m[I 2021-06-28 23:09:18,952][0m Trial 6 finished with value: 68.65242786697495 and parameters: {'x': -8.225109596046423}. Best is trial 1 with value: 2.597698699418838.[0m
    [32m[I 2021-06-28 23:09:18,954][0m Trial 7 finished with value: 41.22835487370143 and parameters: {'x': -6.342582665894188}. Best is trial 1 with value: 2.597698699418838.[0m
    [32m[I 2021-06-28 23:09:18,955][0m Trial 8 finished with value: 39.87219088922785 and parameters: {'x': 6.234756682439809}. Best is trial 1 with value: 2.597698699418838.[0m
    [32m[I 2021-06-28 23:09:18,956][0m Trial 9 finished with value: 4.320442596901529 and parameters: {'x': -1.8222081650847493}. Best is trial 1 with value: 2.597698699418838.[0m
    [32m[I 2021-06-28 23:09:18,961][0m Trial 10 finished with value: 1.5979800173633043 and parameters: {'x': 0.7732916767709996}. Best is trial 10 with value: 1.5979800173633043.[0m
    [32m[I 2021-06-28 23:09:18,965][0m Trial 11 finished with value: 1.0345315007217994 and parameters: {'x': 0.18582653395519022}. Best is trial 11 with value: 1.0345315007217994.[0m
    [32m[I 2021-06-28 23:09:18,968][0m Trial 12 finished with value: 7.060819899863781 and parameters: {'x': 2.461873250162116}. Best is trial 11 with value: 1.0345315007217994.[0m
    [32m[I 2021-06-28 23:09:18,971][0m Trial 13 finished with value: 3.9300176669297717 and parameters: {'x': 1.711729437419878}. Best is trial 11 with value: 1.0345315007217994.[0m
    [32m[I 2021-06-28 23:09:18,975][0m Trial 14 finished with value: 1.0468883547570575 and parameters: {'x': 0.21653719024005427}. Best is trial 11 with value: 1.0345315007217994.[0m
    [32m[I 2021-06-28 23:09:18,978][0m Trial 15 finished with value: 8.572676590204937 and parameters: {'x': -2.7518496670793877}. Best is trial 11 with value: 1.0345315007217994.[0m
    [32m[I 2021-06-28 23:09:18,981][0m Trial 16 finished with value: 93.63719717806818 and parameters: {'x': 9.624821929680994}. Best is trial 11 with value: 1.0345315007217994.[0m
    [32m[I 2021-06-28 23:09:18,985][0m Trial 17 finished with value: 10.927202880804595 and parameters: {'x': -3.150746400585835}. Best is trial 11 with value: 1.0345315007217994.[0m
    [32m[I 2021-06-28 23:09:18,988][0m Trial 18 finished with value: 1.0628486643725503 and parameters: {'x': 0.2506963589136273}. Best is trial 11 with value: 1.0345315007217994.[0m
    [32m[I 2021-06-28 23:09:18,992][0m Trial 19 finished with value: 11.672926127784672 and parameters: {'x': 3.266944463529289}. Best is trial 11 with value: 1.0345315007217994.[0m
    [32m[I 2021-06-28 23:09:18,996][0m Trial 20 finished with value: 96.53703677659003 and parameters: {'x': -9.77430492549675}. Best is trial 11 with value: 1.0345315007217994.[0m
    [32m[I 2021-06-28 23:09:18,999][0m Trial 21 finished with value: 1.0073560041574146 and parameters: {'x': 0.08576715080620578}. Best is trial 21 with value: 1.0073560041574146.[0m
    [32m[I 2021-06-28 23:09:19,002][0m Trial 22 finished with value: 1.5569047360360124 and parameters: {'x': -0.7462605014577767}. Best is trial 21 with value: 1.0073560041574146.[0m
    [32m[I 2021-06-28 23:09:19,005][0m Trial 23 finished with value: 1.6162128738110146 and parameters: {'x': 0.7849922762747508}. Best is trial 21 with value: 1.0073560041574146.[0m
    [32m[I 2021-06-28 23:09:19,009][0m Trial 24 finished with value: 15.627953973496544 and parameters: {'x': -3.824650830271509}. Best is trial 21 with value: 1.0073560041574146.[0m
    [32m[I 2021-06-28 23:09:19,012][0m Trial 25 finished with value: 12.001754025332767 and parameters: {'x': 3.3168892090832287}. Best is trial 21 with value: 1.0073560041574146.[0m
    [32m[I 2021-06-28 23:09:19,015][0m Trial 26 finished with value: 1.1010945372091034 and parameters: {'x': -0.31795367148234577}. Best is trial 21 with value: 1.0073560041574146.[0m
    [32m[I 2021-06-28 23:09:19,019][0m Trial 27 finished with value: 5.690980650893347 and parameters: {'x': -2.165867182191315}. Best is trial 21 with value: 1.0073560041574146.[0m
    [32m[I 2021-06-28 23:09:19,022][0m Trial 28 finished with value: 3.881763808043724 and parameters: {'x': 1.6975758622352415}. Best is trial 21 with value: 1.0073560041574146.[0m
    [32m[I 2021-06-28 23:09:19,026][0m Trial 29 finished with value: 22.704288239315446 and parameters: {'x': -4.658786133674248}. Best is trial 21 with value: 1.0073560041574146.[0m
    [32m[I 2021-06-28 23:09:19,030][0m Trial 30 finished with value: 34.57070775311028 and parameters: {'x': 5.794023451204722}. Best is trial 21 with value: 1.0073560041574146.[0m
    [32m[I 2021-06-28 23:09:19,034][0m Trial 31 finished with value: 1.786438806863386 and parameters: {'x': 0.8868138513032969}. Best is trial 21 with value: 1.0073560041574146.[0m
    [32m[I 2021-06-28 23:09:19,038][0m Trial 32 finished with value: 1.13515549875073 and parameters: {'x': -0.36763500751523914}. Best is trial 21 with value: 1.0073560041574146.[0m
    [32m[I 2021-06-28 23:09:19,042][0m Trial 33 finished with value: 3.034146749479723 and parameters: {'x': -1.4262351662610633}. Best is trial 21 with value: 1.0073560041574146.[0m
    [32m[I 2021-06-28 23:09:19,046][0m Trial 34 finished with value: 8.00690795051327 and parameters: {'x': 2.647056469082832}. Best is trial 21 with value: 1.0073560041574146.[0m
    [32m[I 2021-06-28 23:09:19,049][0m Trial 35 finished with value: 1.0056068731049788 and parameters: {'x': 0.07487905651768616}. Best is trial 35 with value: 1.0056068731049788.[0m
    [32m[I 2021-06-28 23:09:19,053][0m Trial 36 finished with value: 18.948200564457768 and parameters: {'x': 4.236531666877727}. Best is trial 35 with value: 1.0056068731049788.[0m
    [32m[I 2021-06-28 23:09:19,057][0m Trial 37 finished with value: 3.0220522532879452 and parameters: {'x': -1.4219888372585578}. Best is trial 35 with value: 1.0056068731049788.[0m
    [32m[I 2021-06-28 23:09:19,061][0m Trial 38 finished with value: 41.17041626306449 and parameters: {'x': -6.338013589687583}. Best is trial 35 with value: 1.0056068731049788.[0m
    [32m[I 2021-06-28 23:09:19,064][0m Trial 39 finished with value: 3.9786140127217138 and parameters: {'x': 1.725866163038639}. Best is trial 35 with value: 1.0056068731049788.[0m
    [32m[I 2021-06-28 23:09:19,068][0m Trial 40 finished with value: 19.75443303437414 and parameters: {'x': -4.330638871387701}. Best is trial 35 with value: 1.0056068731049788.[0m
    [32m[I 2021-06-28 23:09:19,072][0m Trial 41 finished with value: 1.067267840439896 and parameters: {'x': 0.25936044501792493}. Best is trial 35 with value: 1.0056068731049788.[0m
    [32m[I 2021-06-28 23:09:19,076][0m Trial 42 finished with value: 1.8919174991521621 and parameters: {'x': -0.9444138389245269}. Best is trial 35 with value: 1.0056068731049788.[0m
    [32m[I 2021-06-28 23:09:19,080][0m Trial 43 finished with value: 1.0853379277082995 and parameters: {'x': 0.2921265611140137}. Best is trial 35 with value: 1.0056068731049788.[0m
    [32m[I 2021-06-28 23:09:19,084][0m Trial 44 finished with value: 5.7274003922895185 and parameters: {'x': -2.1742585845040416}. Best is trial 35 with value: 1.0056068731049788.[0m
    [32m[I 2021-06-28 23:09:19,088][0m Trial 45 finished with value: 2.7663831230225453 and parameters: {'x': 1.3290534688350748}. Best is trial 35 with value: 1.0056068731049788.[0m
    [32m[I 2021-06-28 23:09:19,092][0m Trial 46 finished with value: 9.336070192939113 and parameters: {'x': -2.8872253450222956}. Best is trial 35 with value: 1.0056068731049788.[0m
    [32m[I 2021-06-28 23:09:19,096][0m Trial 47 finished with value: 7.29747062572058 and parameters: {'x': 2.5094761656012157}. Best is trial 35 with value: 1.0056068731049788.[0m
    [32m[I 2021-06-28 23:09:19,100][0m Trial 48 finished with value: 15.399524503716533 and parameters: {'x': 3.7946705395483984}. Best is trial 35 with value: 1.0056068731049788.[0m
    [32m[I 2021-06-28 23:09:19,104][0m Trial 49 finished with value: 1.000010000625206 and parameters: {'x': 0.003162376512363424}. Best is trial 49 with value: 1.000010000625206.[0m


Para chequear los resultados podemos hacer lo siguiente:


```python
print('x óptimo: ', study.best_params)
print('y óptimo', study.best_value)
```

    x óptimo:  {'x': 0.003162376512363424}
    y óptimo 1.000010000625206


También podemos sacar un dataframe en pandas con todos los ensayos realizados y analizarlo a nuestro gusto.


```python
study.trials_dataframe()
```




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
      <th>number</th>
      <th>value</th>
      <th>datetime_start</th>
      <th>datetime_complete</th>
      <th>duration</th>
      <th>params_x</th>
      <th>state</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>23.902017</td>
      <td>2021-06-28 23:09:18.935332</td>
      <td>2021-06-28 23:09:18.936417</td>
      <td>0 days 00:00:00.001085</td>
      <td>-4.785605</td>
      <td>COMPLETE</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2.597699</td>
      <td>2021-06-28 23:09:18.939868</td>
      <td>2021-06-28 23:09:18.940357</td>
      <td>0 days 00:00:00.000489</td>
      <td>-1.264001</td>
      <td>COMPLETE</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>31.022406</td>
      <td>2021-06-28 23:09:18.942424</td>
      <td>2021-06-28 23:09:18.942869</td>
      <td>0 days 00:00:00.000445</td>
      <td>-5.479271</td>
      <td>COMPLETE</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>67.733348</td>
      <td>2021-06-28 23:09:18.944934</td>
      <td>2021-06-28 23:09:18.945429</td>
      <td>0 days 00:00:00.000495</td>
      <td>8.169048</td>
      <td>COMPLETE</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>24.480579</td>
      <td>2021-06-28 23:09:18.947517</td>
      <td>2021-06-28 23:09:18.947971</td>
      <td>0 days 00:00:00.000454</td>
      <td>4.845676</td>
      <td>COMPLETE</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>21.092604</td>
      <td>2021-06-28 23:09:18.949708</td>
      <td>2021-06-28 23:09:18.950099</td>
      <td>0 days 00:00:00.000391</td>
      <td>4.482477</td>
      <td>COMPLETE</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>68.652428</td>
      <td>2021-06-28 23:09:18.951713</td>
      <td>2021-06-28 23:09:18.952076</td>
      <td>0 days 00:00:00.000363</td>
      <td>-8.225110</td>
      <td>COMPLETE</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>41.228355</td>
      <td>2021-06-28 23:09:18.953625</td>
      <td>2021-06-28 23:09:18.954000</td>
      <td>0 days 00:00:00.000375</td>
      <td>-6.342583</td>
      <td>COMPLETE</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8</td>
      <td>39.872191</td>
      <td>2021-06-28 23:09:18.955054</td>
      <td>2021-06-28 23:09:18.955310</td>
      <td>0 days 00:00:00.000256</td>
      <td>6.234757</td>
      <td>COMPLETE</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
      <td>4.320443</td>
      <td>2021-06-28 23:09:18.956372</td>
      <td>2021-06-28 23:09:18.956664</td>
      <td>0 days 00:00:00.000292</td>
      <td>-1.822208</td>
      <td>COMPLETE</td>
    </tr>
    <tr>
      <th>10</th>
      <td>10</td>
      <td>1.597980</td>
      <td>2021-06-28 23:09:18.957751</td>
      <td>2021-06-28 23:09:18.961695</td>
      <td>0 days 00:00:00.003944</td>
      <td>0.773292</td>
      <td>COMPLETE</td>
    </tr>
    <tr>
      <th>11</th>
      <td>11</td>
      <td>1.034532</td>
      <td>2021-06-28 23:09:18.962870</td>
      <td>2021-06-28 23:09:18.965450</td>
      <td>0 days 00:00:00.002580</td>
      <td>0.185827</td>
      <td>COMPLETE</td>
    </tr>
    <tr>
      <th>12</th>
      <td>12</td>
      <td>7.060820</td>
      <td>2021-06-28 23:09:18.965972</td>
      <td>2021-06-28 23:09:18.968642</td>
      <td>0 days 00:00:00.002670</td>
      <td>2.461873</td>
      <td>COMPLETE</td>
    </tr>
    <tr>
      <th>13</th>
      <td>13</td>
      <td>3.930018</td>
      <td>2021-06-28 23:09:18.969124</td>
      <td>2021-06-28 23:09:18.971723</td>
      <td>0 days 00:00:00.002599</td>
      <td>1.711729</td>
      <td>COMPLETE</td>
    </tr>
    <tr>
      <th>14</th>
      <td>14</td>
      <td>1.046888</td>
      <td>2021-06-28 23:09:18.972269</td>
      <td>2021-06-28 23:09:18.975136</td>
      <td>0 days 00:00:00.002867</td>
      <td>0.216537</td>
      <td>COMPLETE</td>
    </tr>
    <tr>
      <th>15</th>
      <td>15</td>
      <td>8.572677</td>
      <td>2021-06-28 23:09:18.975670</td>
      <td>2021-06-28 23:09:18.978126</td>
      <td>0 days 00:00:00.002456</td>
      <td>-2.751850</td>
      <td>COMPLETE</td>
    </tr>
    <tr>
      <th>16</th>
      <td>16</td>
      <td>93.637197</td>
      <td>2021-06-28 23:09:18.978572</td>
      <td>2021-06-28 23:09:18.981806</td>
      <td>0 days 00:00:00.003234</td>
      <td>9.624822</td>
      <td>COMPLETE</td>
    </tr>
    <tr>
      <th>17</th>
      <td>17</td>
      <td>10.927203</td>
      <td>2021-06-28 23:09:18.982243</td>
      <td>2021-06-28 23:09:18.985387</td>
      <td>0 days 00:00:00.003144</td>
      <td>-3.150746</td>
      <td>COMPLETE</td>
    </tr>
    <tr>
      <th>18</th>
      <td>18</td>
      <td>1.062849</td>
      <td>2021-06-28 23:09:18.986136</td>
      <td>2021-06-28 23:09:18.988484</td>
      <td>0 days 00:00:00.002348</td>
      <td>0.250696</td>
      <td>COMPLETE</td>
    </tr>
    <tr>
      <th>19</th>
      <td>19</td>
      <td>11.672926</td>
      <td>2021-06-28 23:09:18.988975</td>
      <td>2021-06-28 23:09:18.992569</td>
      <td>0 days 00:00:00.003594</td>
      <td>3.266944</td>
      <td>COMPLETE</td>
    </tr>
    <tr>
      <th>20</th>
      <td>20</td>
      <td>96.537037</td>
      <td>2021-06-28 23:09:18.993133</td>
      <td>2021-06-28 23:09:18.995906</td>
      <td>0 days 00:00:00.002773</td>
      <td>-9.774305</td>
      <td>COMPLETE</td>
    </tr>
    <tr>
      <th>21</th>
      <td>21</td>
      <td>1.007356</td>
      <td>2021-06-28 23:09:18.996279</td>
      <td>2021-06-28 23:09:18.999083</td>
      <td>0 days 00:00:00.002804</td>
      <td>0.085767</td>
      <td>COMPLETE</td>
    </tr>
    <tr>
      <th>22</th>
      <td>22</td>
      <td>1.556905</td>
      <td>2021-06-28 23:09:18.999442</td>
      <td>2021-06-28 23:09:19.002391</td>
      <td>0 days 00:00:00.002949</td>
      <td>-0.746261</td>
      <td>COMPLETE</td>
    </tr>
    <tr>
      <th>23</th>
      <td>23</td>
      <td>1.616213</td>
      <td>2021-06-28 23:09:19.002844</td>
      <td>2021-06-28 23:09:19.005757</td>
      <td>0 days 00:00:00.002913</td>
      <td>0.784992</td>
      <td>COMPLETE</td>
    </tr>
    <tr>
      <th>24</th>
      <td>24</td>
      <td>15.627954</td>
      <td>2021-06-28 23:09:19.006150</td>
      <td>2021-06-28 23:09:19.009011</td>
      <td>0 days 00:00:00.002861</td>
      <td>-3.824651</td>
      <td>COMPLETE</td>
    </tr>
    <tr>
      <th>25</th>
      <td>25</td>
      <td>12.001754</td>
      <td>2021-06-28 23:09:19.009392</td>
      <td>2021-06-28 23:09:19.012153</td>
      <td>0 days 00:00:00.002761</td>
      <td>3.316889</td>
      <td>COMPLETE</td>
    </tr>
    <tr>
      <th>26</th>
      <td>26</td>
      <td>1.101095</td>
      <td>2021-06-28 23:09:19.012580</td>
      <td>2021-06-28 23:09:19.015344</td>
      <td>0 days 00:00:00.002764</td>
      <td>-0.317954</td>
      <td>COMPLETE</td>
    </tr>
    <tr>
      <th>27</th>
      <td>27</td>
      <td>5.690981</td>
      <td>2021-06-28 23:09:19.015695</td>
      <td>2021-06-28 23:09:19.018904</td>
      <td>0 days 00:00:00.003209</td>
      <td>-2.165867</td>
      <td>COMPLETE</td>
    </tr>
    <tr>
      <th>28</th>
      <td>28</td>
      <td>3.881764</td>
      <td>2021-06-28 23:09:19.019647</td>
      <td>2021-06-28 23:09:19.022072</td>
      <td>0 days 00:00:00.002425</td>
      <td>1.697576</td>
      <td>COMPLETE</td>
    </tr>
    <tr>
      <th>29</th>
      <td>29</td>
      <td>22.704288</td>
      <td>2021-06-28 23:09:19.022680</td>
      <td>2021-06-28 23:09:19.026501</td>
      <td>0 days 00:00:00.003821</td>
      <td>-4.658786</td>
      <td>COMPLETE</td>
    </tr>
    <tr>
      <th>30</th>
      <td>30</td>
      <td>34.570708</td>
      <td>2021-06-28 23:09:19.027023</td>
      <td>2021-06-28 23:09:19.030706</td>
      <td>0 days 00:00:00.003683</td>
      <td>5.794023</td>
      <td>COMPLETE</td>
    </tr>
    <tr>
      <th>31</th>
      <td>31</td>
      <td>1.786439</td>
      <td>2021-06-28 23:09:19.031141</td>
      <td>2021-06-28 23:09:19.034242</td>
      <td>0 days 00:00:00.003101</td>
      <td>0.886814</td>
      <td>COMPLETE</td>
    </tr>
    <tr>
      <th>32</th>
      <td>32</td>
      <td>1.135155</td>
      <td>2021-06-28 23:09:19.034923</td>
      <td>2021-06-28 23:09:19.038423</td>
      <td>0 days 00:00:00.003500</td>
      <td>-0.367635</td>
      <td>COMPLETE</td>
    </tr>
    <tr>
      <th>33</th>
      <td>33</td>
      <td>3.034147</td>
      <td>2021-06-28 23:09:19.038979</td>
      <td>2021-06-28 23:09:19.042155</td>
      <td>0 days 00:00:00.003176</td>
      <td>-1.426235</td>
      <td>COMPLETE</td>
    </tr>
    <tr>
      <th>34</th>
      <td>34</td>
      <td>8.006908</td>
      <td>2021-06-28 23:09:19.042866</td>
      <td>2021-06-28 23:09:19.045919</td>
      <td>0 days 00:00:00.003053</td>
      <td>2.647056</td>
      <td>COMPLETE</td>
    </tr>
    <tr>
      <th>35</th>
      <td>35</td>
      <td>1.005607</td>
      <td>2021-06-28 23:09:19.046355</td>
      <td>2021-06-28 23:09:19.049460</td>
      <td>0 days 00:00:00.003105</td>
      <td>0.074879</td>
      <td>COMPLETE</td>
    </tr>
    <tr>
      <th>36</th>
      <td>36</td>
      <td>18.948201</td>
      <td>2021-06-28 23:09:19.049848</td>
      <td>2021-06-28 23:09:19.053248</td>
      <td>0 days 00:00:00.003400</td>
      <td>4.236532</td>
      <td>COMPLETE</td>
    </tr>
    <tr>
      <th>37</th>
      <td>37</td>
      <td>3.022052</td>
      <td>2021-06-28 23:09:19.053992</td>
      <td>2021-06-28 23:09:19.057380</td>
      <td>0 days 00:00:00.003388</td>
      <td>-1.421989</td>
      <td>COMPLETE</td>
    </tr>
    <tr>
      <th>38</th>
      <td>38</td>
      <td>41.170416</td>
      <td>2021-06-28 23:09:19.057840</td>
      <td>2021-06-28 23:09:19.060986</td>
      <td>0 days 00:00:00.003146</td>
      <td>-6.338014</td>
      <td>COMPLETE</td>
    </tr>
    <tr>
      <th>39</th>
      <td>39</td>
      <td>3.978614</td>
      <td>2021-06-28 23:09:19.061569</td>
      <td>2021-06-28 23:09:19.064579</td>
      <td>0 days 00:00:00.003010</td>
      <td>1.725866</td>
      <td>COMPLETE</td>
    </tr>
    <tr>
      <th>40</th>
      <td>40</td>
      <td>19.754433</td>
      <td>2021-06-28 23:09:19.065140</td>
      <td>2021-06-28 23:09:19.068616</td>
      <td>0 days 00:00:00.003476</td>
      <td>-4.330639</td>
      <td>COMPLETE</td>
    </tr>
    <tr>
      <th>41</th>
      <td>41</td>
      <td>1.067268</td>
      <td>2021-06-28 23:09:19.069304</td>
      <td>2021-06-28 23:09:19.072759</td>
      <td>0 days 00:00:00.003455</td>
      <td>0.259360</td>
      <td>COMPLETE</td>
    </tr>
    <tr>
      <th>42</th>
      <td>42</td>
      <td>1.891917</td>
      <td>2021-06-28 23:09:19.073384</td>
      <td>2021-06-28 23:09:19.076815</td>
      <td>0 days 00:00:00.003431</td>
      <td>-0.944414</td>
      <td>COMPLETE</td>
    </tr>
    <tr>
      <th>43</th>
      <td>43</td>
      <td>1.085338</td>
      <td>2021-06-28 23:09:19.077278</td>
      <td>2021-06-28 23:09:19.080316</td>
      <td>0 days 00:00:00.003038</td>
      <td>0.292127</td>
      <td>COMPLETE</td>
    </tr>
    <tr>
      <th>44</th>
      <td>44</td>
      <td>5.727400</td>
      <td>2021-06-28 23:09:19.080888</td>
      <td>2021-06-28 23:09:19.084064</td>
      <td>0 days 00:00:00.003176</td>
      <td>-2.174259</td>
      <td>COMPLETE</td>
    </tr>
    <tr>
      <th>45</th>
      <td>45</td>
      <td>2.766383</td>
      <td>2021-06-28 23:09:19.084748</td>
      <td>2021-06-28 23:09:19.088206</td>
      <td>0 days 00:00:00.003458</td>
      <td>1.329053</td>
      <td>COMPLETE</td>
    </tr>
    <tr>
      <th>46</th>
      <td>46</td>
      <td>9.336070</td>
      <td>2021-06-28 23:09:19.088738</td>
      <td>2021-06-28 23:09:19.092169</td>
      <td>0 days 00:00:00.003431</td>
      <td>-2.887225</td>
      <td>COMPLETE</td>
    </tr>
    <tr>
      <th>47</th>
      <td>47</td>
      <td>7.297471</td>
      <td>2021-06-28 23:09:19.092772</td>
      <td>2021-06-28 23:09:19.096148</td>
      <td>0 days 00:00:00.003376</td>
      <td>2.509476</td>
      <td>COMPLETE</td>
    </tr>
    <tr>
      <th>48</th>
      <td>48</td>
      <td>15.399525</td>
      <td>2021-06-28 23:09:19.096937</td>
      <td>2021-06-28 23:09:19.100724</td>
      <td>0 days 00:00:00.003787</td>
      <td>3.794671</td>
      <td>COMPLETE</td>
    </tr>
    <tr>
      <th>49</th>
      <td>49</td>
      <td>1.000010</td>
      <td>2021-06-28 23:09:19.101393</td>
      <td>2021-06-28 23:09:19.104337</td>
      <td>0 days 00:00:00.002944</td>
      <td>0.003162</td>
      <td>COMPLETE</td>
    </tr>
  </tbody>
</table>
</div>



{% include alert success='Los resultados son los esperados, **x** debe ser cercano a 0 e **y** cercano a 1.'%}

### Guardando los resultados en wandb 

Para almacenar los resultados en Weights & Biases podemos hacer `wandb.init()` y agregaremos el nombre de un proyecto. Además podemos agregar algunos tags para identificar el run que haremos.

Un run sería equivalente a un modelo en el que se probarán distintos hiperparámetros. Cada uno de los ensayos serán combinaciones distintas de hiperparámetros y se irán almacenando mediante `run.log()`. En nuestro caso almacenaremos step como el número del experimento y `trial.params` almacenará todos los parámetros que están siendo optimizados, en nuestro caso x. Luego podemos mediante `trial.value` almacenar el resultado de nuestro valor objetivo.

Finalmente mediante `run.summary` podemos almacenar lo que nosotros queramos. En mi caso me  gusta almacenar el mejor `y` y los parámetros óptimos.


```python
with wandb.init(project="nuevo-proyecto",
                tags = ['optimización','cuadrática']) as run:
    for step, trial in enumerate(study.trials):

        run.log(trial.params, step = step)
        run.log({"y": trial.value})

    run.summary['best_y'] = study.best_value
    run.summary['best_params'] = study.best_params
```

El output de Weights & Biases se ve más o menos así:

---

Tracking run with wandb version 0.10.32
Syncing run <strong style="color:#cdcd00">dark-sky-3</strong> to <a href="https://wandb.ai" target="_blank">Weights & Biases</a> <a href="https://docs.wandb.com/integrations/jupyter.html" target="_blank">(Documentation)</a>.<br/>
Project page: <a href="https://wandb.ai/datacuber/nuevo-proyecto" target="_blank">https://wandb.ai/datacuber/nuevo-proyecto</a><br/>
Run page: <a href="https://wandb.ai/datacuber/nuevo-proyecto/runs/29ttsi72" target="_blank">https://wandb.ai/datacuber/nuevo-proyecto/runs/29ttsi72</a><br/>
Run data is saved locally in <code>/home/alfonso/Documents/kaggle/titanic/wandb/run-20210628_230927-29ttsi72</code><br/><br/>

<br/>Waiting for W&B process to finish, PID 442316<br/>Program ended successfully.
VBox(children=(Label(value=' 0.00MB of 0.00MB uploaded (0.00MB deduped)\r'), FloatProgress(value=0.0, max=1.0)…
Find user logs for this run at: <code>/home/alfonso/Documents/kaggle/titanic/wandb/run-20210628_230927-29ttsi72/logs/debug.log</code>

Find internal logs for this run at: <code>/home/alfonso/Documents/kaggle/titanic/wandb/run-20210628_230927-29ttsi72/logs/debug-internal.log</code>



<h3>Run summary:</h3><br/><style>
    table.wandb td:nth-child(1) { padding: 0 10px; text-align: right }
    </style><table class="wandb">
<tr><td>x</td><td>0.00316</td></tr><tr><td>y</td><td>1.00001</td></tr><tr><td>_runtime</td><td>5</td></tr><tr><td>_timestamp</td><td>1624936172</td></tr><tr><td>_step</td><td>49</td></tr><tr><td>best_y</td><td>1.00001</td></tr></table>



<h3>Run history:</h3><br/><style>
    table.wandb td:nth-child(1) { padding: 0 10px; text-align: right }
    </style><table class="wandb">
<tr><td>x</td><td>▃▄▃▇▆▂▂▇▅▅▅▅▄█▃▅▁▅▄▅▆▄▄▅▇▅▄▄▅▆▄▂▃▅▄▅▅▃▅▅</td></tr><tr><td>y</td><td>▃▁▃▆▂▆▄▄▁▁▁▁▂█▂▁█▁▁▁▂▁▁▁▃▁▁▁▁▂▁▄▂▁▁▁▁▂▁▁</td></tr><tr><td>_runtime</td><td>▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁</td></tr><tr><td>_timestamp</td><td>▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁</td></tr><tr><td>_step</td><td>▁▁▁▁▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇███</td></tr></table><br/>

Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)

<br/>Synced <strong style="color:#cdcd00">dark-sky-3</strong>: <a href="https://wandb.ai/datacuber/nuevo-proyecto/runs/29ttsi72" target="_blank">https://wandb.ai/datacuber/nuevo-proyecto/runs/29ttsi72</a><br/>


---

Los resultados pueden ser visualizados en el portal de **Weights & Biases** mediante el link entregado. En el portal uno puede agregar todos los gráficos que requiera. La idea es poder visualizar el proyecto de la mejor manera posible:

![]({{ site.urlimg }}wandb-optuna/wandb_screenshot.png)



En este rápido tutorial se puede ver cómo utilizar **Weights & Biases** de manera muy rápida. Espero poder realizar otro en el que se puedan ver más beneficios de utilizar esta herramienta pero enfocado de lleno en su uso con modelos de Machine Learning.

<!-- 
```python
api = wandb.Api()
# run is specified by <entity>/<project>/<run id>
run = api.run("datacuber/optuna/2k1vy0cj")
metrics_df = run.history()
metrics_df
```




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
      <th>_step</th>
      <th>x</th>
      <th>_runtime</th>
      <th>mse</th>
      <th>z</th>
      <th>_timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>-6.377177</td>
      <td>2</td>
      <td>85.680586</td>
      <td>8.816779</td>
      <td>1613775185</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>-0.069629</td>
      <td>2</td>
      <td>169.460192</td>
      <td>7.543658</td>
      <td>1613775185</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>-4.119354</td>
      <td>2</td>
      <td>315.556214</td>
      <td>-5.822274</td>
      <td>1613775185</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>-9.321491</td>
      <td>2</td>
      <td>14.458745</td>
      <td>3.759513</td>
      <td>1613775185</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>-4.059086</td>
      <td>2</td>
      <td>10.451453</td>
      <td>1.413108</td>
      <td>1613775185</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>95</th>
      <td>95</td>
      <td>4.880971</td>
      <td>2</td>
      <td>0.139162</td>
      <td>-1.627007</td>
      <td>1613775185</td>
    </tr>
    <tr>
      <th>96</th>
      <td>96</td>
      <td>4.535738</td>
      <td>2</td>
      <td>13.287564</td>
      <td>-3.090475</td>
      <td>1613775185</td>
    </tr>
    <tr>
      <th>97</th>
      <td>97</td>
      <td>5.190690</td>
      <td>2</td>
      <td>3.631573</td>
      <td>-2.548179</td>
      <td>1613775185</td>
    </tr>
    <tr>
      <th>98</th>
      <td>98</td>
      <td>5.976324</td>
      <td>2</td>
      <td>10.780678</td>
      <td>-3.629859</td>
      <td>1613775185</td>
    </tr>
    <tr>
      <th>99</th>
      <td>99</td>
      <td>4.888123</td>
      <td>2</td>
      <td>0.010747</td>
      <td>-1.495896</td>
      <td>1613775185</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 6 columns</p>
</div>




```python
system_metrics = run.history(stream = 'events')
system_metrics
```




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
      <th>system.network.sent</th>
      <th>system.network.recv</th>
      <th>system.disk</th>
      <th>_wandb</th>
      <th>system.gpu.0.temp</th>
      <th>system.gpu.0.memory</th>
      <th>system.gpu.0.gpu</th>
      <th>_runtime</th>
      <th>system.proc.memory.rssMB</th>
      <th>system.proc.memory.availableMB</th>
      <th>system.cpu</th>
      <th>system.proc.cpu.threads</th>
      <th>system.memory</th>
      <th>system.proc.memory.percent</th>
      <th>system.gpu.0.memoryAllocated</th>
      <th>_timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>27276</td>
      <td>36513</td>
      <td>28.5</td>
      <td>True</td>
      <td>46</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>301.94</td>
      <td>28374.51</td>
      <td>9.05</td>
      <td>23.5</td>
      <td>11.4</td>
      <td>0.94</td>
      <td>4.27</td>
      <td>1613775186</td>
    </tr>
  </tbody>
</table>
</div>




```python
run.summary
```




    {'z': -1.4958962538521758, 'mse': 0.010747402305021076, 'best': {'x': 4.888122828892169, 'z': -1.4958962538521758}, '_step': 99, '_runtime': 2, '_timestamp': 1613775185, 'x': 4.888122828892169} -->

[**Alfonso**]({{ site.baseurl }}/contact/)

