---
permalink: /robertuito/ 
title: "Transformers, el Robertuito"
subheadline: "Fine Tuning para HateSpeech en Español"
teaser: "Robertuito, un modelo RoBERTa entrenado en 500 millones de Tweets."
# layout: page-fullwidth
usemathjax: true
category: pp
header: no
image:
    thumb: hate_speech/header.jpeg
tags:
- pytorch
- tutorial
- dl
published: false
---

![picture of me]({{ site.urlimg }}hate_speech/header.jpeg){: .left .show-for-large-up .hide-for-print width="250"}
![picture of me]({{ site.urlimg }}hate_speech/header.jpeg){: .center .hide-for-large-up width="500"}

Hate Speech. Suena bonito, pero es probablemente una de nuestras peores costumbres. Todos tenemos gente que odiamos, y a veces hasta discutir y debatir y pelear es entretenido (a mi me gusta). Pero obviamente esparcir odio es algo que puede afectar demasiado, en especial en la era de las redes sociales <!--more-->. 

La verdad es que trabajos sobre detección de odios hay por montones. De hecho las plataformas como Twitter o Facebook están constantemente desarrollando herramientas que permitan detectar y eliminar este tipo de contenido. Pero la verdad es que no es tan fácil, en especial porque el lenguaje es muy subjetivo. Y dependiendo del idioma puede ser que incluso las palabras que usamos para expresar odio no sean las mismas. E incluso, como ocurre en mi país, Chile, las palabras del uso cotidiano son las mismas que se utilizan al momento de hablar con mucho odio. Eso hace que detectar Odio en Español chileno sea muy desafiante. 

En este artículo me gustaría mostrar uno de los modelos que intenté durante la [Datathon 2022]({{ site.baseurl }}/datathon/) y que si bien no dio tan buenos resultados para la competencia (principalmente porque necesitaba detectar más cosas que sólo Odio) quiero mostrar uno de los proceso de Fine-Tuning que apliqué en el cuál sí hubo resultados para detectar Odio. 

## El Modelo

Bueno el Modelo es el [RoberTuito](https://arxiv.org/pdf/2111.09453.pdf) es un modelo desarrollado en Argentina supongo que un poco siguiendo los pasos del BETO en Chile. La idea es poder entrenar algunos de los modelos basados en Transformers más famosos pero en idioma Español (ya que casi todo el research en este tipo de modelos se ha hecho en Inglés). El modelo está basado en RoBERTa, que no es un modelo nuevo, de hecho es un BERT, pero entrenado siguiendo otras prácticas, que de acuerdo a los autores entrega más estabilidad y robustez al entrenarse en muchísima más data. En el caso de RoberTuito se entrenó en 500 millones de Tweets, lo cual es bastante.

Otra cosa que es interesante es que el modelo se probó en varias tareas, una de ellas es Hate Speech, por lo que al momento de competir realmente pensé que podría dar algunas ventajas competitivas. Ahora el modelo promete ciertas métricas de desempeño, pero como siempre, hay que probarlo.

Ahora, acá hay algunas discrepancias. El paper muestra resultados un poco más optimistas de lo que muestra la implementación en Huggingface, por lo que la idea es chequear el poder de generalización de este modelo en un dataset que es bastante distinto al que fue entrenado, ya que como comenté está en Chileno, y mucho de la tokenización y vocabulario es distinto al que se usa en el corpus en el cual fue pre-entrenado.
Los resultados publicados en el paper son los siguientes:
![picture of me]({{ site.urlimg }}hate_speech/robertuito_paper.png){: .center .show-for-large-up .hide-for-print width="500"}
![picture of me]({{ site.urlimg }}hate_speech/robertuito_paper.png){: .center .hide-for-large-up width="500"}

Los resultados publicados en el repositorio de HuggingFace son estos:

![picture of me]({{ site.urlimg }}hate_speech/robertuito_hf.png){: .center .show-for-large-up .hide-for-print width="500"}
![picture of me]({{ site.urlimg }}hate_speech/robertuito_hf.png){: .center .hide-for-large-up width="500"}

## BERT

Como dijimos RoberTuito es básicamente un BERT pero entrenado de otra manera. BERT es un modelo de lenguaje que fue desarrollado por Google en 2018 por [Devlin et al.](https://arxiv.org/abs/1810.04805). Es un modelo pero que está basado sólo en el Encoder de un Transformer, por lo que sus fortalezas están en poder extraer features, o generar un embedding que pueda represetar un texto para que luego una red neuronal pueda tomarlo y generar predicciones. Fue entrenado una técnica llamada Masked Language Model de manera semi-supervisada, que a pesar de ser simple es super efectiva y creativa. 

Básicamente se le entrega todo la data, y se ocultan, ciertas palabras de tal manera que el modelo pueda predecir cuál es la palabra que se ocultó. Esto es muy útil porque permite que el modelo aprenda a entender el contexto de una frase, y no sólo a memorizar palabras. Y además el proceso de etiquetado es muchísimo más rápido.


## Solución Propuesta

La verdad es que tenía todas mis esperanzas puestas en este RoBERTuito, y si alguno me quiere ayudar a decifrar por qué no dió tan buenos resultados les agradecería montones. 
La implementación pre-entrenada de RoBERTuito se puede encontrar en 🤗 HuggingFace (debido a que es un transformer) pero de una manera un poco extraña que es mediante una librería llamada `pysentimiento`. Esta librería tiene la verdad es que varias funcionalidades bien simpáticas las cuales pueden encontrar en su [Github](https://github.com/pysentimiento/pysentimiento). 

Siendo sincero hice varias pruebas con este modelo utilizando las funcionalidades directamente de la librería pero también haciendo un fine-tuning del modelo, que es lo que voy a mostrarles ahora. 

{% include alert success='Esta semana fue una semana bien importante para Pytorch, se lanzó oficialmente la versión 2.0 el cuál permite el uso de `torch.compile()` para poder compilar/acelerar directamente un modelo sin mucho cambio. Además como miembro de la ⚡⚡Lightning League⚡⚡ tengo que mencionar que también se liberó la versión de Lightning 2.0 (ex Pytorch Lightning), el cual formaliza ya varios cambios que se venían dando hace un tiempo, por lo que adapté el código a los últimos cambios de Lightning que voy aprovechar de presentar.' %}

Básicamente Lightning ahora contiene no sólo Pytorch Lightning, sino también Fabric y Lightning Apps. Probablemente cada uno de estos requiere un tutorial por separado (el cuál habrá), pero principalmente Fabric es el ex Lightning Lite el cuál permite agregar rápidamente características de Lightning a un Modelo en Pytorch Nativo sin cambiar su código (digamos que es sólo un wrapper). Finalmente Lightning Apps permitirá facilitar el deploy de Apps que hacen uso de modelos de Machine o Deep Learning. 

{% include alert alert='Lamentablemente no voy a poder mostrar los beneficios de `torch.compile()` debido que parece ser que los modelos de `pysentimiento` no son compatibles. Como lo sé? La verdad no estoy del todo seguro, pero obtuve un error bien feo el cuál no logré encontrar en ninguna parte a qué se debe. Si alguien sabe y me quiere ayudar estaría muy agradecido, pero este el gran problema de los frameworks de Deep Learning, como trabajan con CUDA, sus errores son muy crípticos:

```bash
AssertionError                            Traceback (most recent call last)
File <timed exec>:4
...
You can suppress this exception and fall back to eager by setting:
    torch._dynamo.config.suppress_errors = True
```
' %}

Bueno, el código principal que entrena el modelo completo es bastante sencillo y está organizado como un notebook de entrenamiento (es bastante corto, pero permite la facilidad de mover el código y probar rápidamente) y un directorio con los distintos módulo de Lightning para el RoBERTuito como se ve a continuación:

![picture of me]({{ site.urlimg }}hate_speech/robertuito.png){: .center width="500"}

```python
import os
import numpy as np
import lightning as L

from box import Box

from robertuito import create_folds, import_data, train_model, evaluate, predict

L.seed_everything(42, workers=True)

# solving forking issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"
```
{: title="Importación de librerías y fijar semilla para reproducibilidad."}

Quizás acá hay dos puntos bien importantes a recalcar: el primero es que ahora la importación es como `import lightning as L`.  Esto dado el cambio que mencionamos antes de unificar los productos ofrecidos por Lightning en una sóla librería. Además suena más cool y es más corto que `pytorch lightning`.

Además como utilizamos un módulo, podemos importar todo directamente desde el módulo robertuito, lo que permite un import mucho más limpio y ordenado.

```python
LABELS = [
    "Odio",
    # "Mujeres",
    # "Comunidad LGBTQ+",
    # "Comunidades Migrantes",
    # "Pueblos Originarios",
]

TRAIN_PATH = "public_data/tweets_train.csv"
TEST_PATH = "Datathon-2022-full/Datos/tweets_test.csv"
train, test = import_data(TRAIN_PATH, TEST_PATH)
train = create_folds(train, LABELS)
```
{: title="Importación de los datos."}

Acá muestro el código de la siguiente forma dado que la [Datathon]({{ site.baseurl }}/datathon/) contemplaba la implementación de un modelo que predijera 5 categorías. Como conté anteriormente, el RoBERTuito no dió tan buenos resultados para predecir las 5 categorías, pero sí es útil para hacer un fine-tuning de Odio (tarea para el cual el modelo ya había sido efectivamente pre-entrenado).

Podemos notar que `import_data()` permite importar los sets de train y test (ojo, que al momento de la competencia no teníamos acceso al set de test, pero ahora lo muestro para ver la capacidad del modelo gracias a que me lo facilitaron amablemente). Y por otro lado `create_folds()` creará una columna adicional que define qué filas corresponderán al fold de validación en cada iteración.

```python
dm_config = Box(dict(
    train = train,
    test = test,
    labels = LABELS,
    batch_size = 32,
    tokenizer = 'pysentimiento/robertuito-hate-speech',
))

model_config = Box(dict(
    model_name = 'pysentimiento/robertuito-hate-speech',
    dropout = 0.2,
    hidden_size = 768,
    n_labels = len(LABELS),
    train_size = 56,
    batch_size = dm_config.batch_size,
    warmup=0.2,
    w_decay=0.001,
    lr = 3e-4
))

training_config = Box(dict(
    max_epochs = 30,
    patience = 10,
    fast_dev_run=False,
    overfit_batches=0
))
```
{: title="Configuración del modelo."}

`dm_config` contendrá la configuración del Data Module: Cuáles son los dataframes de train y test, cuáles serán las Etiquetas a utilizar, el `batch_size` y el tokenizador del modelo el cuál sacaremos desde 🤗 Transformers.

`model_config` contendrá hiperparámetros del modelo a utilizar: Modelo Pre-entrenado, dropout rate entre la unión del backbone y el clasificador, `decay` y `learning_rate` para el optimizador que en este caso es `AdamW` (Adam con weight decay).

Finalmente `training_config` tendrá configuración del proceso de entrenamiento: Número de `epochs`, `patience` para el EarlyStopping y flags para debuggear.

{% include alert tip='Todos los diccionarios están envueltos en un `Box`. `Box` permite envolver un diccionario para poder llamarlo con notación de punto. En vez de dict["key"] puedo llamar los elementos como dict.key, lo cual me permite ahorrar algunos caracteres al escribir.'%}

El entrenamiento del modelo se describe acá:

```python
score = []
preds = np.empty((len(test), 5))
for fold in range(5):
    trainer, model, dm= train_model([fold], dm_config, model_config, training_config)
    f1 = evaluate(trainer, model, dm, threshold = 0.5, custom=False)
    preds[:, fold] = predict(trainer, model, dm, validation=False).reshape(-1)
    print(f"Fold {fold} F1: {f1}")
    score.append(f1)
```
{: title="Entrenamiento del Modelo."}


Es bastante sencillo nuevamente. En el cual se crea una lista para guardar los puntajes de cada fold, además de un numpy array para guardar las predicciones. 

El entrenamiento cuenta con 3 etapas muy sencillas `train_model()` en el cual se entrena el modelo, `evaluate()` para poder evaluar el modelo (se evalúa en los 4 folds de train y se mide el puntaje en el fold de validación). `predict()` genera las predicciones en formato tipo blending (una forma bien popular en Kaggle que no está en ninguna otra parte.) Básicamente entrenamos el modelo con 4 de 5 folds y predecimos en todo el test set. Este tipo de predicción se suele utilizar en librerías de Stacking. Les dejo acá un gif explicativo que se puede encontrar [acá](https://github.com/vecxoz/vecstack):

El puntaje con el que me presentaba al leaderboard lo calculaba simplemente como el promedio de los folds:

```python
print(f"Mean 5 Fold CV Score: {np.mean(score)}")
```

![picture of me](https://raw.githubusercontent.com/vecxoz/vecstack/master/pic/animation1.gif){: .center width="500"}

```python
from sklearn.metrics import f1_score
y_pred = np.where(preds.mean(axis = 1) >= 0.5, 1, 0)
f1_score(test.Odio.values, y_pred)
```
{: title="Medición en el Test Set."}

Obviamente esta parte no era posible durante la competencia, pero la implementamos ahora para simular el puntaje en datos no vistos por el modelo. Puntaje que era calculado de manera oculta en la plataforma de competencia. En este caso sólo calculamos un simple `f1_score` debido a que estamos utilizando sólo una clase. 


{% include alert success='Este es básicamente todo el modelo. Obviamente de manera resumida. Pero con poquitas líneas de código es posible hacer un fine-tuning de un Transformer. Para los que le interesa ver en detalle el código vamos a ir parte por parte. Vamos a ir explicando por archivo para mostrar en qué consiste cada parte.'%}

```python
from .dataset import HateDataset
from .datamodule import HateDataModule
from .model import RoberTuito
from .trainer import evaluate, train_model, predict
from .utils import create_folds, import_data

__all__ = [
    "HateDataset",
    "HateDataModule",
    "RoberTuito",
    "evaluate",
    "train_model",
    "predict",
    "create_folds",
    "import_data",
]
```
{: title="__init__.py"}

Estoy seguro que muchos no conocen esto acerca de crear módulos en Python.

El archivo `__init__.py` permite definir que una carpeta será un modulo, además de permitir acortar las rutas de importación. Por ejemplo la primera línea: `from .dataset import HateDataset` permite acortar el import del `HateDataset` ya que importa el HateDataset desde el archivo `dataset.py` que está en la carpeta/módulo robertuito. Esto permite que podamos llamar como `from robertuito import HateDataset`, en vez de `from robertuito.dataset import HateDataset`. Cada línea permitirá importar hacer lo mismo para cada elemento, de esa manera podría eventualmente tener un archivo para cada clase/función que requiera importar, lo cual permite que todo sea más ordenado.

```python
import pandas as pd
from sklearn.model_selection import KFold

def import_data(train_path, test_path):
    train = pd.read_csv(train_path, index_col=0)
    test = pd.read_csv(test_path, index_col=0)
    return train, test


def create_folds(df, labels, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    df["kfold"] = -1

    for fold, (_, val_idx) in enumerate(kf.split(df.drop(columns=labels))):
        df.loc[val_idx, "kfold"] = fold

    return df
```
{: title="utils.py"}

En este archivo sólo cree funciones utilitarias para poder importar mis archivos CSV sencillamente recibiendo los paths de cada archivo y luego crear los folds para un cierto `df`. *Ojo: Sólo lo hacemos para el set de train*.

```python
import pandas as pd
import torch
import torch.nn as nn
from pysentimiento.preprocessing import preprocess_tweet
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class HateDataset(Dataset):
    def __init__(
        self,
        dataset: pd.DataFrame,
        tokenizer: str,
        max_token_len: int,
        labels: list,
        text_field="text",
        validation=False,
        fold: list = None,
    ):

        if validation and fold is not None:
            data = dataset.query(f"kfold in {fold}")
            self.tweets = data[text_field].values
            self.labels = self.create_labels(data[labels].values)

        elif fold is not None:
            data = dataset.query(f"kfold not in {fold}")
            self.tweets = data[text_field].values
            self.labels = self.create_labels(data[labels].values)

        else:
            self.tweets = dataset[text_field].values
            self.labels = None

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.max_token_len = max_token_len

    @staticmethod
    def create_labels(df: pd.DataFrame):
        return df.astype("bool").astype("int64")

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, idx):

        text = preprocess_tweet(self.tweets[idx])
        text = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,  # adds BOS and EOS tokens
            max_length=self.max_token_len,
            return_tensors="pt",  # returns torch tensors
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
        )

        return dict(
            input_ids=text["input_ids"].flatten(),
            attention_mask=text["attention_mask"].flatten(),
            labels=torch.tensor(self.labels[idx], dtype=torch.float32)
            if self.labels is not None
            else torch.tensor(0, dtype=torch.float32),
        )
```
{: title="dataset.py"}

En este archivo definimos el Pytorch Dataset el que se encargará de convertir cada texto recibido en un output compatible con transformers. Revisemos el detalle:

Primero, esta es una clase bastante más sofisticada que una clase normal, esto porque generé una lógica para transformar el texto por folds. Por lo tanto, esta misma clase se puede utilizar para los folds de entrenamiento, el fold de validación, o en caso que no quisiera ningún fold (entrenar con todo el train set).

Definí un `staticmethod` cortito para poder transformar las etiquetas, que pueden ir entre 0 y 3, a labels binarias, como 0 o 1 (en caso de que haya más de un anotador que consideró el texto como Odio). Luego quizás el método más importante el el `__getitem__` que utiliza el método `.encode_plus()` para generar el output que tendrá: `input_ids` que es el texto tokenizado codificado con el id del token correspondiente, `attention_mask` es un elemento que tiene 1s para cada token que es efectivamente una palabra y 0s para cuando no hay palabras. Esto permite que todas las secuencias sean del mismo largo a pesar de tener distinto número de palabras. `labels` contiene el label correspondiente para cada texto. En el caso del dataset de test rellené los labels con 0 (Esta fue la implementación original en la competencia cuando no tenía labels disponibles).

{% include alert tip='Por alguna razón os chicos de pysentimiento no incluyeron como parte del proceso de normalización de datos el pre-procesamiento de los tokens como los utiliza RoBERTuito. Por esto, es necesario utilizar la función `preprocess_tweet()` para poder convertir al texto a la forma que utilizará el tokenizador. Eventualmente ellos deberían crear su propio tokenizador que incluya todo este proceso de manera interna, pero no estamos ahí aún.'%}

```python
import lightning as L
import pandas as pd
from .dataset import HateDataset
from torch.utils.data import DataLoader

class HateDataModule(L.LightningDataModule):
    def __init__(
        self, 
        train: pd.DataFrame,
        labels: list,
        fold: int,
        batch_size: list,
        tokenizer: str,
        max_token_len: int = 128,
        test: pd.DataFrame = None):
        
        super().__init__()
        self.train = train
        self.test = test
        self.tokenizer = tokenizer
        self.labels = labels
        self.fold = fold
        self.batch_size = batch_size
        self.max_token_len = max_token_len
        
    def setup(self, stage=None):
        self.train_dataset = HateDataset(
            self.train,
            self.tokenizer,
            self.max_token_len,
            self.labels,
            fold=self.fold,
        )
        
        self.val_dataset = HateDataset(
            self.train, 
            self.tokenizer,
            self.max_token_len,
            self.labels,
            fold = self.fold,
            validation=True,
        )
        
        self.test_dataset = HateDataset(
            self.test, 
            self.tokenizer,
            self.max_token_len,
            self.labels,
        )
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=6,
            shuffle=True,
            drop_last=True,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=6,
            shuffle=False,
            drop_last=True,
        )
    
    def predict_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=6,
            shuffle=False,
        )
```
{: title="datamodule.py"}

En este caso se define el `LightningDataModule`. Esto permite el armado de los datasets y los dataloaders para entrenar el modelo. 
En este caso definimos los siguientes elementos:

* `.setup()`: Genera el train, validation y test set. Estos se van a ir creando fold a fold a partir del loop de entrenamiento.
* `*_dataloader()`: Generará el dataloader para cada etapa, Sólo definí para train y validación para el proceso de entrenamiento (validación es necesario para implementar el EarlyStopping) y predict para poder utilizar el `.predict()`.

```python
import lightning as L
import torch
import torch.nn as nn
from transformers import AutoModel


class RoberTuito(L.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.backbone = AutoModel.from_pretrained(config.model_name)
        self.dropout = nn.Dropout(config.dropout)

        self.classifier = nn.Linear(self.backbone.config.hidden_size, config.n_labels)
        self.criterion = nn.BCEWithLogitsLoss()
        nn.init.xavier_uniform_(self.classifier.weight)

    def forward(self, input_ids, attention_mask):
        x = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = torch.mean(x.last_hidden_state, dim=1)  # average pooling

        x = self.dropout(pooled_output)
        x = self.classifier(x)

        return x
    
    def training_step(self, batch, batch_idx):
        logits = self(batch["input_ids"], batch["attention_mask"])
        labels = batch["labels"].reshape(-1,1)
        loss = self.criterion(logits, labels)
        self.log("train_loss", loss, prog_bar=True, logger=False)
        return loss
    
    def validation_step(self, batch, batch_idx):
        logits = self(batch["input_ids"], batch["attention_mask"])
        labels = batch["labels"].reshape(-1,1)
        loss = self.criterion(logits, labels)
        self.log("val_loss", loss, prog_bar=True, logger=False)
        return loss

    def predict_step(self, batch, batch_idx):
        logits = self(batch["input_ids"], batch["attention_mask"])
        return logits
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.lr, weight_decay=self.config.w_decay)
        return optimizer
```
{: title="model.py"}

En el archivo `model.py` definimos finalmente el RoBERTuito. Robertuito se importa del HuggingFace Hub y será el backbone de nuestro modelo. La salida del Encoder BERT se promediará y saldrá como un average_pooling. No sé cuál es la razón teórica por la que se hace esto, pero es sugerido en varios tutoriales y en la documentación misma de HuggingFace como una manera de entregar más estabilidad al modelo además de hacer calzar las dimensiones. 

Pasamos por un dropout para luego conectar con un MLP que servirá para predecir. Además implementamos `xavier initialization` para el MLP (esto tiene pesos pre-determinados en vez de partir completamente random). 

Como función de costo utilizamos Cross-Entropy, pero la manera más eficiente de hacerlo en Pytorch (por temas de estabilidad numérica) es no aplicar Sigmoid a la última capa y utilizar `nn.BCEWithLogitsLoss()` como Loss Function.

El resto es parte de la implementación en Lightning: calcularemos los logits utilizando los `input_ids` y el `attention_mask`, y comparamos con los labels para obtener el loss (ojo, con las dimensiones, aplico un reshape para garantizar dimensiones compatibles).

El último archivo es `trainer.py` en el cual definimos 3 funciones necesarias para entrenar el modelo:

```python
import numpy as np
import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.metrics import f1_score

from .datamodule import HateDataModule
from .model import RoberTuito
from .utils import f1_custom


def train_model(fold: list, dm_config, model_config, training_config):

    dm = HateDataModule(**dm_config, fold=fold)
    model = RoberTuito(config=model_config)
    mc = ModelCheckpoint(
        dirpath="checkpoints",
        save_last=True,
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min",
    )

    mc.CHECKPOINT_NAME_LAST = f"best-checkpoint-latest-{fold}"
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=training_config.patience,
    )

    trainer = L.Trainer(
        deterministic=True,
        callbacks=[mc, early_stop_callback],
        max_epochs=training_config.max_epochs,
        accelerator="gpu",
        devices=1,
        fast_dev_run=training_config.fast_dev_run,
        overfit_batches=training_config.overfit_batches,
    )

    trainer.fit(model, dm)

    return trainer, model, dm


def predict(trainer, model, dm, validation=True):
    if validation:
        preds = trainer.predict(model, dm.val_dataloader())
    else:
        preds = trainer.predict(model, dm)

    results = []
    for pred in preds:
        results.extend(torch.sigmoid(pred).detach().cpu().numpy())

    return np.array(results)


def evaluate(trainer, model, dm, threshold=0.5, custom=True):
    y = dm.val_dataset.labels
    preds = np.where(predict(trainer, model, dm, validation=True) >= threshold, 1, 0)

    if custom:
        return f1_custom(y, preds)
    else:
        return f1_score(y, preds)
```
{: title="trainer.py"}

`train_model()` es una pequeña función que permite:
* El Datamodule.
* El Modelo.
* Checkpoint.
* EarlyStopping.
* Entrenamiento de un sólo fold. Cada fold será entrenado por medio de un `for loop`. 
* El lightning Trainer.

`train_model()` devolverá el trainer (necesario para predecir más adelante), el modelo entrenado, y el datamodule.

`evaluate()` permite evaluar el modelo en los distintos folds. Utilizará el criterio de `threshold = 0.5` y utilizará el `f1_score()` como métrica de evaluación (también implementé el f1_custom que fue la métrica utilizada durante la competencia, pero se requiere de más clases para poder utilizarla). 

`predict()` será una función para predecir en batches y luego aplicar `torch.sigmoid()` y luego aplanarlo.

<q>Y listo!!</q>

Bueno, hubo varias configuraciones que probé, pero lamentablemente no me dieron los resultados que esperaba en la competencia. RoBERTuito fue el primer gran approach que intenté y esperaba que fuera un modelo hiperpoderoso. Hay que ser justos, funciona bastante bien para la detección de Odio, pero falla estrepitosamente para al agregar más clases. Algunos de los resultados obtenidos:

Durante la competencia obtuve un F1_custom cercano a 0.45 para 5 clases y 0.73 para sólo Odio. Esto lo logré entrenando durante 30 epochs. Pero luego de jugar con varias configuraciones ahora que tengo acceso al test set logré otras configuraciones que dieron mejores resultados sólo para Odio:

| Epochs | LB Score | Test F1_Score |
| ------ | :------: | :-----------: |
| 1      |   0.79   |     0.826     |
| 3      |   0.61   |     0.80      |
| 5      |   0.77   |     0.78      |


{% include alert alert='La verdad no he tenido el tiempo para poder estudiar el por qué se da este fenómeno. Pero estoy intentando averiguar por qué sólo un epoch genera tan buenos resultados y luego se va degradando. No estoy seguro si esto es un comportamiento normal de los transformers, pero algunas hipótesis que tengo pueden ser que tengo demasiado parámetros para tan pocos datos (cerca de 109 millones sólo para RoBERTa) o si el Loss Function no es el apropiado para optimizar la métrica en cuestión.' %}

En las últimas semanas he estado estudiando bien en detalle el funcionamiento de los Transformers mediante el curso de HuggingFace. Voy a estar de poco compartiendo más del aprendizaje que llevo y que pueda ser de utilidad para ustedes también.

Nos vemos a la otra,

[**Alfonso**]({{ site.baseurl }}/contact/)
