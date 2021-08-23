---
permalink: /lightning/
title: "Pytorch Lightning"
subheadline: "Eliminando el Boiler Plate Code"
teaser: "Otra librería espectacular de Deep Learning"
# layout: page-fullwidth
usemathjax: true
category: quick
header: no
image:
    thumb: pl/PTL.png
tags:
- python
- ML
- dl
- tutorial
published: true
---

![picture of me]({{ site.urlimg }}pl/PTL.png){: .left .show-for-large-up .hide-for-print width="500"}
![picture of me]({{ site.urlimg }}pl/PTL.png){: .center .hide-for-large-up width="250"}

Una de las cosas que más se le critican a Pytorch es la verbosidad para poder llevar a cabo una tarea, normalmente muchas líneas de código. Lamentablemente es cierto. Pytorch no nació como una herramienta para abstraer el proceso de modelamiento. Todo lo contrario es una API de bajo nivel desarrollada por Facebook para hacer Research. Entonces, ¿eso quiere decir que siempre habrá que escribir mucho código? NO, para eso llego Pytorch Lightning. <!--more--> 

Pytorch Lightning lo presentan como una <q>especie de equivalente</q> a Keras en el ecosistema Tensorflow (aunque no es tan así). Pero, más que una abstracción del código, viene a ser una manera de organizar el código evitando el exceso de boilerplate característico en Pytorch.

Dado que Pytorch nació como una herramienta enfocada en investigación, está super orientado en detallar cada parte del proceso. Una de las ventajas que esto tiene es que es muy fácil encontrar implementaciones de lo últimos avances en Deep Learning (por ejemplo acá en [Papers with Code](https://paperswithcode.com/)). El problema es que, cuando se quiere prototipar algo rápido puede ser latero. Muchas veces, incluso cuesta recordar todos los pasos necesarios para implementar el entrenamiento de un modelo. Vamos a ver entonces cuál es la diferencia entre Pytorch Nativo y Lightning.

{% include alert warning='Pytorch, y también Pytorch Lightning se basan fuertemente en la programación orientada a objetos. Por lo que si no sabes muy bien qué son las clases en Python, este es un buen momento para comenzar a estudiarlas.'%}

Comencemos importando las siguientes librerías:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from tqdm import tqdm
import numpy as np
from torch.utils.data import random_split
from torch.utils.data import Dataset
import pytorch_lightning as pl
pl.seed_everything(123)
```

    Global seed set to 123

Un aspecto espectacular que fue pensado en `Pytorch-Lightning` es el `seed_everything()`. Esto fijará la semilla de todos los procesos aleatorios involucrados en el proceso, es decir, `Pytorch`, `numpy` y `python.random`.

## Un ejemplo sencillo utilizando MNIST

```python
from sklearn.datasets import fetch_openml
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
```
{: title="Importando la data desde Scikit-Learn."}

{% include alert info='Muchos se preguntarán por qué utilizar MNIST desde Scikit-Learn siendo que `torchvision` tiene esta data incluida para importar directamente. Bueno, la verdad es que la versión de `torchvision` viene `pre-cocinada` para llegar y usar. En la realidad rara vez ocurre eso. Por lo tanto, para acercarnos más a cómo los distintos procesos de importación de datos se ejecutan en la realidad es que lo haremos desde Scikit-Learn.'%}

En este caso, X corresponde a un `numpy array` con elementos que representan una imágen de un dígito de <var>28x28</var>.

```python
import matplotlib.pyplot as plt
plt.imshow(X[0,:].reshape(28,28))
plt.axis('off');
```
![picture of me]({{ site.urlimg }}pl/digit.png){: .center}

Para que Pytorch pueda hacer uso de los datos, estos tienen que ser una clase que hereda de la clase `Dataset`. Una clase dataset tiene que tener 3 partes:

* El constructor \_\_init\_\_() con los parámetros necesarios para instanciar la clase (normalmente definidos por uno).
* Un \_\_len\_\_() que será le encargada de contar el número de observaciones (en este caso imágenes) del Dataset. Esto es fácilmente solucionable con un `len()`.
* Un \_\_getitem\_\_() que será el método encargado de tomar los elementos uno a uno y convertirlos en tensores compatibles con `Pytorch`. Esta parte es la que hay que poner más atención ya que variará dependiendo del tipo de dato a usar.

En este caso nuestra clase se creará con 3 parámetros:
* X, que serán las imágenes en formato `numpy` array.
* y, que serán las etiquetas, también en formato `numpy`.
* transforms, que será un pipeline de transformaciones de `Albumentations`.



```python
class MNISTDataset(Dataset):
    def __init__(self, X, y, transform = None):
        self.X = X
        self.y = y.astype('int64')
        self.transform = transform
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        image = self.X[idx, :].reshape(28,28)
        label = self.y[idx]
        
        if self.transform:
            image = self.transform(image = image)['image']
        else:
            image = torch.from_numpy(image)
       
       return {'image': image,
                'label': torch.tensor(label).long()}
            
    
transform = A.Compose([
     A.Normalize(mean=(0.1307), std=(0.3081)),
    ToTensorV2(),
])
data = MNISTDataset(X,y, transform = transform)
train, validation, test = random_split(data, lengths = [50000, 10000, 10000])
```
Si nos fijamos bien \_\_getitem\_\_ es el método para obtener cada uno de los elementos por indice (idx) y en el caso de estas imágenes, hacer un `.reshape` al tamaño correspondiente (<var>28x28</var>). Como estamos utilizando transformaciones en `Albumentations`, tenemos que considerar un paso para aplicar las transformaciones. Estas transformaciones incluyen `ToTensorV2()` que es la manera en la que `Albumentations` transforma en tensores de `Pytorch`. En caso de que no hayan transformaciones nos encargaremos de transformar en tensor desde numpy.

La clase `MNISTDataset()` retornará las imágenes y los labels asociados. Notar que estos resultados pasan por `random_split` que se encargará de dividir el dataset en 3 partes train, validation y test.

{% include alert info='Hay que notar que realizar el split de la data fue posible debido a que en este caso los 3 subsets del proceso son sometidos a las mismas transformaciones. En el caso de que de no ser así y cada subset requiera de distintas transformaciones, entonces es más conveniente dividir la data al inicio y luego instanciar 3 clases MNISTDataset para cada subset con sus respectivas transformaciones.'%}

## DataLoaders

Una vez que los datos han sido separados, se deben crear los DataLoaders. Los DataLoaders son elementos en Pytorch que se encargan de ir cargando los datos de manera gradual en memoria, sea local o GPU. 

Lo más importante de esta parte es que aquí es en donde se decide el tamaño del batch que se irá cargando. Además es importante mencionar que el set de entrenamiento hay que mezclarlo para evitar que se aprendan elementos en orden (lo cual puede llevar a overfitting). También es necesario fijar `pin_memory = True` para pre-alocar espacio en la GPU (no es necesario en caso de entrenar en CPU) y `num_workers` suficiente para evitar que cuellos de botella para paralelizar todo lo que sea posible.

```python
dataloaders = {'train': torch.utils.data.DataLoader(dataset = train, 
                             batch_size = 2048,
                             shuffle = True, 
                            pin_memory = True,
                            num_workers = 10),
               'validation': torch.utils.data.DataLoader(dataset = validation, 
                             batch_size = 2048,
                             shuffle = False, 
                            pin_memory = True,
                            num_workers = 10),
               'test': torch.utils.data.DataLoader(dataset = test, 
                             batch_size = 2048,
                             shuffle = False, 
                            pin_memory = True,
                            num_workers = 10
                            )
              }
```

## Modelo en Pytorch Nativo

Una vez que se tienen los DataLoader es necesario construir la arquitectura del modelo. En este caso utilizaremos Redes Convolucionales. No voy a entrar tanto en detalle en esta parte. Pero básicamente todo modelo hereda desde `nn.Module` y sólo es necesario hacer un override del constructor (tomando en cuenta las características con las que se va a construir el modelo) y del forward que muestra cómo los datos van viajando por las distintas capas generadas en el constructor.

**NOTA**: `CNN_BLOCK` es una función que condensa una red Convolucional estándar. Los parámetros denotan lo siguiente:
* **c_in** y **c_out**: Son los canales de entrada y de salida de la red. Normalmente en la primera capa se utiliza 1 canal de entrada para imágenes blanco y negro y 3 para imágenes RGB. Aunque recientemente descubrí que imágenes satelitales pueden tener más de 3 canales.
* **k** es el tamaño del filtro o kernel.
* **p** es el padding.
* **s** es el stride.
* **pk** es el tamaño del pooling.
* **ps** es el stride del pooling.


```python
def CNN_block(c_in, c_out, k = 3, p = 1, s = 1, pk = 2, ps = 2):
    return nn.Sequential(
            nn.Conv2d(in_channels = c_in, out_channels= c_out, kernel_size = k, padding = p, stride = s),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = pk, stride = ps)
    )

class CNN(nn.Module):
    def __init__(self, n_channels = 1, n_outputs = 10):
        super().__init__()
        self.conv1 = CNN_block(n_channels, 64)
        self.conv2 = CNN_block(64, 128)
        self.fc = nn.Linear(128*7*7, n_outputs) # filtros x tamaño
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

model = CNN()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
```
Además al instanciar el modelo, se debe traspasar a la GPU en caso de tener, y se deben definir la función de Costo, que normalmente se le llama `criterion`, y el optimizador, que en este caso se está utilizando Adam con un learning rate de <var>1e-3</var>.

Luego para entrenar el modelo viene la parte más `latera`, `fome`, `tediosa`, `aparatosa` y `propensa error`: <mark>Los loop de entrenamiento</mark>. Dependiendo del modelo hay que fijarse en muchos aspectos:

1. Agregar Loop de Epochs.
2. Agregar Loop de DataLoaders.
3. Pasar los datos a la GPU.
4. Cambiar el modo del modelo a `train()`.
5. Reiniciar los gradientes.
6. Hacer un Forward Pass.
7. Calcular el loss.
8. Calcular el backpropagation con `.backward`.
9. Actualizar los pesos con `step`.
10. Repetir los pasos `2`, `3`, `4`, `6` y `7` pero aplicado a Test.
11. Definir métricas para mostrar en el entrenamiento. 
12. Agregar alguna barrita de progreso para hacer la espera menos tediosa (normalmente con `tqdm`).

Todos estos pasos se pueden ver reflejados en la siguiente función:

```python
def fit(model, dataloader, epochs = 5):
    for epoch in range(1,epochs+1):
        model.train()
        train_loss, train_acc = [],[]
        bar = tqdm(dataloader['train'],position=0, leave=True)
        for batch in bar:
            X,y = batch['image'].to(device), batch['label'].to(device)
            optimizer.zero_grad()
            y_hat = model(X)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            acc = (y == torch.argmax(y_hat, axis = 1)).sum().item()/len(y) ## accuracy del batch
            train_acc.append(acc)
            bar.set_description(f'Loss: {np.mean(train_loss):.3f}, Accuracy: {np.mean(train_acc):.3f}')
        bar = tqdm(dataloader['validation'],position=0, leave=True)
        val_loss, val_acc = [],[]
        model.eval()
        for X,y in bar:
            X,y = batch['image'].to(device), batch['label'].to(device)
            y_hat = model(X)
            loss = criterion(y_hat, y)
            val_loss.append(loss.item())
            acc = (y == torch.argmax(y_hat, axis = 1)).sum().item()/len(y) ## accuracy del batch
            val_acc.append(acc)
            bar.set_description(f'Loss: {np.mean(train_loss):.3f}, Accuracy: {np.mean(train_acc):.3f}')
        print(f'Epoch {epoch}/{epochs}, Training Loss: {np.mean(train_loss):.3f}, Validation Loss: {np.mean(val_loss):.3f}, Accuracy {np.mean(train_acc):.3f}, Validation Accuracy: {np.mean(val_acc):.3f}')
```
{: title="Definición Loop de entrenamiento"}

```python
fit(model, dataloaders)
```
{: title="Ejecución del Entrenamiento"}


    Loss: 0.682, Accuracy: 0.807: 100%|██████████| 25/25 [00:05<00:00,  4.73it/s]
    Loss: 0.682, Accuracy: 0.807: 100%|██████████| 5/5 [00:00<00:00,  7.10it/s]
      0%|          | 0/25 [00:00<?, ?it/s]

    Epoch 1/5, Training Loss: 0.682, Validation Loss: 0.228, Accuracy 0.807, Validation Accuracy: 0.930


    Loss: 0.175, Accuracy: 0.949: 100%|██████████| 25/25 [00:05<00:00,  4.73it/s]
    Loss: 0.175, Accuracy: 0.949: 100%|██████████| 5/5 [00:00<00:00,  7.35it/s]
      0%|          | 0/25 [00:00<?, ?it/s]

    Epoch 2/5, Training Loss: 0.175, Validation Loss: 0.112, Accuracy 0.949, Validation Accuracy: 0.973


    Loss: 0.098, Accuracy: 0.972: 100%|██████████| 25/25 [00:05<00:00,  4.72it/s]
    Loss: 0.098, Accuracy: 0.972: 100%|██████████| 5/5 [00:00<00:00,  6.83it/s]
      0%|          | 0/25 [00:00<?, ?it/s]

    Epoch 3/5, Training Loss: 0.098, Validation Loss: 0.059, Accuracy 0.972, Validation Accuracy: 0.985


    Loss: 0.069, Accuracy: 0.980: 100%|██████████| 25/25 [00:05<00:00,  4.78it/s]
    Loss: 0.069, Accuracy: 0.980: 100%|██████████| 5/5 [00:00<00:00,  7.10it/s]
      0%|          | 0/25 [00:00<?, ?it/s]

    Epoch 4/5, Training Loss: 0.069, Validation Loss: 0.070, Accuracy 0.980, Validation Accuracy: 0.982


    Loss: 0.055, Accuracy: 0.984: 100%|██████████| 25/25 [00:05<00:00,  4.96it/s]
    Loss: 0.055, Accuracy: 0.984: 100%|██████████| 5/5 [00:00<00:00,  7.68it/s]

    Epoch 5/5, Training Loss: 0.055, Validation Loss: 0.035, Accuracy 0.984, Validation Accuracy: 0.987


El problema con esto es que si quiero calcular el Accuracy con otro dataset no visto o si quiero hacer inferencia tengo que crear más loops y realmente el proceso se hace muy pero muy latero. Puedes ver ejemplos de esto en otros de mis tutoriales: [Perro vs Gatos]({{ site.baseurl }}/dog-cats) o en [Detección de Emociones]({{ site.baseurl }}/emotions).


# Pytorch Lightning Way 

El desarrollo de Pytorch Lightning fue iniciado por William Falcon (aunque hoy hay mucha gente contribuyendo al proyecto), un Investigador en Deep Learning que encontraba que si bien Pytorch era de gran utilidad tenía estos inconvenientes de código muy complicado de escribir, pero por sobre todo, propenso a error.
El ecosistema de Pytorch Lightning está creciendo día a día con librerías como `Flash`, que es una interfaz de mucho más alto nivel para Pytorch para resolver tareas específicas, y `torchmetrics` que permite calcular las métricas más relevantes para Redes Neuronales, sin tener que hacer esos loops y operaciones ultracomplicadas que se pueden ver en mis tutoriales anteriores.

{% include alert alert='Uno pudiera decir, no necesito estas métricas. Todo esto ya está implementado en Scikit-Learn. Y es cierto. Pero hay al menos dos inconvenientes: uno es que las métricas de Scikit-Learn no aceptan tensores (por lo que habŕia que transformar todo a Numpy, lo cual puede ser bien costoso) pero por sobre todo es que las métricas de Scikit no corren en GPU por lo que van a ser mucho más lentas.'%}


```python
import pytorch_lightning as pl
import torchmetrics
```

Pytorch Lightning define dos clases adicionales de las cuales se puede heredar: el `LightningDataModule`, que será una manera de organizar la data y el `LightningModule` que permitirá organizar el modelo.

En este caso para crear un Dataset crearemos una clase pero que herede desde el `LightningDataModule`, Para esto es necesario definir:
* Un constructor, con los parámetros que se consideren necesarios para construir el modelo.
* `setup()` que será el método encargado de generar las clases Dataset.
* `*_dataloaders()` que serán los encargados de cargar los datos.
    * `train_dataloader()` que se utilizará exclusivamente para para cargar los datos de entrenamiento.
    * `val_dataloader()` que se utilizará para medir la performance al entrenar y chequear en tiempo real si hay overfitting.
    * `test_dataloader()` que se utilizará para medir la performance final del modelo. Estos normalmente son datos no vistos anteriormente por el modelo.
    * `predict_dataloader()` es opcional y se utilizará para ejecutar un método predict. Sí leíste bien, Pytorch Lightning tuvo la brillante idea de implementar un `.predict()` e incluso un `.fit()`. Sigue leyendo para entender cómo se usa.

{% include alert warning='Todos los nombres de los métodos junto con sus parámetros son estándar y no se pueden cambiar. Ya que de hacerlo Pytorch Lightning arrojará errores como loco. Y no quieres eso.'%}

```python
class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, transform, batch_size = 2048):
        super().__init__()
        self.batch_size = batch_size
        self.transform = transform
    def setup(self, stage = None):
        data = MNISTDataset(X,y, transform = self.transform)
        self.train, self.validation, self.test = random_split(data, lengths = [50000, 10000, 10000])
    def train_dataloader(self):
        return torch.utils.data.DataLoader(dataset = self.train, batch_size = self.batch_size, shuffle = True, pin_memory = True, num_workers = 10)
    def val_dataloader(self):
        return torch.utils.data.DataLoader(dataset = self.validation, batch_size = self.batch_size, shuffle = False, pin_memory = True, num_workers = 10)
    def test_dataloader(self):
        return torch.utils.data.DataLoader(dataset = self.test, batch_size = self.batch_size, shuffle = False, pin_memory = True, num_workers = 10)
    def predict_dataloader(self):
        return torch.utils.data.DataLoader(dataset = self.test, shuffle = False, pin_memory = True, num_workers = 10)
```
{: title="Lightning Data Module"}


Por otro lado, el modelo se creará dentro de una clase que heredará del `LightningModule`, y contendrá todo lo relacionado al modelo más su entrenamiento. Los métodos que deben definirse son:
* El constructor con los parámetros necesarios para la construcción del modelo.
*  `forward()` que contendrá el proceso en el cómo los datos viajarán a través de la red. Hasta acá es igual a lo que teníamos con Pytorch Nativo, ahora parte lo nuevo.
* `configure_optimizers()` permitirá definir el optimizador a usar.
* `*_step()` que incluirá todo el proceso de entrenamiento, validación y/o testeo que se utiliza en los loops. El gran detalle acá es que sólo se colocan los pasos esenciales y no es necesario agregar:
  * Loops para epochs.
  * Loops para dataloaders.
  * Traspaso de los datos a la GPU.
  * Modo del modelo.
  * `optimizer.zero_grad()`.
  * `loss.backward()`.
  * `optimizer.step()`.

Además convenientemente tiene un `.log()` que permitirá mostrar en el proceso de entrenamiento la información que se considere pertinente. Incluso se pueden agregar barras de progreso.
Finalmente, existe de manera opcional un `predict_step` que definirá cómo se hace inferencia con el anhelado método `.predict()`.

```python
def CNN_block(c_in, c_out, k = 3, p = 1, s = 1, pk = 2, ps = 2):
    return nn.Sequential(
            nn.Conv2d(in_channels = c_in, out_channels= c_out, kernel_size = k, padding = p, stride = s),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = pk, stride = ps)
    )

class CNN(pl.LightningModule):
    
    def __init__(self, n_channels=1, n_outputs=10):
        super().__init__()
        self.conv1 = CNN_block(n_channels, 64)
        self.conv2 = CNN_block(64, 128)
        self.fc = torch.nn.Linear(128*7*7, n_outputs)
        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()
        

    def forward(self, x):
        x = self.conv1(x)
        #print(x)
        x = self.conv2(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch['image'], batch['label']
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.train_acc(y_hat, y)
        self.log('train_loss', loss)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=False)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch['image'], batch['label']
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.valid_acc(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.valid_acc, on_step=True, on_epoch=True)
    
    def test_step(self, batch, batch_idx):
        x, y = batch['image'], batch['label']
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.test_acc(y_hat, y)
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', self.test_acc, on_step=True, on_epoch=True)
        
    def predict_step(self, batch, batch_idx, dataloader_idx):
        x = batch['image']
        y_hat = self(x) 
        return torch.argmax(y_hat)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
```
{: title="Lightning Module"}


## Entrenamiento del Modelo

Cuando se utiliza Pytorch Lighntning, al entrenar el modelo no tendremos que volver a crear esos training loops gigantes, en cambio debemos:

* Instanciar el Modelo.
* Instanciar un ModelCheckpoint (esto es opcional, pero de no hacerlo hay un warning bien molesto).
* Instanciar un `pl.Trainer` que permitirá definir el número de epochs, el número de gpus a utilizar (dependiendo de la disponibilidad).
* La barra de progreso (yei!!)
* Y el ansiado `.fit` en el cual se deben agregar el modelo (`LighningModule`) y el DataModule (`LightningDataModule`).
* Y ya, comienza el entrenamiento!!

```python
%%time
from pytorch_lightning.callbacks import ModelCheckpoint

model = CNN() # Instancia del Modelo
dm = MNISTDataModule(transform = transform) # Instancia del Data Module
mc = ModelCheckpoint(monitor="val_loss") # Instancia del Model Checkpoint
# Instancia del Trainer
trainer = pl.Trainer(max_epochs = 5, 
                     gpus = 1,
                     callbacks = [mc],
                     progress_bar_refresh_rate=20)
# Fit del Modelo
trainer.fit(model, dm)
```

    GPU available: True, used: True
    TPU available: False, using: 0 TPU cores
    LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
    
      | Name      | Type       | Params
    -----------------------------------------
    0 | conv1     | Sequential | 640   
    1 | conv2     | Sequential | 73.9 K
    2 | fc        | Linear     | 62.7 K
    3 | train_acc | Accuracy   | 0     
    4 | valid_acc | Accuracy   | 0     
    5 | test_acc  | Accuracy   | 0     
    -----------------------------------------
    137 K     Trainable params
    0         Non-trainable params
    137 K     Total params
    0.549     Total estimated model params size (MB)

    Global seed set to 123


    Epoch 4: 100%|██████████| 30/30 [00:03<00:00,  8.91it/s, loss=0.0574, v_num=23, val_loss=0.0546]
    CPU times: user 12 s, sys: 2.3 s, total: 14.3 s
    Wall time: 17 s

{% include alert success='Hemos podido entrenar el modelo de manera mucho más sencilla. Si bien el código se organiza de mejor manera, no se genera una abstracción al punto de no entender nada de lo que sucede en el entrenamiento. Esa es la gracia de Pytorch Lightnining. Organizar, optimizar, pero no esconder.'%}

Adicionalmente el trainer tiene algunos elementos que permitirán chequear el desempeño en los datasets de validación o de test:

```python
trainer.validate()
```
{: title="Resultados de Validación"}


    DATALOADER:0 VALIDATE RESULTS
    {'test_acc': 0.98580002784729,
    'test_acc_epoch': 0.98580002784729,
    'test_loss': 0.049540963023900986,
    'val_acc': 0.9836999773979187,
    'val_acc_epoch': 0.9836999773979187,
    'val_loss': 0.05461977422237396}

```python
trainer.test()
```
{: title="Resultados en el Test Set"}

    DATALOADER:0 TEST RESULTS
    {'test_acc': 0.98580002784729,
    'test_acc_epoch': 0.98580002784729,
    'test_loss': 0.049540963023900986}


Finalmente si es que se define un `predict_step` y un `predict_dataloader` se puede utilizar el ansiado `.predict()` ingresando el modelo entrenado y el datamodule:

```python
results = trainer.predict(model, datamodule = dm)
```

    LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]


    Predicting: 100%|██████████| 10000/10000 [00:08<00:00, 1133.43it/s]

Debido a que en el `predict_dataloader()` se colocaron los mismos datos de test es que el resultados de las predicciones es de 10000 elementos:

```python
len(results)
```

{% include alert success='Adicionalmente se puede utilizar con un DataLoader definido o como una inferencia normal en Pytorch. Esto entrega más flexibilidd al momento de poner el modelo en Produccion.'%}

## Revisando los resultados

Si queremos mirar los `n` primeros resultados podemos hacer algo así:

```python
n = 20
results[:20]
```
    [tensor(2, device='cuda:0'),
    tensor(7, device='cuda:0'),
    tensor(4, device='cuda:0'),
    tensor(1, device='cuda:0'),
    tensor(5, device='cuda:0'),
    tensor(2, device='cuda:0'),
    tensor(7, device='cuda:0'),
    tensor(0, device='cuda:0'),
    tensor(2, device='cuda:0'),
    tensor(5, device='cuda:0'),
    tensor(9, device='cuda:0'),
    tensor(8, device='cuda:0'),
    tensor(3, device='cuda:0'),
    tensor(7, device='cuda:0'),
    tensor(0, device='cuda:0'),
    tensor(0, device='cuda:0'),
    tensor(8, device='cuda:0'),
    tensor(1, device='cuda:0'),
    tensor(7, device='cuda:0'),
    tensor(5, device='cuda:0')]

Si nos fijamos, el resultado es una lista con 20 tensores resultantes que viven en la GPU. Por lo tanto, si queremos ver la respuesta sin esas indicaciones de tensores podemos traerlos a la CPU y verlos de la siguiente forma:

```python
[value.cpu().item() for value in results[:n]]
```
    [2, 7, 4, 1, 5, 2, 7, 0, 2, 5, 9, 8, 3, 7, 0, 0, 8, 1, 7, 5]

Finalmente si queremos verificar que estas predicciones son correctas, y las queremos comparar con la imagen:

```python
for value, (elemento, pred) in enumerate(zip(dm.predict_dataloader(), results[:n])):
    im = elemento['image'].squeeze(0).numpy().transpose(1,2,0)
    plt.imshow(im)
    plt.title(f'La predicción fue {pred}')
    plt.axis('off')
    plt.show()
    if value == n:
        break
```
![png]({{ site.urlimg }}pl/PL_tutorial/output_23_0.png){: .center}
![png]({{ site.urlimg }}pl/PL_tutorial/output_23_1.png){: .center}
![png]({{ site.urlimg }}pl/PL_tutorial/output_23_2.png){: .center}
![png]({{ site.urlimg }}pl/PL_tutorial/output_23_3.png){: .center}
![png]({{ site.urlimg }}pl/PL_tutorial/output_23_4.png){: .center}
![png]({{ site.urlimg }}pl/PL_tutorial/output_23_5.png){: .center}
![png]({{ site.urlimg }}pl/PL_tutorial/output_23_6.png){: .center}
![png]({{ site.urlimg }}pl/PL_tutorial/output_23_7.png){: .center}
![png]({{ site.urlimg }}pl/PL_tutorial/output_23_8.png){: .center}
![png]({{ site.urlimg }}pl/PL_tutorial/output_23_9.png){: .center}
![png]({{ site.urlimg }}pl/PL_tutorial/output_23_10.png){: .center}
![png]({{ site.urlimg }}pl/PL_tutorial/output_23_11.png){: .center}
![png]({{ site.urlimg }}pl/PL_tutorial/output_23_12.png){: .center}
![png]({{ site.urlimg }}pl/PL_tutorial/output_23_13.png){: .center}
![png]({{ site.urlimg }}pl/PL_tutorial/output_23_14.png){: .center}
![png]({{ site.urlimg }}pl/PL_tutorial/output_23_15.png){: .center}
![png]({{ site.urlimg }}pl/PL_tutorial/output_23_16.png){: .center}
![png]({{ site.urlimg }}pl/PL_tutorial/output_23_17.png){: .center}
![png]({{ site.urlimg }}pl/PL_tutorial/output_23_18.png){: .center}
![png]({{ site.urlimg }}pl/PL_tutorial/output_23_19.png){: .center}

Eso fue este pequeño tutorial explicando la organización básica en `Pytorch-Lightning`. Obviamente, esto es sólo el principio, y Lightning ofrece varias funcionalidades extras como por ejemplo el almacenamiento de Hiperparámetros, integración con Loggers como Weights & Biases o Tensorboard, Early Stopping, LR Scheduler, batch size finder, integración con Hydra además de herramientas para correr de manera muy sencilla en multiples GPUs o incluso TPUs.

Espero les haya gustado, nos vemos a la próxima.

[**Alfonso**]({{ site.baseurl }}/contact/)

