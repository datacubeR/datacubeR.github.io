```python
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.ticker import MaxNLocator

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pylab import rcParams

from multiprocessing import cpu_count
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics.functional import accuracy
from sklearn.metrics import classification_report, confusion_matrix

pl.__version__

```




    '1.3.8'




```python
#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'
#sns.set(style = 'whitegrid', palette = 'muted', font_scale = 1.2)
#HAPPY_COLORS_PALETTE = ['#01BEFE', '#FFDD00','#FF7D00','#FF006D','#ADFF02','#8F00FF']
#sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
#rcParams['figure.figsize'] = 6,4

```


```python
pl.seed_everything(42)
```

    Global seed set to 42





    42




```python
X_train = pd.read_csv('career-con-2019/X_train.csv')
y_train = pd.read_csv('career-con-2019/y_train.csv')
X_train.head()
# TODO: READ DESCRIPTIONS OF THE DATA IN KAGGLE...
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
      <th>row_id</th>
      <th>series_id</th>
      <th>measurement_number</th>
      <th>orientation_X</th>
      <th>orientation_Y</th>
      <th>orientation_Z</th>
      <th>orientation_W</th>
      <th>angular_velocity_X</th>
      <th>angular_velocity_Y</th>
      <th>angular_velocity_Z</th>
      <th>linear_acceleration_X</th>
      <th>linear_acceleration_Y</th>
      <th>linear_acceleration_Z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0_0</td>
      <td>0</td>
      <td>0</td>
      <td>-0.75853</td>
      <td>-0.63435</td>
      <td>-0.10488</td>
      <td>-0.10597</td>
      <td>0.107650</td>
      <td>0.017561</td>
      <td>0.000767</td>
      <td>-0.74857</td>
      <td>2.1030</td>
      <td>-9.7532</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0_1</td>
      <td>0</td>
      <td>1</td>
      <td>-0.75853</td>
      <td>-0.63434</td>
      <td>-0.10490</td>
      <td>-0.10600</td>
      <td>0.067851</td>
      <td>0.029939</td>
      <td>0.003386</td>
      <td>0.33995</td>
      <td>1.5064</td>
      <td>-9.4128</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0_2</td>
      <td>0</td>
      <td>2</td>
      <td>-0.75853</td>
      <td>-0.63435</td>
      <td>-0.10492</td>
      <td>-0.10597</td>
      <td>0.007275</td>
      <td>0.028934</td>
      <td>-0.005978</td>
      <td>-0.26429</td>
      <td>1.5922</td>
      <td>-8.7267</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0_3</td>
      <td>0</td>
      <td>3</td>
      <td>-0.75852</td>
      <td>-0.63436</td>
      <td>-0.10495</td>
      <td>-0.10597</td>
      <td>-0.013053</td>
      <td>0.019448</td>
      <td>-0.008974</td>
      <td>0.42684</td>
      <td>1.0993</td>
      <td>-10.0960</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0_4</td>
      <td>0</td>
      <td>4</td>
      <td>-0.75852</td>
      <td>-0.63435</td>
      <td>-0.10495</td>
      <td>-0.10596</td>
      <td>0.005135</td>
      <td>0.007652</td>
      <td>0.005245</td>
      <td>-0.50969</td>
      <td>1.4689</td>
      <td>-10.4410</td>
    </tr>
  </tbody>
</table>
</div>




```python
y_train.head()
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
      <th>series_id</th>
      <th>group_id</th>
      <th>surface</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>13</td>
      <td>fine_concrete</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>31</td>
      <td>concrete</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>20</td>
      <td>concrete</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>31</td>
      <td>concrete</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>22</td>
      <td>soft_tiles</td>
    </tr>
  </tbody>
</table>
</div>




```python
def plot_ts(idx):
    return X_train.query(f'series_id == {idx}').iloc[:,3:].plot(figsize = (20,10), title = y_train.iloc[idx]['surface']);
```


```python
plot_ts(0);
```


    
![png](output_6_0.png)
    



```python
plot_ts(1);
```


    
![png](output_7_0.png)
    



```python
plot_ts(4);
```


    
![png](output_8_0.png)
    



```python
plot_ts(6);
```


    
![png](output_9_0.png)
    



```python
plot_ts(8);
```


    
![png](output_10_0.png)
    


## Preprocessing


```python
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(y_train.surface)
encoded_labels[:5]
```




    array([2, 1, 1, 1, 6])




```python
label_encoder.classes_
```




    array(['carpet', 'concrete', 'fine_concrete', 'hard_tiles',
           'hard_tiles_large_space', 'soft_pvc', 'soft_tiles', 'tiled',
           'wood'], dtype=object)




```python
y_train['label'] = encoded_labels
y_train.head()
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
      <th>series_id</th>
      <th>group_id</th>
      <th>surface</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>13</td>
      <td>fine_concrete</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>31</td>
      <td>concrete</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>20</td>
      <td>concrete</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>31</td>
      <td>concrete</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>22</td>
      <td>soft_tiles</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>




```python
X_train
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
      <th>row_id</th>
      <th>series_id</th>
      <th>measurement_number</th>
      <th>orientation_X</th>
      <th>orientation_Y</th>
      <th>orientation_Z</th>
      <th>orientation_W</th>
      <th>angular_velocity_X</th>
      <th>angular_velocity_Y</th>
      <th>angular_velocity_Z</th>
      <th>linear_acceleration_X</th>
      <th>linear_acceleration_Y</th>
      <th>linear_acceleration_Z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0_0</td>
      <td>0</td>
      <td>0</td>
      <td>-0.75853</td>
      <td>-0.63435</td>
      <td>-0.104880</td>
      <td>-0.105970</td>
      <td>0.107650</td>
      <td>0.017561</td>
      <td>0.000767</td>
      <td>-0.74857</td>
      <td>2.1030</td>
      <td>-9.7532</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0_1</td>
      <td>0</td>
      <td>1</td>
      <td>-0.75853</td>
      <td>-0.63434</td>
      <td>-0.104900</td>
      <td>-0.106000</td>
      <td>0.067851</td>
      <td>0.029939</td>
      <td>0.003386</td>
      <td>0.33995</td>
      <td>1.5064</td>
      <td>-9.4128</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0_2</td>
      <td>0</td>
      <td>2</td>
      <td>-0.75853</td>
      <td>-0.63435</td>
      <td>-0.104920</td>
      <td>-0.105970</td>
      <td>0.007275</td>
      <td>0.028934</td>
      <td>-0.005978</td>
      <td>-0.26429</td>
      <td>1.5922</td>
      <td>-8.7267</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0_3</td>
      <td>0</td>
      <td>3</td>
      <td>-0.75852</td>
      <td>-0.63436</td>
      <td>-0.104950</td>
      <td>-0.105970</td>
      <td>-0.013053</td>
      <td>0.019448</td>
      <td>-0.008974</td>
      <td>0.42684</td>
      <td>1.0993</td>
      <td>-10.0960</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0_4</td>
      <td>0</td>
      <td>4</td>
      <td>-0.75852</td>
      <td>-0.63435</td>
      <td>-0.104950</td>
      <td>-0.105960</td>
      <td>0.005135</td>
      <td>0.007652</td>
      <td>0.005245</td>
      <td>-0.50969</td>
      <td>1.4689</td>
      <td>-10.4410</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>487675</th>
      <td>3809_123</td>
      <td>3809</td>
      <td>123</td>
      <td>0.62871</td>
      <td>-0.76878</td>
      <td>-0.084391</td>
      <td>0.081093</td>
      <td>0.003167</td>
      <td>0.093760</td>
      <td>-0.142740</td>
      <td>3.27180</td>
      <td>2.0115</td>
      <td>-9.0063</td>
    </tr>
    <tr>
      <th>487676</th>
      <td>3809_124</td>
      <td>3809</td>
      <td>124</td>
      <td>0.62884</td>
      <td>-0.76868</td>
      <td>-0.084365</td>
      <td>0.081099</td>
      <td>0.014994</td>
      <td>0.032637</td>
      <td>-0.132380</td>
      <td>4.42750</td>
      <td>3.0696</td>
      <td>-8.1257</td>
    </tr>
    <tr>
      <th>487677</th>
      <td>3809_125</td>
      <td>3809</td>
      <td>125</td>
      <td>0.62891</td>
      <td>-0.76861</td>
      <td>-0.084345</td>
      <td>0.081178</td>
      <td>-0.031184</td>
      <td>-0.003961</td>
      <td>-0.138940</td>
      <td>2.70480</td>
      <td>4.2622</td>
      <td>-8.1443</td>
    </tr>
    <tr>
      <th>487678</th>
      <td>3809_126</td>
      <td>3809</td>
      <td>126</td>
      <td>0.62903</td>
      <td>-0.76850</td>
      <td>-0.084414</td>
      <td>0.081231</td>
      <td>-0.069153</td>
      <td>0.013229</td>
      <td>-0.130210</td>
      <td>2.54100</td>
      <td>4.7130</td>
      <td>-9.4435</td>
    </tr>
    <tr>
      <th>487679</th>
      <td>3809_127</td>
      <td>3809</td>
      <td>127</td>
      <td>0.62915</td>
      <td>-0.76839</td>
      <td>-0.084441</td>
      <td>0.081284</td>
      <td>-0.042769</td>
      <td>0.034049</td>
      <td>-0.125800</td>
      <td>0.82391</td>
      <td>4.2751</td>
      <td>-10.4980</td>
    </tr>
  </tbody>
</table>
<p>487680 rows × 13 columns</p>
</div>




```python
FEATURE_COLUMNS = X_train.columns.tolist()[3:]
FEATURE_COLUMNS
```




    ['orientation_X',
     'orientation_Y',
     'orientation_Z',
     'orientation_W',
     'angular_velocity_X',
     'angular_velocity_Y',
     'angular_velocity_Z',
     'linear_acceleration_X',
     'linear_acceleration_Y',
     'linear_acceleration_Z']




```python
(X_train.series_id.value_counts() == 128).sum() == len(y_train) # all the sequences have a label
```




    True




```python
sequences = []
for series_id, group in X_train.groupby('series_id'):
    sequence_features = group[FEATURE_COLUMNS]
    label = y_train.query(f'series_id == {series_id}').iloc[0].label
    sequences.append((sequence_features, label))
```


```python
train_sequences, test_sequences = train_test_split(sequences, test_size = 0.2)
len(train_sequences), len(test_sequences)
```




    (3048, 762)



# Create Dataset


```python
class SurfaceDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences
    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, idx):
        sequence, label = self.sequences[idx]
        
        return dict(
            sequence = torch.tensor(sequence.to_numpy(), dtype = torch.float32),
            label = torch.tensor(label).long()
        )
```


```python
class SurfaceDataModule(pl.LightningDataModule):
    def __init__(self, train_sequences, test_sequences, batch_size):
        super().__init__()
        self.train_sequences = train_sequences
        self.test_sequences = test_sequences
        self.batch_size = batch_size
        
    def setup(self, stage = None):
        self.train_dataset = SurfaceDataset(self.train_sequences)
        self.test_dataset = SurfaceDataset(self.test_sequences)
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size = self.batch_size, 
            pin_memory = True, 
            num_workers = cpu_count(), 
            shuffle = True)
    def val_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size = self.batch_size, 
            pin_memory = True, 
            num_workers = cpu_count(), 
            shuffle = False)
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size = self.batch_size, 
            pin_memory = True, 
            num_workers = cpu_count(), 
            shuffle = False)
```


```python
N_EPOCHS = 250
BATCH_SIZE = 64

data_module = SurfaceDataModule(train_sequences, test_sequences, BATCH_SIZE)
data_module.setup()
```

# Model


```python
class SequenceModel(nn.Module):
    def __init__(self, n_features, n_classes, n_hidden = 256, n_layers = 3):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size = n_features,
            hidden_size = n_hidden, # number of neurons for each layer...
            num_layers = n_layers,
            batch_first = True,
            dropout = 0.75
        )
        self.classifier = nn.Linear(n_hidden, n_classes)
    def forward(self, x):
        #self.lstm.flatten_parameters() # it seems I need to do this for multi GPU...
        _, (hidden, _) = self.lstm(x)
        out = hidden[-1] # get the last cell state
        return self.classifier(out)
```


```python
class SurfacePredictor(pl.LightningModule):
    def __init__(self, n_features: int, n_classes: int):
        super().__init__()
        self.model = SequenceModel(n_features, n_classes)
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, x, labels = None):
        output = self.model(x)
        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels)
        return loss, output
    
    def training_step(self, batch, batch_idx):
        sequences = batch['sequence']
        labels = batch['label']
        loss, outputs = self(sequences, labels)
        predictions = torch.argmax(outputs, dim = 1)
        step_accuracy = accuracy(predictions, labels)
        self.log('train_loss', loss, prog_bar = True, logger = True)
        self.log('train_accuracy', step_accuracy, prog_bar = True, logger = False)
        return {'loss': loss, 'accuracy': step_accuracy}
    
    def validation_step(self, batch, batch_idx):
        sequences = batch['sequence']
        labels = batch['label']
        loss, outputs = self(sequences, labels)
        predictions = torch.argmax(outputs, dim = 1)
        step_accuracy = accuracy(predictions, labels)
        self.log('val_loss', loss, prog_bar = True, logger = True)
        self.log('val_accuracy', step_accuracy, prog_bar = True, logger = False)
        return {'loss': loss, 'accuracy': step_accuracy}
    
    def test_step(self, batch, batch_idx):
        sequences = batch['sequence']
        labels = batch['label']
        loss, outputs = self(sequences, labels)
        predictions = torch.argmax(outputs, dim = 1)
        step_accuracy = accuracy(predictions, labels)
        self.log('test_loss', loss, prog_bar = True, logger = True)
        self.log('test_accuracy', step_accuracy, prog_bar = True, logger = False)
        return {'loss': loss, 'accuracy': step_accuracy}
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr = 0.0001)
    
    
```


```python
model = SurfacePredictor(n_features=len(FEATURE_COLUMNS), n_classes = len(label_encoder.classes_))
```


```python
checkpoint_callback = ModelCheckpoint(
    dirpath = 'checkpoints',
    filename = 'best-checkpoint',
    save_top_k = 1,
    verbose = True,
    monitor = 'val_loss', 
    mode = 'min'
    )

trainer = pl.Trainer(callbacks = [checkpoint_callback], 
                    max_epochs = N_EPOCHS,
                    gpus = 1, 
                    progress_bar_refresh_rate = 30,
                    deterministic=True,
                    fast_dev_run=False)
```

    /home/alfonso/miniconda3/envs/dl/lib/python3.7/site-packages/pytorch_lightning/callbacks/model_checkpoint.py:360: UserWarning: Checkpoint directory checkpoints exists and is not empty.
      rank_zero_warn(f"Checkpoint directory {dirpath} exists and is not empty.")
    GPU available: True, used: True
    TPU available: False, using: 0 TPU cores



```python
#trainer.fit(model, data_module)
```


```python
trained_model = SurfacePredictor.load_from_checkpoint(
    "checkpoints/best-checkpoint-v1.ckpt",
    n_features = len(FEATURE_COLUMNS),
    n_classes = len(label_encoder.classes_)
)
```


```python
test_dataset = SurfaceDataset(test_sequences)

predictions = []
labels = []

for item in tqdm(test_dataset):
    sequence = item['sequence']
    label = item['label']
    
    _, output = trained_model(sequence.unsqueeze(dim = 0))
    prediction = torch.argmax(output, dim = 1)
    predictions.append(prediction.item())
    labels.append(label.item())
```

    100%|██████████| 762/762 [00:20<00:00, 37.79it/s]



```python
print(
    classification_report(labels, predictions, target_names = label_encoder.classes_, digits = 4)
)
```

                            precision    recall  f1-score   support
    
                    carpet     0.7568    0.5957    0.6667        47
                  concrete     0.7845    0.8256    0.8045       172
             fine_concrete     0.6935    0.6615    0.6772        65
                hard_tiles     0.8000    0.8000    0.8000         5
    hard_tiles_large_space     0.8793    0.8095    0.8430        63
                  soft_pvc     0.8511    0.8511    0.8511       141
                soft_tiles     0.9167    0.8730    0.8943        63
                     tiled     0.7640    0.7640    0.7640        89
                      wood     0.6899    0.7607    0.7236       117
    
                  accuracy                         0.7874       762
                 macro avg     0.7929    0.7712    0.7805       762
              weighted avg     0.7893    0.7874    0.7872       762
    



```python
print(
    classification_report(labels, predictions, target_names = label_encoder.classes_, digits = 4)
)
```

                            precision    recall  f1-score   support
    
                    carpet     0.7561    0.6596    0.7045        47
                  concrete     0.7766    0.8488    0.8111       172
             fine_concrete     0.7931    0.7077    0.7480        65
                hard_tiles     0.8000    0.8000    0.8000         5
    hard_tiles_large_space     0.9444    0.8095    0.8718        63
                  soft_pvc     0.8671    0.8794    0.8732       141
                soft_tiles     0.8730    0.8730    0.8730        63
                     tiled     0.8333    0.7303    0.7784        89
                      wood     0.7273    0.8205    0.7711       117
    
                  accuracy                         0.8110       762
                 macro avg     0.8190    0.7921    0.8035       762
              weighted avg     0.8145    0.8110    0.8107       762
    



```python
from scikitplot.metrics import plot_confusion_matrix
ax = plot_confusion_matrix(labels, predictions, figsize = (10,8))
ax.set_xticklabels(label_encoder.classes_, rotation = 90, fontsize = 10)
ax.set_yticklabels(label_encoder.classes_, fontsize = 10);
#plt.xticklabels();
```


    
![png](output_34_0.png)
    



```python
value = 10
arr = test_dataset[value]['sequence'].numpy()
plt.figure(figsize = (20,10))
for idx, cl in enumerate(FEATURE_COLUMNS):#range(arr.shape[1]):
    plt.plot(arr[:,idx], label = cl)
    plt.legend();
    

```


    
![png](output_35_0.png)
    



```python

```
