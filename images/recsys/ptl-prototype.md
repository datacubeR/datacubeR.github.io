```python
import pandas as pd
import numpy as np
```


```python
movies = pd.read_csv('ml-25m/movies.csv')
print(movies.shape)
movies.columns
```

    (62423, 3)





    Index(['movieId', 'title', 'genres'], dtype='object')




```python
year = 2010
movies['year'] = movies.title.str.extract(r'\((\d{4})\)').astype("float")
movie_id_removed = movies.query('year < @year').movieId.tolist()
movies = movies.query('year >= @year')
movies
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
      <th>movieId</th>
      <th>title</th>
      <th>genres</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>14156</th>
      <td>73268</td>
      <td>Daybreakers (2010)</td>
      <td>Action|Drama|Horror|Thriller</td>
      <td>2010.0</td>
    </tr>
    <tr>
      <th>14161</th>
      <td>73319</td>
      <td>Leap Year (2010)</td>
      <td>Comedy|Romance</td>
      <td>2010.0</td>
    </tr>
    <tr>
      <th>14162</th>
      <td>73321</td>
      <td>Book of Eli, The (2010)</td>
      <td>Action|Adventure|Drama</td>
      <td>2010.0</td>
    </tr>
    <tr>
      <th>14222</th>
      <td>73744</td>
      <td>If You Love (Jos rakastat) (2010)</td>
      <td>Drama|Musical|Romance</td>
      <td>2010.0</td>
    </tr>
    <tr>
      <th>14256</th>
      <td>73929</td>
      <td>Legion (2010)</td>
      <td>Action|Fantasy|Horror|Thriller</td>
      <td>2010.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>62412</th>
      <td>209143</td>
      <td>The Painting (2019)</td>
      <td>Animation|Documentary</td>
      <td>2019.0</td>
    </tr>
    <tr>
      <th>62413</th>
      <td>209145</td>
      <td>Liberté (2019)</td>
      <td>Drama</td>
      <td>2019.0</td>
    </tr>
    <tr>
      <th>62415</th>
      <td>209151</td>
      <td>Mao Zedong 1949 (2019)</td>
      <td>(no genres listed)</td>
      <td>2019.0</td>
    </tr>
    <tr>
      <th>62418</th>
      <td>209157</td>
      <td>We (2018)</td>
      <td>Drama</td>
      <td>2018.0</td>
    </tr>
    <tr>
      <th>62420</th>
      <td>209163</td>
      <td>Bad Poems (2018)</td>
      <td>Comedy|Drama</td>
      <td>2018.0</td>
    </tr>
  </tbody>
</table>
<p>20489 rows × 4 columns</p>
</div>




```python
len(movie_id_removed)
```




    41524




```python
movies_mapping = movies[['movieId','title']].set_index('movieId').to_dict()['title']
```

# Ratings


```python
ratings = pd.read_csv('ml-25m/ratings.csv', parse_dates=['timestamp'])
print(ratings.columns)
print(ratings.shape)
ratings.userId.nunique()

```

    Index(['userId', 'movieId', 'rating', 'timestamp'], dtype='object')
    (25000095, 4)





    162541




```python

ratings = ratings.query('movieId not in @movie_id_removed')
ratings['rating'] = 1
ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')
ratings

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
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>712</th>
      <td>3</td>
      <td>73268</td>
      <td>1</td>
      <td>2015-08-13 14:11:38</td>
    </tr>
    <tr>
      <th>713</th>
      <td>3</td>
      <td>73321</td>
      <td>1</td>
      <td>2015-08-13 13:52:05</td>
    </tr>
    <tr>
      <th>715</th>
      <td>3</td>
      <td>74458</td>
      <td>1</td>
      <td>2017-04-21 14:39:18</td>
    </tr>
    <tr>
      <th>716</th>
      <td>3</td>
      <td>74789</td>
      <td>1</td>
      <td>2019-08-18 00:59:42</td>
    </tr>
    <tr>
      <th>717</th>
      <td>3</td>
      <td>76077</td>
      <td>1</td>
      <td>2017-01-18 16:15:09</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>24999773</th>
      <td>162538</td>
      <td>111617</td>
      <td>1</td>
      <td>2015-08-05 14:15:09</td>
    </tr>
    <tr>
      <th>24999774</th>
      <td>162538</td>
      <td>112138</td>
      <td>1</td>
      <td>2015-08-05 14:14:35</td>
    </tr>
    <tr>
      <th>24999775</th>
      <td>162538</td>
      <td>112556</td>
      <td>1</td>
      <td>2015-08-05 14:25:33</td>
    </tr>
    <tr>
      <th>24999776</th>
      <td>162538</td>
      <td>116797</td>
      <td>1</td>
      <td>2015-08-05 13:25:21</td>
    </tr>
    <tr>
      <th>24999777</th>
      <td>162538</td>
      <td>126548</td>
      <td>1</td>
      <td>2015-08-05 14:24:57</td>
    </tr>
  </tbody>
</table>
<p>2711937 rows × 4 columns</p>
</div>



# Label Encoder


```python
from sklearn.preprocessing import LabelEncoder

user_encoder = LabelEncoder()
movie_encoder = LabelEncoder()
ratings['userId'] = user_encoder.fit_transform(ratings.userId)
ratings['movieId'] = movie_encoder.fit_transform(ratings.movieId)

user_encoder.classes_
movie_encoder.classes_
```




    array([ 73268,  73319,  73321, ..., 209151, 209157, 209163])




```python
from scipy.sparse import csr_matrix
np.random.seed(42)
def create_matrix(data, user_col, item_col, rating_col):
    """
    creates the sparse user-item interaction matrix

    Parameters
    ----------
    data : DataFrame
        implicit rating data

    user_col : str
        user column name

    item_col : str
        item column name

    ratings_col : str
        implicit rating column name
    """
    
    data[[user_col, item_col]] = data[[user_col, item_col]].astype('category')
    
    rows = data[user_col].cat.codes
    cols = data[item_col].cat.codes
    rating = data[rating_col]
    user_item_matrix = csr_matrix((rating, (rows, cols)))
    return user_item_matrix

user_item_matrix = create_matrix(ratings, 'userId', 'movieId', 'rating')
```


```python
user_item_matrix.shape
```




    (60780, 20455)



## Train Test Split


```python
ratings['test'] = ratings.groupby(['userId'])['timestamp'].rank(method='first', ascending=False)

train_ratings = ratings.query('test != 1').drop(columns = ['test', 'timestamp'])
test_ratings = ratings.query('test == 1').drop(columns = ['test', 'timestamp'])

```


```python
train_ratings.shape, test_ratings.shape
```




    ((2651157, 3), (60780, 3))



## Problema de Clasificación


```python
# bla = train_ratings.drop_duplicates(subset = ['userId', 'movieId'], keep = 'first')

# %%time
# unique_movies = set(train_ratings.movieId)
# def create_negative_movies(df, userid = 'userId', movieid = 'movieId',neg_examples = 4):
#     unique_movies = set(df[movieid])
    
#     movies = []
#     uids = df[userid].unique()
#     for u in uids:
#         movies.extend(np.random.choice(list(unique_movies - set(df[movieid][df[userid] == u])), size = neg_examples))
        
#     return uids, movies

# %%time 
# neg_examples = 4
# users, negative_movies = create_negative_movies(train_ratings)
# negative_movies_df = pd.DataFrame(dict(userId = np.repeat(users, [neg_examples]*len(users)),
#                 movieId = negative_movies,
#                 ratings = np.zeros(len(negative_movies)))
#                 )

# negative_movies_df.to_csv('negative_movies.csv', index = False)
```

## Nueva Implementación


```python
print('Training Dimensions: ', train_ratings.userId.nunique(), train_ratings.movieId.nunique())
print('Test Dimensions: ', test_ratings.userId.nunique(), test_ratings.movieId.nunique())

print('Movies in Train: ', train_ratings.sum())
print('Movies in Test: ', test_ratings.sum())
```

    Training Dimensions:  56706 20391
    Test Dimensions:  60780 4176
    Movies in Train:  rating    2651157
    dtype: int64
    Movies in Test:  rating    60780
    dtype: int64



```python
train_users = train_ratings.userId.unique().tolist()
test_users = test_ratings.userId.unique().tolist()

print(len(train_users))
print(len(test_users))

```

    56706
    60780



```python
def create_negative_df(user_ids, user_item, neg_examples = 4, test = False):
    
    movies_id = np.arange(user_item.shape[1])
    negative_movies = []
    examples = []
    for i in range(len(user_ids)):

        interacted = user_item[i].nonzero()[1]
        x = ~np.isin(movies_id, interacted)
        x = np.argwhere(x).squeeze(1)
        
        if test:
            size = neg_examples
        else:
            size = len(interacted)*neg_examples
        
        x = np.random.choice(x, size = size)
        negative_movies.extend(x)
        examples.append(size)
        
    negative_movies_df = pd.DataFrame(dict(userId = np.repeat(user_ids, examples),
                        movieId = negative_movies,
                        rating = np.zeros(len(negative_movies)))
                        )
    return negative_movies_df

```


```python
%%time
train_negative_movies_df = create_negative_df(train_users, user_item_matrix, neg_examples = 4)
train_negative_movies_df.shape
```

    CPU times: user 22.9 s, sys: 268 ms, total: 23.1 s
    Wall time: 23.1 s





    (10146692, 3)




```python
%%time
test_negative_movies_df = create_negative_df(test_users, user_item_matrix, neg_examples = 99, test = True)
test_negative_movies_df.shape
```

    CPU times: user 24.3 s, sys: 160 ms, total: 24.5 s
    Wall time: 24.5 s





    (6017220, 3)




```python
full_training_df = train_ratings.append(train_negative_movies_df)
full_test_df = test_ratings.append(test_negative_movies_df)

full_training_df.shape, full_test_df.shape
```




    ((12797849, 3), (6078000, 3))




```python
full_training_df.info(memory_usage='deep'), full_test_df.info(memory_usage='deep')
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 12797849 entries, 712 to 10146691
    Data columns (total 3 columns):
     #   Column   Dtype  
    ---  ------   -----  
     0   userId   int64  
     1   movieId  int64  
     2   rating   float64
    dtypes: float64(1), int64(2)
    memory usage: 390.6 MB
    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 6078000 entries, 734 to 6017219
    Data columns (total 3 columns):
     #   Column   Dtype  
    ---  ------   -----  
     0   userId   int64  
     1   movieId  int64  
     2   rating   float64
    dtypes: float64(1), int64(2)
    memory usage: 185.5 MB





    (None, None)




```python
full_training_df.columns
```




    Index(['userId', 'movieId', 'rating'], dtype='object')



# Creating the Neural Network


```python
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

pl.seed_everything(42, workers=True)
```

    Global seed set to 42





    42




```python
from torch.utils.data import Dataset, DataLoader
from multiprocessing import cpu_count

class MovieData(Dataset):
    def __init__(self, users, movies, ratings):
        self.users = users
        self.movies = movies
        self.ratings = ratings
        
    def __len__(self):
        return len(self.ratings)
        
    def __getitem__(self, idx):
    
        users = self.users.iloc[idx]
        movies = self.movies.iloc[idx]
        ratings = self.ratings.iloc[idx]

        return dict(
            users = torch.tensor(users, dtype=torch.long),
            movies = torch.tensor(movies, dtype=torch.long),
            ratings = torch.tensor(ratings, dtype=torch.float)
        )

class MovieDataModule(pl.LightningDataModule):
    def __init__(self, train_df, test_df, batch_size = 512):
        super().__init__()
        
        self.train_df = train_df 
        self.test_df = test_df 
        self.batch_size = batch_size
        
    def setup(self, stage=None):
        
        self.train_data = MovieData(self.train_df.userId, self.train_df.movieId, self.train_df.rating)
        self.test_data = MovieData(self.test_df.userId, self.test_df.movieId, self.test_df.rating)
    
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers = 10)
    
    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers = 10)
    

```


```python
class NCF(nn.Module):
    def __init__(self, dim_users, dim_movies, n_out = 1):
        super().__init__()
        
        self.user_embedding = nn.Embedding(dim_users, 8)
        self.movie_embedding = nn.Embedding(dim_movies, 8)
        
        self.encoder = nn.Sequential(
                            nn.Linear(16,64),
                            nn.ReLU(inplace=True),
                            nn.Linear(64,32),
                            nn.ReLU(inplace=True),
                            nn.Linear(32,n_out)
                        )
        
    def forward(self, users, movies):
        user_emb = self.user_embedding(users)
        movie_emb = self.movie_embedding(movies)
        
        x = torch.cat((user_emb, movie_emb), dim = 1)
        x = self.encoder(x)
        return x
    
    
class RecSys(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.criterion = nn.BCEWithLogitsLoss()
        
    def forward(self,users, movies):
        x = self.model(users, movies)
        return x
        
    def training_step(self, batch, batch_idx):
        users, movies, ratings = batch['users'], batch['movies'], batch['ratings']
        preds = self(users, movies)
        # print('preds:',  preds.shape)
        # print('ratings: ', ratings.shape)
        loss = self.criterion(preds, ratings.view(-1,1))
        self.log('train_loss', loss,  prog_bar = True, logger = True)
        return {'loss': loss}
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr = 1e-3)


```


```python
full_test_df.userId.astype('int64').max(), full_test_df.movieId.astype('int64').max(), full_test_df.shape
```




    (60779, 20454, (6078000, 3))




```python
dim_users = full_training_df.userId.astype('int64').max() + 1
dim_movies = full_training_df.movieId.astype('int64').max() + 1
print(dim_users, dim_movies)

```

    60780 20455



```python

model = NCF(dim_users, dim_movies)
```


```python
dm = MovieDataModule(full_training_df, full_test_df, batch_size=512)
dm.setup()
```


```python
train_batch = next(iter(dm.train_dataloader()))
```


```python
value = np.random.randint(1,32)
train_batch['users'][value], train_batch['movies'][value], train_batch['ratings'][value]
```




    (tensor(55545), tensor(15787), tensor(0.))




```python
train_batch['users'].shape, train_batch['movies'].shape, train_batch['ratings'].shape
```




    (torch.Size([512]), torch.Size([512]), torch.Size([512]))




```python
recommender = RecSys(model)
```


```python
recommender(train_batch['users'], train_batch['movies']).shape
```




    torch.Size([512, 1])




```python
mc = ModelCheckpoint(
    dirpath = 'checkpoints',
    #filename = 'best-checkpoint',
    save_last = True,
    save_top_k = 1,
    verbose = True,
    monitor = 'train_loss', 
    mode = 'min'
    )

mc.CHECKPOINT_NAME_LAST = 'best-checkpoint-latest'
```


```python
trainer = pl.Trainer(max_epochs=5,
                    accelerator="gpu",
                    devices=1, 
                    callbacks=[mc], 
                    progress_bar_refresh_rate=30, 
                    # fast_dev_run=True,
                    #overfit_batches=1
                    )
trainer.fit(recommender, dm)

```

    /home/alfonso/miniconda3/envs/dl/lib/python3.7/site-packages/pytorch_lightning/trainer/connectors/callback_connector.py:97: LightningDeprecationWarning: Setting `Trainer(progress_bar_refresh_rate=30)` is deprecated in v1.5 and will be removed in v1.7. Please pass `pytorch_lightning.callbacks.progress.TQDMProgressBar` with `refresh_rate` directly to the Trainer's `callbacks` argument instead. Or, to disable the progress bar pass `enable_progress_bar = False` to the Trainer.
      f"Setting `Trainer(progress_bar_refresh_rate={progress_bar_refresh_rate})` is deprecated in v1.5 and"
    GPU available: True, used: True
    TPU available: False, using: 0 TPU cores
    IPU available: False, using: 0 IPUs
    HPU available: False, using: 0 HPUs
    /home/alfonso/miniconda3/envs/dl/lib/python3.7/site-packages/pytorch_lightning/callbacks/model_checkpoint.py:608: UserWarning: Checkpoint directory /home/alfonso/Documents/kaggle/recom/checkpoints exists and is not empty.
      rank_zero_warn(f"Checkpoint directory {dirpath} exists and is not empty.")
    LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
    
      | Name      | Type              | Params
    ------------------------------------------------
    0 | model     | NCF               | 653 K 
    1 | criterion | BCEWithLogitsLoss | 0     
    ------------------------------------------------
    653 K     Trainable params
    0         Non-trainable params
    653 K     Total params
    2.612     Total estimated model params size (MB)



    Training: 0it [00:00, ?it/s]


    Epoch 0, global step 24996: 'train_loss' reached 0.08028 (best 0.08028), saving model to '/home/alfonso/Documents/kaggle/recom/checkpoints/epoch=0-step=24996.ckpt' as top 1
    Epoch 1, global step 49992: 'train_loss' was not in top 1
    Epoch 2, global step 74988: 'train_loss' reached 0.07823 (best 0.07823), saving model to '/home/alfonso/Documents/kaggle/recom/checkpoints/epoch=2-step=74988.ckpt' as top 1
    Epoch 3, global step 99984: 'train_loss' reached 0.06737 (best 0.06737), saving model to '/home/alfonso/Documents/kaggle/recom/checkpoints/epoch=3-step=99984.ckpt' as top 1
    Epoch 4, global step 124980: 'train_loss' reached 0.06487 (best 0.06487), saving model to '/home/alfonso/Documents/kaggle/recom/checkpoints/epoch=4-step=124980.ckpt' as top 1



```python
# from torchmetrics.functional import retrieval_hit_rate

# preds = torch.tensor([[0.9, 0.3, 0.9,0.4],[0.9, 0.3, 0.9,0.4]])
# target = torch.tensor([[False, False, True,False],[False, True, False,False]])
# retrieval_hit_rate(preds, target, k=2)
```


```python
@torch.inference_mode()
def predict(model, dm):
    model.eval()
    preds = []
    for item in dm.test_dataloader():
        
        pred = torch.sigmoid(model(item['users'], item['movies']))
        preds.extend(pred.cpu().detach().numpy())
        
    return preds
```


```python
predictions= np.array(predict(recommender, dm))
predictions.shape
```

    Exception ignored in: Exception ignored in: <function _MultiProcessingDataLoaderIter.__del__ at 0x7fdcf8434170><function _MultiProcessingDataLoaderIter.__del__ at 0x7fdcf8434170>
    
    Traceback (most recent call last):
    Traceback (most recent call last):
      File "/home/alfonso/miniconda3/envs/dl/lib/python3.7/site-packages/torch/utils/data/dataloader.py", line 1358, in __del__
      File "/home/alfonso/miniconda3/envs/dl/lib/python3.7/site-packages/torch/utils/data/dataloader.py", line 1358, in __del__
        self._shutdown_workers()
      File "/home/alfonso/miniconda3/envs/dl/lib/python3.7/site-packages/torch/utils/data/dataloader.py", line 1341, in _shutdown_workers
        if w.is_alive():Exception ignored in: 
      File "/home/alfonso/miniconda3/envs/dl/lib/python3.7/multiprocessing/process.py", line 151, in is_alive
    <function _MultiProcessingDataLoaderIter.__del__ at 0x7fdcf8434170>
    Traceback (most recent call last):
        assert self._parent_pid == os.getpid(), 'can only test a child process'  File "/home/alfonso/miniconda3/envs/dl/lib/python3.7/site-packages/torch/utils/data/dataloader.py", line 1358, in __del__
    
    AssertionError    self._shutdown_workers(): 
    can only test a child process  File "/home/alfonso/miniconda3/envs/dl/lib/python3.7/site-packages/torch/utils/data/dataloader.py", line 1341, in _shutdown_workers
    
        if w.is_alive():
      File "/home/alfonso/miniconda3/envs/dl/lib/python3.7/multiprocessing/process.py", line 151, in is_alive
        assert self._parent_pid == os.getpid(), 'can only test a child process'
        AssertionErrorself._shutdown_workers(): 
    can only test a child process
      File "/home/alfonso/miniconda3/envs/dl/lib/python3.7/site-packages/torch/utils/data/dataloader.py", line 1341, in _shutdown_workers
        if w.is_alive():
      File "/home/alfonso/miniconda3/envs/dl/lib/python3.7/multiprocessing/process.py", line 151, in is_alive
        assert self._parent_pid == os.getpid(), 'can only test a child process'
    AssertionError: can only test a child process





    (6078000, 1)




```python
full_test_df['preds'] = predictions
full_test_df
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
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
      <th>preds</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>734</th>
      <td>0</td>
      <td>230</td>
      <td>1.0</td>
      <td>0.987030</td>
    </tr>
    <tr>
      <th>1066</th>
      <td>1</td>
      <td>929</td>
      <td>1.0</td>
      <td>0.749257</td>
    </tr>
    <tr>
      <th>2855</th>
      <td>2</td>
      <td>465</td>
      <td>1.0</td>
      <td>0.911151</td>
    </tr>
    <tr>
      <th>2889</th>
      <td>3</td>
      <td>2505</td>
      <td>1.0</td>
      <td>0.973490</td>
    </tr>
    <tr>
      <th>3015</th>
      <td>4</td>
      <td>9907</td>
      <td>1.0</td>
      <td>0.959640</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>6017215</th>
      <td>60779</td>
      <td>2442</td>
      <td>0.0</td>
      <td>0.006992</td>
    </tr>
    <tr>
      <th>6017216</th>
      <td>60779</td>
      <td>10800</td>
      <td>0.0</td>
      <td>0.003167</td>
    </tr>
    <tr>
      <th>6017217</th>
      <td>60779</td>
      <td>17767</td>
      <td>0.0</td>
      <td>0.000137</td>
    </tr>
    <tr>
      <th>6017218</th>
      <td>60779</td>
      <td>7073</td>
      <td>0.0</td>
      <td>0.000070</td>
    </tr>
    <tr>
      <th>6017219</th>
      <td>60779</td>
      <td>2124</td>
      <td>0.0</td>
      <td>0.000730</td>
    </tr>
  </tbody>
</table>
<p>6078000 rows × 4 columns</p>
</div>




```python
recomendations = full_test_df.sort_values(by = ['userId','preds'], ascending=[True, False]).groupby('userId').head(10)
```


```python
# Hit Ratio @ 10

recomendations.rating.sum()/recomendations.userId.nunique()
```




    0.9457880881869036



## Índices Iniciales


```python
def back_to_normal(df, user_encoder, movie_encoder, movies_mapping):
    
    idx_movies = df.movieId.tolist()
    idx_users = df.userId.tolist()
    return pd.DataFrame(dict(userId = user_encoder.classes_[idx_users],
                    movieId = pd.Series(movie_encoder.classes_[idx_movies]).map(movies_mapping),
                    rating = df.rating.tolist()))
```


```python
visto= back_to_normal(train_ratings, user_encoder, movie_encoder, movies_mapping)
visto.shape
```




    (2651157, 3)




```python
recomendar = back_to_normal(recomendations, user_encoder, movie_encoder, movies_mapping)
recomendar.shape
```




    (607800, 3)




```python
user = 4
print(visto.query('userId == @user')['movieId'])
recomendar.query('userId == @user')

```

    193                                Shutter Island (2010)
    194    Percy Jackson & the Olympians: The Lightning T...
    195                      How to Train Your Dragon (2010)
    196                           Clash of the Titans (2010)
    197                                    Iron Man 2 (2010)
                                 ...                        
    303             Spider-Man: Into the Spider-Verse (2018)
    304             John Wick: Chapter 3 – Parabellum (2019)
    305                    Pokémon: Detective Pikachu (2019)
    306                               Ford v. Ferrari (2019)
    307         Fast & Furious Presents: Hobbs & Shaw (2019)
    Name: movieId, Length: 115, dtype: object





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
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10</th>
      <td>4</td>
      <td>Thor: The Dark World (2013)</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>4</td>
      <td>Margin Call (2011)</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>4</td>
      <td>Kubo and the Two Strings (2016)</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>4</td>
      <td>John Carter (2012)</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>4</td>
      <td>Autómata (Automata) (2014)</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>4</td>
      <td>You Were Never Really Here (2017)</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>4</td>
      <td>Aloha (2015)</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>4</td>
      <td>Thanks for Sharing (2012)</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>4</td>
      <td>Eva (2011)</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>4</td>
      <td>Magic Mike XXL (2015)</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
user = 6265
print(visto.query('userId == @user')['movieId'])
recomendar.query('userId == @user')

```

    100326    Cabin in the Woods, The (2012)
    100327                Snowpiercer (2013)
    100328                  Gone Girl (2014)
    100329         The Imitation Game (2014)
    Name: movieId, dtype: object





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
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>22630</th>
      <td>6265</td>
      <td>Midnight in Paris (2011)</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>22631</th>
      <td>6265</td>
      <td>Friends with Benefits (2011)</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>22632</th>
      <td>6265</td>
      <td>Saw VII 3D - The Final Chapter (2010)</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>22633</th>
      <td>6265</td>
      <td>Searching (2018)</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>22634</th>
      <td>6265</td>
      <td>Aladdin (2019)</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>22635</th>
      <td>6265</td>
      <td>The Dark Tower (2017)</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>22636</th>
      <td>6265</td>
      <td>The BFG (2016)</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>22637</th>
      <td>6265</td>
      <td>ARQ (2016)</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>22638</th>
      <td>6265</td>
      <td>A Wrinkle in Time (2018)</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>22639</th>
      <td>6265</td>
      <td>Magic of Belle Isle, The (2012)</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
user = 21962
print(visto.query('userId == @user')['movieId'])
recomendar.query('userId == @user')

```

    353218                                Shutter Island (2010)
    353219                           Alice in Wonderland (2010)
    353220                                   Toy Story 3 (2010)
    353221    Shrek Forever After (a.k.a. Shrek: The Final C...
    Name: movieId, dtype: object





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
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>82220</th>
      <td>21962</td>
      <td>Iron Man 2 (2010)</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>82221</th>
      <td>21962</td>
      <td>Alien: Covenant (2017)</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>82222</th>
      <td>21962</td>
      <td>The Huntsman Winter's War (2016)</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>82223</th>
      <td>21962</td>
      <td>Sisters (2015)</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>82224</th>
      <td>21962</td>
      <td>Scary Movie 5 (Scary MoVie) (2013)</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>82225</th>
      <td>21962</td>
      <td>Cop Car (2015)</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>82226</th>
      <td>21962</td>
      <td>Norwegian Wood (Noruwei no mori) (2010)</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>82227</th>
      <td>21962</td>
      <td>Oslo, August 31st (Oslo, 31. august) (2011)</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>82228</th>
      <td>21962</td>
      <td>Country Strong (2010)</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>82229</th>
      <td>21962</td>
      <td>Premature (2014)</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
user = 17568
print(visto.query('userId == @user')['movieId'])
recomendar.query('userId == @user')
```

    279619                                     Inception (2010)
    279620                                        Easy A (2010)
    279621         Men in Black III (M.III.B.) (M.I.B.³) (2012)
    279622                                           Ted (2012)
    279623                                   Cloud Atlas (2012)
    279624                              Django Unchained (2012)
    279625                                       Elysium (2013)
    279626                      Wolf of Wall Street, The (2013)
    279627                                The Lego Movie (2014)
    279628    Birdman: Or (The Unexpected Virtue of Ignoranc...
    279629                                      Deadpool (2016)
    279630                                Big Short, The (2015)
    Name: movieId, dtype: object





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
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>65420</th>
      <td>17568</td>
      <td>Dark Knight Rises, The (2012)</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>65421</th>
      <td>17568</td>
      <td>Star Trek Into Darkness (2013)</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>65422</th>
      <td>17568</td>
      <td>Star Trek Into Darkness (2013)</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>65423</th>
      <td>17568</td>
      <td>Furious 7 (2015)</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>65424</th>
      <td>17568</td>
      <td>Beasts of the Southern Wild (2012)</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>65425</th>
      <td>17568</td>
      <td>Stonehearst Asylum (2014)</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>65426</th>
      <td>17568</td>
      <td>The Purge: Election Year (2016)</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>65427</th>
      <td>17568</td>
      <td>Creed II (2018)</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>65428</th>
      <td>17568</td>
      <td>Danny Collins (2015)</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>65429</th>
      <td>17568</td>
      <td>Max Steel (2016)</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
user = 63
print(visto.query('userId == @user')['movieId'])
recomendar.query('userId == @user')

```

    623                              Easy A (2010)
    624                             Tangled (2010)
    625                         Bridesmaids (2011)
    626                     Horrible Bosses (2011)
    627                Crazy, Stupid, Love. (2011)
    628                      21 Jump Street (2012)
    629                       Pitch Perfect (2012)
    630    Perks of Being a Wallflower, The (2012)
    631                   Great Gatsby, The (2013)
    632                      Now You See Me (2013)
    633                   We're the Millers (2013)
    634                          About Time (2013)
    635            Wolf of Wall Street, The (2013)
    636                           Gone Girl (2014)
    637                          Inside Out (2015)
    638                                Room (2015)
    639                               Moana (2016)
    640                                Coco (2017)
    Name: movieId, dtype: object





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
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>200</th>
      <td>63</td>
      <td>Spotlight (2015)</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>201</th>
      <td>63</td>
      <td>Twilight Saga: Eclipse, The (2010)</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>202</th>
      <td>63</td>
      <td>Sorcerer's Apprentice, The (2010)</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>203</th>
      <td>63</td>
      <td>Melancholia (2011)</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>204</th>
      <td>63</td>
      <td>Oz the Great and Powerful (2013)</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>205</th>
      <td>63</td>
      <td>Venom (2018)</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>206</th>
      <td>63</td>
      <td>Selma (2014)</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>207</th>
      <td>63</td>
      <td>Burlesque (2010)</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>208</th>
      <td>63</td>
      <td>Silent Hill: Revelation 3D (2012)</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>209</th>
      <td>63</td>
      <td>Double, The (2011)</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
user = 162532
print(visto.query('userId == @user')['movieId'])
recomendar.query('userId == @user')

```

    2650878                      How to Train Your Dragon (2010)
    2650879                                      Kick-Ass (2010)
    2650880                    Exit Through the Gift Shop (2010)
    2650881                                    Iron Man 2 (2010)
    2650882                                 Despicable Me (2010)
    2650883                                     Inception (2010)
    2650884                   Scott Pilgrim vs. the World (2010)
    2650885                           Social Network, The (2010)
    2650886                                        Easy A (2010)
    2650887    Harry Potter and the Deathly Hallows: Part 1 (...
    2650888                            King's Speech, The (2010)
    2650889                                   Source Code (2011)
    2650890                                          Thor (2011)
    2650891                            X-Men: First Class (2011)
    2650892    Harry Potter and the Deathly Hallows: Part 2 (...
    2650893            Captain America: The First Avenger (2011)
    2650894                                 Avengers, The (2012)
    2650895                                          Hugo (2011)
    2650896                              The Hunger Games (2012)
    2650897                        Dark Knight Rises, The (2012)
    2650898            Sherlock Holmes: A Game of Shadows (2011)
    2650899                                  Intouchables (2011)
    2650900                                        Looper (2012)
    2650901                                          Argo (2012)
    2650902                       Silver Linings Playbook (2012)
    2650903            Hobbit: An Unexpected Journey, The (2012)
    2650904                                    Iron Man 3 (2013)
    Name: movieId, dtype: object





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
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>607750</th>
      <td>162532</td>
      <td>Guardians of the Galaxy (2014)</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>607751</th>
      <td>162532</td>
      <td>Only the Brave (2017)</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>607752</th>
      <td>162532</td>
      <td>Immigrant, The (2013)</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>607753</th>
      <td>162532</td>
      <td>Diary of a Wimpy Kid: Rodrick Rules (2011)</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>607754</th>
      <td>162532</td>
      <td>Spy Kids: All the Time in the World in 4D (2011)</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>607755</th>
      <td>162532</td>
      <td>The Belko Experiment (2017)</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>607756</th>
      <td>162532</td>
      <td>All the Way (2016)</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>607757</th>
      <td>162532</td>
      <td>Come Together (2016)</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>607758</th>
      <td>162532</td>
      <td>Batman: Gotham by Gaslight (2018)</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>607759</th>
      <td>162532</td>
      <td>Kizumonogatari Part 1: Tekketsu (2016)</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
