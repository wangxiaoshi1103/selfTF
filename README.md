# SELF-TF: Semantic Fusion and Early Lifecycle Feature-Enhanced time-series Model for Trending topic Popularity Forecasting Time series forecasting models


## Set-up

- Install the required packages (pip or conda)
    - `pip install -r requirements.txt`

- Download data xlsx csv
    - https://github.com/rahadiana/twitter_trend_world
    - https://github.com/wangxiaoshi1103/trending_weibo_data
- Built-in data
    -  trending_weibo_new/data_csv/weibo
    -  trending_weibo_new/data_csv/twitter

- only dependencies 
    - Before training the model,please modify 
        - mv data_csv data
        - mkdir data/chinese-roberta-wwm-ext  ## download bert or use you own pretrained text representation

- Train on weibo/twitter dataset    
    - `python ./main.py`
    - #When you execute main.py for the second time, please change the model name, or clear the existing model in the logs.
- Inferance on weibo/twitter dataset
    - `python ./main_infer.py`
    - #When you execute main_infer.py, please modify the model name to be consistent with the one in main.py. 

    
- Semantic Fusion fusion transformer
    - `model: selftft_transformer`

- Use TensorBoard to visualize the training process
    - refer to use_tensorboard.txt
