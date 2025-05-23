# SELF-TF: Semantic Fusion and Early Lifecycle Feature-Enhanced time-series Model for Trending topic Popularity Forecasting Time series forecasting models


## Set-up

- Install the required packages (pip or conda)
    - `pip install -r requirements.txt`

- Download data csv
    - https://github.com/wangxiaoshi1103/trending_weibo_data  
- Built-in data
    -  trending_weibo_new/data_csv/weibo   (built in data less than 25M, if you want more data, please download from git above)
    -  trending_weibo_new/data_csv/twitter

- only dependencies 
    - Before training the model,please modify 
        - mv data_csv data
        - mkdir data/chinese-roberta-wwm-ext  ## download bert or use you own pretrained text representation

- config semantic representation、features、model parameters、data set etc
    - conf/weibo.yaml    ## The present configuration can only execute the basic setup. If you wish to replicate the paper's results, please refer to conf/weibo_1234.yaml
    - conf/twitter.yaml

- Train on weibo/twitter dataset    
    - `python ./main.py`
    - #When you execute main.py for the second time, please change the model name, or clear the existing model in the logs.
- Inference on weibo/twitter dataset
    - `python ./main_infer.py`
    - #When you execute main_infer.py, please modify the model name to be consistent with the one in main.py. 

    
- Semantic Fusion fusion transformer
    - `model: selftft_transformer`

- Use TensorBoard to visualize the training process
    - refer to use_tensorboard.txt
