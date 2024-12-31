# SELF-TF: Semantic Fusion and Early Lifecycle Feature-Enhanced time-series Model for Trending topic Popularity Forecasting Time series forecasting models


## Set-up

- Install the required packages (pip or conda)
    - `pip install -r requirements.txt`

- Download data xlsx
    -  https://github.com/(Anonymous)/1
    -  https://github.com/rahadiana/twitter_trend_world
- Built-in data
    -  trending_weibo_new/data_csv/weibo
    -  trending_weibo_new/data_csv/twitter

- dependencies
    - please download bert/bge model or pretrain your own text representation model
    - Before training the model, please modify the text representation model path which used by BertTokenizer like
        - tokenizer = BertTokenizer.from_pretrained('./data/chinese-roberta-wwm-ext')

- Train on weibo/twitter dataset    
    - `python ./main.py`
- Inferance on weibo/twitter dataset
    - `python ./main_infer.py`
    
    
- Semantic Fusion fusion transformer
    - `model: selftft_transformer`

- Use TensorBoard to visualize the training process
    - use_tensorboard.txt
