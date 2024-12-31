# coding=utf-8
# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Custom formatting functions for Twitter dataset.

Defines dataset specific column definitions and data transformations. Uses
entity specific z-score normalization.
"""
import numpy as np
import data_formatters.base
import data_formatters.utils as utils
import pandas as pd
import sklearn.preprocessing
import yaml
from sklearn.model_selection import train_test_split
import pdb
from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn

GenericDataFormatter = data_formatters.base.GenericDataFormatter
DataTypes = data_formatters.base.DataTypes
InputTypes = data_formatters.base.InputTypes


class TwitterFormatter(GenericDataFormatter):
  """Defines and formats data for the twitter dataset.

  Note that per-entity z-score normalization is used here, and is implemented
  across functions.

  Attributes:
    column_definition: Defines input and data type of column used in the
      twitter.
    identifiers: Entity identifiers used in twitters.
  """

  _column_definition_bak = [
      ('id', DataTypes.REAL_VALUED, InputTypes.ID),
      ('hours_from_start', DataTypes.REAL_VALUED, InputTypes.TIME),
      ('power_usage', DataTypes.REAL_VALUED, InputTypes.TARGET),
      ('hour', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
      ('day_of_week', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
      ('hours_from_start', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
      ('categorical_id', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
  ]

  _column_definition = [
      ('id', DataTypes.REAL_VALUED, InputTypes.ID),  # id
      ('minsFromStart', DataTypes.REAL_VALUED, InputTypes.TIME),  # time
      ('hotPos', DataTypes.REAL_VALUED, InputTypes.TARGET),  # target
      ('minsFromStart', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),  # 1：原始采集数据
      ('minsFromFirst', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),  # 1：距离上榜时间的分钟
      ('is_business_day', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),  # 4
      ('popular', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),  # 1
      ('active_time_hr', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),  # 3：加工后特征
      ('climbing_time', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),  # 3
      ('top_hotPos', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),  # 3
      ('max_span_up_hotPos', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),  # 3
      ('max_span_down_hotPos', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),  # 3
      ('hotPos_flucation_avg_span', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),  # 3
      ('max_span_up_popular', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),  # 3
      ('max_span_down_popular', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),  # 3
      ('popular_avg', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),  # 3
      ('topicName', DataTypes.TEXT, InputTypes.STATIC_INPUT),  # 2
      ('query', DataTypes.TEXT, InputTypes.STATIC_INPUT),  # 2
  ]

  def __init__(self):
    """Initialises formatter."""

    self.identifiers = None
    self._real_scalers = None
    self._cat_scalers = None
    self._target_scaler = None
    self._num_classes_per_cat_input = None
    self._label_to_integer_mappings = None
    self._category_to_vectors = {}
    self._time_steps = self.get_fixed_params()['total_time_steps']
    self._is_bert = self.get_fixed_params()['is_bert']
    self._hidden_layer_size = self.get_fixed_params()['hidden_layer_size']


  def day_of_week(self, df):
      #df['time'] = pd.to_datetime(df['time'], format ='%Y-%m-%d %H:%M')
      df['time'] = pd.to_datetime(df['time'])
      df['dayOfWeek'] = df['time'].dt.dayofweek
      return df

  def split_data_bak1(self, df, valid_boundary=1315, test_boundary=1339):
    """Splits data frame into training-validation-test data frames.

    This also calibrates scaling object, and transforms data for each split.

    Args:
      df: Source data frame to split.
      valid_boundary: Starting year for validation data
      test_boundary: Starting year for test data

    Returns:
      Tuple of transformed (train, valid, test) data.
    """

    print('Formatting train-valid-test splits.')
    #pdb.set_trace()

    index = df['days_from_start']
    train = df.loc[index < valid_boundary]
    valid = df.loc[(index >= valid_boundary - 7) & (index < test_boundary)]
    test = df.loc[index >= test_boundary - 7]

    #self.set_scalers(train)
    self.set_scalers(df)

    return (self.transform_inputs(data) for data in [train, valid, test])

  def split_data_bak(self, df, valid_boundary=1315, test_boundary=1339):
    ## 按行数划分训练集，验证集，测试集
    #train_rows = 8000
    #valid_rows = 1000
    #test_rows = 999

    #train = df.iloc[:train_rows]
    #valid = df.iloc[train_rows:train_rows + valid_rows]
    #test = df.iloc[train_rows + valid_rows:train_rows + valid_rows + test_rows]
    train = df
    valid = df
    test = df

    self.set_scalers(df)

    return (self.transform_inputs(data) for data in [train, valid, test])
      

  def split_data(self, df, valid_boundary=1315, test_boundary=1339):
      df = self.day_of_week(df)

      # 计算mins from start
      df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S')
      df['minsFromStart'] = (df['time'] - df['time'].min()) / np.timedelta64(1, 'm')
      df['minsFromStart'] = df['minsFromStart'].astype(float)

      #df['timeOnBoard'] = pd.to_datetime(df['timeOnBoard'], format='%Y-%m-%d %H:%M:%S')
      #df['FromFirstinsFromFirst'] = (df['time'] - df['timeOnBoard'].min()) / np.timedelta64(1, 'm')
      df['minsFromFirst'] = df['minsFromStart'].astype(float)


      #df['timeOnBoard'] = df['timeOnBoard'].apply(lambda x: x.timestamp())

      # df['time_first'] = pd.to_datetime(df['time_first'], format='%Y-%m-%d %H:%M:%S')
      # df['time_first'] = df['time_first'].apply(lambda x: x.timestamp())

      #text_to_code = {'G{:02d}'.format(i): i for i in range(len(df['introduct'].unique()))}
      #df['id'] = df['introduct'].astype('category').cat.codes.apply(lambda x:'G{:02d}'.format(x)) 
      #df['topicName'] = df['id']  
      #df['introductId'] = df['id']

      # 话题类型split
      #split_types = df['topicType'].str.split(',', expand=True).stack().reset_index(level=1, drop=True)
      #split_types.name = 'topicType'
      #df = df.drop(columns=['topicType']).join(split_types)
      #df = df.groupby('topicName').filter(lambda x: (x['minsFromStart'].max() - x['minsFromStart'].min()) >= 60)
      df['topicType'] = 1
      #pdb.set_trace()
      df = df.groupby('id').filter(lambda x: (x['minsFromStart'].max() - x['minsFromStart'].min()) >= 30)
      #df = df.groupby('id')

      train_list = []
      valid_list = []
      test_list = []

      #pdb.set_trace()
      for t in df['topicType'].unique():
          df_type = df[df['topicType'] == t]
          grouped = list(df_type.groupby('id'))
          train_groups, temp_groups = train_test_split(grouped, test_size=0.3, random_state=42)
          valid_groups, test_groups = train_test_split(temp_groups, test_size=1/3, random_state=42)
          train = pd.concat([group for _, group in train_groups])
          valid = pd.concat([group for _, group in valid_groups])
          test = pd.concat([group for _, group in test_groups])
          train_list.append(train)
          valid_list.append(valid)
          test_list.append(test)

          #train, temp = train_test_split(df_type, test_size=0.3, random_state=42)
          #valid, test = train_test_split(temp, test_size=1/3, random_state=42)
          #train_list.append(train)
          #valid_list.append(valid)
          #test_list.append(test)
      train = pd.concat(train_list)
      valid = pd.concat(valid_list)
      test = pd.concat(test_list)

      self.set_scalers(df)
      if self._is_bert > 0:
          self.set_bert_embeddings(df)
      return (self.transform_inputs(data) for data in [train, valid, test])

    # index = df['dayOfWeek']
    # train = df.loc[index < 6]
    # valid = df.loc[(index == 6)]
    # test = df.loc[index > 6]
    #
    # #self.set_scalers(train)
    # self.set_scalers(df)
    #
    # return (self.transform_inputs(data) for data in [train, valid, test])


  def set_scalers(self, df):
    """Calibrates scalers using the data supplied.

    Args:
      df: Data to use to calibrate scalers.
    """
    print('Setting scalers with training data...')

    column_definitions = self.get_column_definition()
    id_column = utils.get_single_col_by_input_type(InputTypes.ID,
                                                   column_definitions)
    target_column = utils.get_single_col_by_input_type(InputTypes.TARGET,
                                                       column_definitions)

    # Format real scalers
    real_inputs = utils.extract_cols_from_data_type(
        DataTypes.REAL_VALUED, column_definitions,
        {InputTypes.ID, InputTypes.TIME})

    #pdb.set_trace()
    real_inputs = [col for col in real_inputs if pd.api.types.is_numeric_dtype(df[col])]


    # Initialise scaler caches
    self._real_scalers = {}
    self._target_scaler = {}
    identifiers = []
    for identifier, sliced in df.groupby(id_column):
      #pdb.set_trace()
      #if identifier=='1092亿背后的流量与力量':
      #    pdb.set_trace()
      #    pass

      if len(sliced) >= self._time_steps:

        data = sliced[real_inputs].values
        targets = sliced[[target_column]].values
        self._real_scalers[identifier] \
      = sklearn.preprocessing.StandardScaler().fit(data)

        self._target_scaler[identifier] \
      = sklearn.preprocessing.StandardScaler().fit(targets)
      identifiers.append(identifier)



    # Format categorical scalers
    categorical_inputs = utils.extract_cols_from_data_type(
        DataTypes.CATEGORICAL, column_definitions,
        {InputTypes.ID, InputTypes.TIME})

    categorical_inputs_text = utils.extract_cols_from_data_type(
        DataTypes.TEXT, column_definitions,
        {InputTypes.ID, InputTypes.TIME})

    categorical_inputs += categorical_inputs_text

    categorical_scalers = {}
    num_classes = []
    label_to_integer_mappings = {}  
    for col in categorical_inputs:
      # Set all to str so that we don't have mixed integer/string columns
      srs = df[col].apply(str)
      scaler = sklearn.preprocessing.LabelEncoder().fit(srs.values)
      categorical_scalers[col] = scaler
      num_classes.append(srs.nunique())
      label_to_integer_mappings[col] = {label: idx for idx, label in enumerate(scaler.classes_)}

    self._label_to_integer_mappings = label_to_integer_mappings
    # Set categorical scaler outputs
    self._cat_scalers = categorical_scalers
    self._num_classes_per_cat_input = num_classes

    # Extract identifiers in case required
    self.identifiers = identifiers


  def set_bert_embeddings(self, df):
    # Format categorical scalers
    column_definitions = self.get_column_definition()
    categorical_inputs = utils.extract_cols_from_data_type(
        DataTypes.TEXT, column_definitions,
        {InputTypes.ID, InputTypes.TIME})

    # bert
    tokenizer = BertTokenizer.from_pretrained('./data/chinese-roberta-wwm-ext')
    model = BertModel.from_pretrained('./data/chinese-roberta-wwm-ext')
    linear = nn.Linear(768, self._hidden_layer_size)

    # bge-large
    #tokenizer = BertTokenizer.from_pretrained('../BGE/bge-large-zh-v1.5')
    #model = BertModel.from_pretrained('../BGE/bge-large-zh-v1.5')
    #linear = nn.Linear(1024, self._hidden_layer_size)

    # bge-base
    #tokenizer = BertTokenizer.from_pretrained('../BGE/bge-base-zh-v1.5')
    #model = BertModel.from_pretrained('../BGE/bge-base-zh-v1.5')
    #linear = nn.Linear(768, self._hidden_layer_size)

    # bge-small
    #tokenizer = BertTokenizer.from_pretrained('../BGE/bge-small-zh-v1.5')
    #model = BertModel.from_pretrained('../BGE/bge-small-zh-v1.5')
    #linear = nn.Linear(512, self._hidden_layer_size)

    for col in categorical_inputs:
        col_to_vector = {}
        unique_values = set()
        for index, row in df.iterrows():
            text = str(row[col])
            if text in unique_values:
                continue
            unique_values.add(text)
            if col == "introductId":
                text = str(row["introduct"])
            encoded_input = tokenizer(text, return_tensors='pt')
            with torch.no_grad():
                outputs = model(**encoded_input).last_hidden_state[:, 0, :]
            transformed_output = linear(outputs)
            if col == "introductId":
                text = str(row[col])
            index = self._label_to_integer_mappings[col][text]
            col_to_vector[index] = transformed_output
            #col_to_vector[index] = outputs
       
        sorted_indices = sorted(col_to_vector.keys())  
        sorted_vectors = [col_to_vector[idx] for idx in sorted_indices]  
        sorted_tensor = torch.stack(sorted_vectors).squeeze(1) 
                                      
        #pdb.set_trace()
        self._category_to_vectors[col] = sorted_tensor
      

  def transform_inputs(self, df):
    """Performs feature transformations.

    This includes both feature engineering, preprocessing and normalisation.

    Args:
      df: Data frame to transform.

    Returns:
      Transformed data frame.

    """

    if self._real_scalers is None and self._cat_scalers is None:
      raise ValueError('Scalers have not been set!')

    # Extract relevant columns
    column_definitions = self.get_column_definition()
    id_col = utils.get_single_col_by_input_type(InputTypes.ID,
                                                column_definitions)
    real_inputs = utils.extract_cols_from_data_type(
        DataTypes.REAL_VALUED, column_definitions,
        {InputTypes.ID, InputTypes.TIME})

    real_inputs = [col for col in real_inputs if pd.api.types.is_numeric_dtype(df[col])]

    categorical_inputs = utils.extract_cols_from_data_type(
        DataTypes.CATEGORICAL, column_definitions,
        {InputTypes.ID, InputTypes.TIME})

    categorical_inputs_text = utils.extract_cols_from_data_type(
        DataTypes.TEXT, column_definitions,
        {InputTypes.ID, InputTypes.TIME})

    categorical_inputs += categorical_inputs_text

    # Transform real inputs per entity
    df_list = []
    #pdb.set_trace()
    for identifier, sliced in df.groupby(id_col):
      #pdb.set_trace()

      # Filter out any trajectories that are too short
      if len(sliced) >= self._time_steps:
        sliced_copy = sliced.copy()
        sliced_copy[real_inputs] = self._real_scalers[identifier].transform(
            sliced_copy[real_inputs].values)
        df_list.append(sliced_copy)
      else:
          #pdb.set_trace()
          pass

    #pdb.set_trace()
    output = pd.concat(df_list, axis=0)

    # Format categorical inputs
    for col in categorical_inputs:
      string_df = output[col].apply(str)
      #pdb.set_trace()
      # if len(transformed_values) != len(output):
      #   print(f"Length mismatch for column {col}: {len(transformed_values)} != {len(output)}")
      output[col] = self._cat_scalers[col].transform(string_df)
    #pdb.set_trace()

    return output

  def transform_inputs_bert(self, df):
    """Performs feature transformations.

    This includes both feature engineering, preprocessing and normalisation.

    Args:
      df: Data frame to transform.

    Returns:
      Transformed data frame.

    """

    if self._real_scalers is None and self._cat_scalers is None:
      raise ValueError('Scalers have not been set!')

    # Extract relevant columns
    column_definitions = self.get_column_definition()
    id_col = utils.get_single_col_by_input_type(InputTypes.ID,
                                                column_definitions)
    real_inputs = utils.extract_cols_from_data_type(
        DataTypes.REAL_VALUED, column_definitions,
        {InputTypes.ID, InputTypes.TIME})

    real_inputs = [col for col in real_inputs if pd.api.types.is_numeric_dtype(df[col])]

    categorical_inputs = utils.extract_cols_from_data_type(
        DataTypes.CATEGORICAL, column_definitions,
        {InputTypes.ID, InputTypes.TIME})

    # Transform real inputs per entity
    df_list = []
    #pdb.set_trace()
    for identifier, sliced in df.groupby(id_col):
      #pdb.set_trace()

      # Filter out any trajectories that are too short
      if len(sliced) >= self._time_steps:
        sliced_copy = sliced.copy()
        sliced_copy[real_inputs] = self._real_scalers[identifier].transform(
            sliced_copy[real_inputs].values)
        df_list.append(sliced_copy)
      else:
          #pdb.set_trace()
          pass

    #pdb.set_trace()
    output = pd.concat(df_list, axis=0)

    return output

  def format_predictions(self, predictions):
    """Reverts any normalisation to give predictions in original scale.

    Args:
      predictions: Dataframe of model predictions.

    Returns:
      Data frame of unnormalised predictions.
    """

    if self._target_scaler is None:
      raise ValueError('Scalers have not been set!')

    column_names = predictions.columns

    df_list = []
    for identifier, sliced in predictions.groupby('identifier'):
      sliced_copy = sliced.copy()
      #pdb.set_trace()
      target_scaler = self._target_scaler[identifier]

      for col in column_names:
        if col not in {'forecast_time', 'identifier'}:
          sliced_copy[col] = target_scaler.inverse_transform(sliced_copy[col].values.reshape(-1,1))
      df_list.append(sliced_copy)

    output = pd.concat(df_list, axis=0)

    return output


  # Default params
  def get_fixed_params(self):
    """Returns fixed model parameters for twitters."""

    fixed_params2 = {
        'total_time_steps': 8 * 24,
        'num_encoder_steps': 7 * 24,
        'num_epochs': 100,
        'early_stopping_patience': 5,
        'multiprocessing_workers': 5
    }

    def load_yaml_params(file_path):
        with open(file_path, 'r') as file:
            params = yaml.safe_load(file)
        return params

    twitter_params = load_yaml_params('conf/twitter.yaml')
    fixed_params = {
        'total_time_steps': twitter_params['total_time_steps'],   # wake up 从yaml配置中取
        'num_encoder_steps': twitter_params['num_encoder_steps'],
        'hidden_layer_size': twitter_params['hidden_layer_size'],
        'is_bert': twitter_params['is_bert'],
        'num_epochs': 1,
        'early_stopping_patience': 5,
        'multiprocessing_workers': 5
    }
    return fixed_params

  def get_default_model_params(self):
    """Returns default optimised model parameters."""

    model_params = {
        'dropout_rate': 0.1,
        'hidden_layer_size': 160,
        'learning_rate': 0.001,
        'minibatch_size': 64,
        'max_gradient_norm': 0.01,
        'num_heads': 4,
        'stack_size': 1
    }

    return model_params

  def get_num_samples_for_calibration(self):
    """Gets the default number of training and validation samples.

    Use to sub-sample the data for network calibration and a value of -1 uses
    all available samples.

    Returns:
      Tuple of (training samples, validation samples)
    """
    return 450000, 50000

  def get_bert_embeddings(self):
    return self._category_to_vectors 
