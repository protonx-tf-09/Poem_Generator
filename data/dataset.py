import requests
import re
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from constant import SIX, EIGHT

class Dataset():
  def __init__(self):
    self.dataPath = None
    self.dataset = []
    self.tokenizer = None


  def download_from_github(self, url):
    '''
    Download the data from a github url and save it to a file
    '''
    url = url.replace('blob/', '')
    url = url.replace('github.com', 'raw.githubusercontent.com')
    try:
      response = requests.get(url)
      response.raise_for_status()
      self.dataPath = './truyen_kieu_data.txt'
      with open('truyen_kieu_data.txt', 'wb') as f:
        f.write(response.content)
      print('Download complete!')
    except Exception as e:
      print(f"Error in download data function: {e}")
      return None

  def load_data(self):
    '''
    Load and preprocess the data
    '''
    try:
      with open(self.dataPath, 'r') as f:
        dataset = f.read()

      dataset = re.sub(r'^\s+|[\d\.\,\!\?\'\"]*|\s+$', '', dataset.lower())
      dataset = dataset.split('\n')
      dataset = [d.strip() for d in dataset]

      for idx, row in enumerate(dataset):
        if len(row.split()) > EIGHT+SIX: # Find and correct manually if the length exceeds that of two sentences.
          print(f'index row: {idx}, content: {row}')
        elif len(row.split()) > EIGHT:
          words = row.split()
          if len(dataset[idx-1].split()) == SIX:
            first_part, second_part = words[:EIGHT], words[EIGHT:]
          else:
            first_part, second_part = words[:SIX], words[SIX:]
          dataset[idx] = [' '.join(first_part), ' '.join(second_part)]
        elif len(row.split()) < SIX: # Check if row is empty
          del dataset[idx]

      for item in dataset:
        if isinstance(item, list):
          self.dataset.extend(item)
        else:
          self.dataset.append(item)
      print('Preprocessing complete!')
      return self.dataset
    except Exception as e:
      print(f"Error in preprocessing data: {e}")
      return None

  def change_sentences_format(self, train_data):
    '''
    Combine each pair of sentences 6-8 into one input sentence
    '''
    input_sentence = []
    for i in range(0, len(train_data), 2):
      input_sentence.append(train_data[i] + " " + train_data[i+1])
    return input_sentence

  def split_dataset(self, n_train):
    '''
    Split the dataset into train and test set
    '''
    train_data = self.dataset[:n_train]
    test_data = self.dataset[n_train:]
    return train_data, test_data

  def build_tokenizer(self, input_sentence):
    '''
    Build the tokenizer
    '''
    try:
      self.tokenizer = Tokenizer()
      self.tokenizer.fit_on_texts(input_sentence)
      return self.tokenizer
    except Exception as e:
      print(f"Error in building tokenizer: {e}")
      return None

  def tokenize(self, input_sentence, padding='pre'):
    '''
    Tokenize the input sequences
    '''
    try:
      total_words = len(self.tokenizer.word_index) + 1
      input_sequences = []
      for line in input_sentence:
        token_list = self.tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
          n_gram_sequence = token_list[:i+1]
          input_sequences.append(n_gram_sequence)
      max_sequence_len = max([len(x) for x in input_sequences])
      input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding=padding))
      predictors, label = input_sequences[:,:-1],input_sequences[:,-1]
      labels = to_categorical(label, num_classes=total_words)
      return predictors, labels, max_sequence_len, total_words
    except Exception as e:
      print(f"Error in tokenizing data: {e}")
      return None, None, None, None

  def prepare_test_data(self, test_data):
    '''
    Prepare the test data, Take only sentence 6 as input_test_sentences.
    '''
    input_test_sentences = []
    for i in range(0, len(test_data), 2):
      input_test_sentences.append(test_data[i])
    return input_test_sentences