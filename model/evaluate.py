import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from constant import SIX, EIGHT, NUM_NEXT_WORDS

class EvaluateModel():
  def __init__(self, model, tokenizer, max_sequence_len):
    self.model = model
    self.tokenizer = tokenizer
    self.max_sequence_len = max_sequence_len

  def calculate_perplexity(self, input_test_sentences):
    '''
    Calculate the perplexity of the model on the given sentences.
    '''
    try:
      perplexity_values = []

      for sen in input_test_sentences:
          token_list = self.tokenizer.texts_to_sequences([sen])[0]
          log_prob_sum = 0.0
          num_predictions = len(token_list) - 1

          for i in range(1, len(token_list)):
              token_list_padded = pad_sequences([token_list[:i]], maxlen=self.max_sequence_len-1, padding='pre')
              predicted = self.model.predict(token_list_padded, verbose=0)[0]
              next_word_prob = predicted[token_list[i]]
              log_prob_sum += np.log(next_word_prob)

          perplexity = np.exp(-log_prob_sum / num_predictions)
          perplexity_values.append(perplexity)

      return perplexity_values
    except Exception as e:
      print(f"Error in calculating perplexity: {e}")
      return None