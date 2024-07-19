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
      total_num_predictions = SIX + EIGHT
      perplexity_values = []
      for sen in input_test_sentences:
        token_list = self.tokenizer.texts_to_sequences([sen])[0]
        log_prob_sum = 0.0
        total_log_prob_sum = 0.0      
        for _ in range(NUM_NEXT_WORDS):
          token_list_padded = pad_sequences([token_list], maxlen=self.max_sequence_len-1, padding='pre')
          predicted_probs = self.model.predict(token_list_padded, verbose=0)[0]
          next_word_prob = predicted_probs[np.argmax(predicted_probs)]
          log_prob_sum += np.log(next_word_prob)
        total_log_prob_sum += log_prob_sum
        average_log_prob = total_log_prob_sum / total_num_predictions
        perplexity = np.exp(-average_log_prob)
        perplexity_values.append(perplexity)
      return perplexity_values
    except Exception as e:
      print(f"Error in calculating perplexity: {e}")
      return None
