import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from constant import SIX, EIGHT, NUM_NEXT_WORDS

class GeneratePoem():
  def __init__(self, model, tokenizer, max_sequence_len, next_words):
    self.model = model
    self.tokenizer = tokenizer
    self.max_sequence_len = max_sequence_len
    self.next_words = next_words

  def generate_poem(self, input_test_sentences):
    '''
    Generate the poem based on the seed sentences provided.
    '''
    try:
      text_out = []
      for idx, poem in enumerate(input_test_sentences):
        for _ in range(self.next_words):
          token_list = self.tokenizer.texts_to_sequences([poem])[0]
          token_list = pad_sequences([token_list], maxlen=self.max_sequence_len-1, padding='pre')
          predicted = self.model.predict(token_list, verbose=0)
          predicted_index = np.argmax(predicted, axis=1)[0]
          output_word = ''
          if predicted_index in self.tokenizer.index_word:
            output_word = self.tokenizer.index_word[predicted_index]
            poem += ' ' + output_word
        words = poem.split()
        # text_in = " ".join(words[:SIX])
        text_out.append(" ".join(words[SIX:]))
    
      return text_out
    except Exception as e:
      print(f"Error in generating poem: {e}")
      return None, None
