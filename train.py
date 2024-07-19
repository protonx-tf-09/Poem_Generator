from data.dataset import Dataset
from model.model import create_model
from model.generate import GeneratePoem
from model.evaluate import EvaluateModel
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from constant import SIX, EIGHT, NUM_NEXT_WORDS

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--epochs", default=200, type=int)
parser.add_argument("--embedding-size", default=100, type=int)
parser.add_argument("--size-train", default=3240, type=int)
args = parser.parse_args()

# Load and prepare the dataset
url = "https://github.com/duyet/truyenkieu-word2vec/blob/master/truyen_kieu_data.txt"
dataset = Dataset()
dataset.download_from_github(url)
dataset.load_data()

# Split data and prepare training and test sequences
trainset, testset = dataset.split_dataset(args.size_train)
train_input_sentences = dataset.change_sentences_format(trainset)
test_input_sentences = dataset.prepare_test_data(testset)
tokenizer = dataset.build_tokenizer(train_input_sentences)
input_sequences, labels, max_sequence_len, vocab_size = dataset.tokenize(train_input_sentences)

# Build and train the model
model = create_model(vocab_size=vocab_size, embedding_dim=args.embedding_size, input_length=max_sequence_len - 1)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
callbacks = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)
model.fit(input_sequences, labels, epochs=args.epochs, callbacks=[callbacks])

# Evaluate the model using perplexity
evaluator = EvaluateModel(model, tokenizer, max_sequence_len)
perplexity = evaluator.calculate_perplexity(test_input_sentences)
print("Model Perplexity:", np.mean(perplexity))

# Generate poems
gen_poem = GeneratePoem(model, tokenizer, max_sequence_len, NUM_NEXT_WORDS)
poem_out = gen_poem.generate_poem(test_input_sentences)

# Print the result predicted and perplexity
for idx, sen in enumerate(test_input_sentences):
  print(f'Perplexity: {perplexity[idx]} cho câu "{sen}" --> "{poem_out[idx]}"')
