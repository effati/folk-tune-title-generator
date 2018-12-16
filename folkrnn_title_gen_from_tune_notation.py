import codecs
import difflib

from keras.models import Model
from keras.layers import Input, LSTM, Dense
import numpy as np

batch_size = 64
epochs = 100
latent_dim = 256
max_string = 200

data_path = 'data/tunes.txt'
test_set_output_path = 'data/test_set.txt'

input_texts = []
target_texts = []
existing_titles = set()
tunes = {}
input_characters = set()
target_characters = set()
with codecs.open(data_path, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')
chunk_size = 5
for i in xrange(0, len(lines)-1, chunk_size):
    chunk = lines[i:i + chunk_size]

    target_text = chunk[0].split('T:')[1].strip()
    input_text = ''.join(chunk[1:4])[:max_string].strip()
    if target_text in existing_titles:
        distance = difflib.SequenceMatcher(None, tunes[target_text].lower(), input_text.lower()).ratio()
        if distance > 0.6:
            continue
    tunes[target_text] = input_text
    existing_titles.add(target_text)
    target_text = '\t' + target_text + '\n'
    input_texts.append(input_text)
    target_texts.append(target_text)
    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)
    for char in target_text:
        if char not in target_characters:
            target_characters.add(char)

rng_state = np.random.get_state()
np.random.shuffle(input_texts)
np.random.set_state(rng_state)
np.random.shuffle(target_texts)

test_ratio = int(len(input_texts) * 0.9)
test_set_input = input_texts[test_ratio:]
test_set_target = target_texts[test_ratio:]

with codecs.open(test_set_output_path, 'w+', encoding='utf-8') as f:
    for tune in range(len(test_set_input)):
        f.write("T:{}\n".format(test_set_target[tune].strip()))
        f.write("{}".format(test_set_input[tune].strip()))

input_texts = input_texts[:test_ratio]
target_texts = target_texts[:test_ratio]

input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

input_token_index = dict(
    [(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict(
    [(char, i) for i, char in enumerate(target_characters)])
#
# token_index_target_path = "data/token_index.txt"
# with codecs.open(token_index_target_path, 'w+', encoding='utf-8') as f:
#     json.dump(target_token_index, f)
# print("Token index successfully dumped into '{}'".format(token_index_target_path))

encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
    dtype='float32')
decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')
decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.
    for t, char in enumerate(target_text):
        decoder_input_data[i, t, target_token_index[char]] = 1.
        if t > 0:
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.

encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(None, num_decoder_tokens))

decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)
model.save('title_tune_gen_v2.h5')
