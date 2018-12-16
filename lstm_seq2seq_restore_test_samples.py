from __future__ import print_function

import codecs
import difflib

from keras.models import Model, load_model
from keras.layers import Input
import numpy as np

batch_size = 64  # Batch size for training.
epochs = 100  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 10000  # Number of samples to train on.
obligatory_output_token_size = 47
data_path = 'data/tunes.txt'

input_texts_full = []
target_texts_full = []
input_characters = set()
target_characters = set()
lines = codecs.open(data_path, 'r', encoding='utf-8').read().split('\n')
chunk_size = 5
for i in xrange(0, len(lines)-1, chunk_size):
    chunk = lines[i:i + chunk_size]

    target_text = chunk[0].split('T:')[1].strip()
    input_text = ''.join(chunk[1:4])[:200].strip()

    target_text = '\t' + target_text + '\n'
    input_texts_full.append(input_text)
    target_texts_full.append(target_text)

    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)
    for char in target_text:
        if char not in target_characters:
            target_characters.add(char)

rng_state = np.random.get_state()
np.random.shuffle(input_texts_full)
np.random.set_state(rng_state)
np.random.shuffle(target_texts_full)

data_ratio = int(len(target_texts_full) * 0.8)

target_texts = list(target_texts_full)[:data_ratio]
input_texts = list(input_texts_full)[data_ratio:]

input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

print('Number of samples:', len(input_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)

input_token_index = dict(
    [(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict(
    [(char, i) for i, char in enumerate(target_characters)])

encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
    dtype='float32')

for i, input_text in enumerate(input_texts):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.

model = load_model('title_tune_gen.h5')

encoder_inputs = model.input[0]   # input_1
encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output   # lstm_1
encoder_states = [state_h_enc, state_c_enc]
encoder_model = Model(encoder_inputs, encoder_states)

decoder_inputs = model.input[1]   # input_2
decoder_state_input_h = Input(shape=(latent_dim,), name='input_3')
decoder_state_input_c = Input(shape=(latent_dim,), name='input_4')
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_lstm = model.layers[3]
decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h_dec, state_c_dec]
decoder_dense = model.layers[4]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

reverse_input_char_index = dict(
    (i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())


def decode_sequence(input_seq):
    states_value = encoder_model.predict(input_seq)

    target_seq = np.zeros((1, 1, num_decoder_tokens))
    target_seq[0, 0, target_token_index['\t']] = 1.

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        if (sampled_char == '\n' or
           len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        states_value = [h, c]

    return decoded_sentence


with open('output/lstm_test_titles.txt', 'w+') as f:
    distances = []
    total_seqs = 2000
    for seq_index in range(total_seqs):
        input_seq = encoder_input_data[seq_index: seq_index + 1]
        try:
            decoded_sentence = decode_sequence(input_seq).strip()
            f.write('Decoded sentence: {}\n'.format(decoded_sentence))

            actual_sentence = list(target_texts_full)[data_ratio + seq_index].strip()
            distance = difflib.SequenceMatcher(None, decoded_sentence, actual_sentence).ratio()
            distances.append(distance)

            f.write('Actual sentence: {}\n'.format(actual_sentence))

            f.write('{}'.format(distance))

            f.write('{}\n\n'.format('-' * 50))

            if distance > 0.75:
                print('Actual:', actual_sentence)
                print('Decoded:', decoded_sentence)
                print('{}\n'.format(distance))

                print("{}/{} done\n".format(seq_index, total_seqs))
        except ValueError:
            pass
    print('Mean:', np.mean(np.array(distances)))
