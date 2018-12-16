import codecs

from keras.models import Model, load_model
from keras.layers import Input
import numpy as np

batch_size = 64  # Batch size for training.
epochs = 100  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 10000  # Number of samples to train on.
obligatory_output_token_size = 47
data_path = 'data/tunes.txt'

input_texts = []
target_texts = []
input_characters = set()
target_characters = set()
lines = codecs.open(data_path, 'r', encoding='utf-8').read().split('\n')
chunk_size = 5
for i in xrange(0, len(lines)-1, chunk_size):
    chunk = lines[i:i + chunk_size]
    target_text = chunk[0].split('T:')[1].strip()
    target_text = '\t' + target_text + '\n'
    target_texts.append(target_text)

    for char in target_text:
        if char not in target_characters:
            target_characters.add(char)


def get_samples(i):
        temp = ''.join(lines[i:i + chunk_size])
        input_text = temp[:min(200, len(temp))].strip()

        for char in input_text:
            if char not in input_characters:
                if len(input_characters) == obligatory_output_token_size:
                    return
                input_characters.add(char)
        input_texts.append(input_text)


lines = codecs.open("data/scraped_samples_full.txt", 'r', encoding='utf-8').read().split('\n')
chunk_size = 4
for i in xrange(0, len(lines) - 1, chunk_size):
    get_samples(i)
    if len(input_characters) >= obligatory_output_token_size:
        break

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


with open('output/lstm_output_report.txt', 'w+') as f:
    for seq_index in range(len(input_texts)):
        input_seq = encoder_input_data[seq_index: seq_index + 1]
        try:
            decoded_sentence = decode_sequence(input_seq)
#            print('-')
#            print('Input sentence:', input_texts[seq_index])
#            print('Decoded sentence:', decoded_sentence)
#            f.write('Input sentence: {}\n'.format(input_texts[seq_index]))
#            f.write('Decoded sentence: {}\n'.format(decoded_sentence))
            f.write('{}\\\\'.format('-' * 50))
            f.write("Title: {}\\\\".format(decoded_sentence.strip()))
            f.write("{}\n\\\\".format(input_texts[seq_index]))
        except ValueError:
            pass
