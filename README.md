# What

The sequence-to-sequence model is a powerful tool to translate between any two sequences of arbitrary length and content. This project experimented the possibility to translate between Irish traditional folk tunes and their titles. The result was a model that could decode new untitled tunes into a comprehensible title. The model was not very successful in translating tunes back to their original title.

# How

## Learning a new model
Use `folkrnn_title_gen_from_tune_notation.py` to generate a new model. Make sure to change the name of the model inside the code if you don't want to replace the model already in the folder. The script will need the set of tunes in `data/tunes.txt`. From it, it will produce a test set at `data/test_set.txt`.

## Evaulating the model
Using the test set `data/test_set.txt`, you can run the script `lstm_seq2seq_restore_test_samples.txt` to try to recreate the titles of already titled tunes, using only their tune notation. This is capped at 2000 tunes for now, but feel free to go all the way. It will write out all the recreations to a file, together with the Levenshtein distance between the actual sentence and the decoded one. It will then print out the mean of all those distances.

## Generating new titles
To see what the results are with the untitled samples, use `lstm_seq2seq_restore_new_samples.py`. This will take in the scraped samples over at `data/scraped_samples_full.txt` and try to create new titles for it, using the model. It will then write out the results to a file.
