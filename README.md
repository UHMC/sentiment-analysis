# A simple sentiment analysis pipeline for binary classification of text

Training input is via files containing tab-separated data/label pairs, one per line.

Inference is available via the get_sentiment(sentence) function whose output ranges from 0 (negative sentiment) to 1 (positive sentiment).

Some code based on an official tutorial here: https://www.tensorflow.org/tutorials/keras/basic_text_classification

Datasets sourced from: https://www.kaggle.com/marklvl/sentiment-labelled-sentences-data-set/version/2#
