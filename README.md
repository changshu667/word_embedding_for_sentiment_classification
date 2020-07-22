# Sentiment-Specific Word Embedding for Twitter Sentiment Classification
This project is trying to generate word embedding using combined sentiment and sytactic information. Previous approach only considered sytactic information, this enhanced word embedding could improve the sentiment classification performance.
## Problem
This project generated enhanced word embedding containing both syntactic and sentiment information to improve Twitter sentiment classification performance. The traditional approach only used syntactic-specific word embedding leads to suboptimal performance in the sentiment classification. With only syntactic-specific word embeddings, words with similar syntactic meaning but opposite sentiment meaning would have similar representation extracted, which makes it difficult to distinguish for the classification network.
## Approach
In additional to the traditional embedding network training method which only learns syntactic meaning by corrupting the normal text and making neural network determine whether the input text has been corrupted. I included semantic label into the training process. Therefore, both the syntactic and sentiment information can be learnt. Besides, after analyzing the characteristics of the tweets, I designed a specific text preprocessing pipeline for better data cleaning. Finally, I compared the classification performance in sentiment classification with the traditional approach and the new approach. Outcome: Using combined syntactic and sentiment word embedding rather than only syntactic word embedding improves the sentiment classification accuracy.
## Key Techniques
NLP, Word Embedding, Sentiment Classification, Deep Learning, Neural Network, PyTorch
## Language
Python

