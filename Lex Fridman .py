#!/usr/bin/env python
# coding: utf-8

# In[12]:


data =open ("tweets.txt").read()


# We've imported the data already, so lets start by doing a bit of visualization. There are a couple of ways I'd like to look at this first is through a word cloud and then the second is going to be a count of the most used words. 

# In[13]:


import matplotlib.pyplot as plt
from wordcloud import WordCloud

word_cloud2 = WordCloud(collocations = False, background_color = 'white').generate(data)
plt.imshow(word_cloud2, interpolation='bilinear')

plt.axis("off")

plt.show()


# Okay so I'm already starting to notice a problem but lets address that in a minute

# In[14]:


with open('tweets.txt', 'r') as f:
    text = f.read()
    words = text.split()

# frequency of words
word_counts = {}
for word in words:
    if word in word_counts:
        word_counts[word] += 1
    else:
        word_counts[word] = 1

# sort the words 
sorted_word_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

# 30 most common words
most_common_words = sorted_word_counts[:30]

# words and frequencies into lists to plot
words = [x[0] for x in most_common_words]
frequencies = [x[1] for x in most_common_words]
plt.bar(words, frequencies)
plt.xticks(rotation=90)
plt.show()


# So, the problem here is pretty clear. We're taking too many words that have no real meaning to the overall show. So lets remove the 1000 most common words and redo this. 
# 
# https://en.wiktionary.org/wiki/Wiktionary:Frequency_lists :For the 100 most common words
# 

# In[8]:


import csv

with open('100.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    words_to_remove = [row[0] for row in reader]

with open('tweets.txt', 'r', encoding='utf-8') as f:
    text = f.read()

words = text.split()

filtered_words = [word for word in words if word not in words_to_remove]

filtered_text = ' '.join(filtered_words)

with open('filtered_text.txt', 'w', encoding='utf-8') as f:
    f.write(filtered_text)


# Sick. So now lets using that new file filtered_text use that for our graphs again. 

# In[9]:


data =open ("filtered_text.txt").read()

word_cloud2 = WordCloud(collocations = False, background_color = 'white').generate(data)
plt.imshow(word_cloud2, interpolation='bilinear')

plt.axis("off")

plt.show()


# That looks better but lets see what our most common words look like I still see a few key words that may cause issues. 

# In[11]:


with open('filtered_text.txt', 'r') as f:
    text = f.read()
    words = text.split()

# frequency of words
word_counts = {}
for word in words:
    if word in word_counts:
        word_counts[word] += 1
    else:
        word_counts[word] = 1

# sort the words 
sorted_word_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

# 30 most common words
most_common_words = sorted_word_counts[:30]

# words and frequencies into lists to plot
words = [x[0] for x in most_common_words]
frequencies = [x[1] for x in most_common_words]
plt.bar(words, frequencies)
plt.xticks(rotation=90)
plt.show()


# ok so still no dice it looks like we may need to filter for the 1000 most used words. This is pushing us into some level of risk since the 1000 most common words do include things such as magnets, old, men, act, plant cover etc. I'm also going to manually include Uh and Um to this list to filter out vocal ticks that the scraper caught, and at this point it may be better to start manually filtering specifically based on what the most common words are, and that would be a valid approach as well.
# 

# In[15]:


with open('1000.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    words_to_remove = [row[0] for row in reader]

with open('filtered_text.txt', 'r', encoding='utf-8') as f:
    text = f.read()

words = text.split()

filtered_words = [word for word in words if word not in words_to_remove]

filtered_text = ' '.join(filtered_words)

with open('filtered_text2.txt', 'w', encoding='utf-8') as f:
    f.write(filtered_text)


# So lets try this again Starting with our bar graphs
# 

# In[16]:


with open('filtered_text2.txt', 'r') as f:
    text = f.read()
    words = text.split()

# frequency of words
word_counts = {}
for word in words:
    if word in word_counts:
        word_counts[word] += 1
    else:
        word_counts[word] = 1

# sort the words 
sorted_word_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

# 30 most common words
most_common_words = sorted_word_counts[:30]

# words and frequencies into lists to plot
words = [x[0] for x in most_common_words]
frequencies = [x[1] for x in most_common_words]
plt.bar(words, frequencies)
plt.xticks(rotation=90)
plt.show()


# Right... So lets start removing the most commmon words based on specifically the check we're doing now. We'll go back to using filtered_text.txt that removed the 100 most common words and not this since as mentioned we are experiencing some risk here. I'm also going to remove anything that's not a space or a char below since I hadn't done that above, and then recheck the chart encase this solves the issue.
# 

# In[17]:


with open('filtered_text.txt', 'r') as file:
  lines = file.readlines()


with open('filtered_text.txt', 'w') as file:
  for line in lines:
    for char in line:
      if char.isalpha() or char.isspace():
        file.write(char)


# In[19]:


with open('filtered_text.txt', 'r') as f:
    text = f.read()
    words = text.split()

# frequency of words
word_counts = {}
for word in words:
    if word in word_counts:
        word_counts[word] += 1
    else:
        word_counts[word] = 1

# sort the words 
sorted_word_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

# 30 most common words
most_common_words = sorted_word_counts[30:61]

# words and frequencies into lists to plot
words = [x[0] for x in most_common_words]
frequencies = [x[1] for x in most_common_words]
plt.bar(words, frequencies)
plt.xticks(rotation=90)
plt.show()


# This didn't solve it so now lets just take a different slice. 

# In[21]:


# 30 most common words
most_common_words = sorted_word_counts[30:61]

# words and frequencies into lists to plot
words = [x[0] for x in most_common_words]
frequencies = [x[1] for x in most_common_words]
plt.bar(words, frequencies)
plt.xticks(rotation=90)
plt.show()


# That's actually much better so lets do this save process but change out most_common_words 

# In[22]:


#  most common words
most_common_words = sorted_word_counts[61:92]

# words and frequencies into lists to plot
words = [x[0] for x in most_common_words]
frequencies = [x[1] for x in most_common_words]
plt.bar(words, frequencies)
plt.xticks(rotation=90)
plt.show()


# Lets try one more time and increase the slice this time just to save some time
# 

# In[29]:


# most common words
most_common_words = sorted_word_counts[93:144]

# words and frequencies into lists to plot
words = [x[0] for x in most_common_words]
frequencies = [x[1] for x in most_common_words]
plt.bar(words, frequencies)
plt.xticks(rotation=90)
plt.show()


# That actually looks a lot better, but lets reduce it back to about 30.  
# 
# 

# In[32]:


# most common words
most_common_words = sorted_word_counts[111:144]

# words and frequencies into lists to plot
words = [x[0] for x in most_common_words]
frequencies = [x[1] for x in most_common_words]
plt.bar(words, frequencies)
plt.xticks(rotation=90)
plt.show()


# And then one more time here for the next set
# 

# In[33]:


# most common words
most_common_words = sorted_word_counts[144:175]

# words and frequencies into lists to plot
words = [x[0] for x in most_common_words]
frequencies = [x[1] for x in most_common_words]
plt.bar(words, frequencies)
plt.xticks(rotation=90)
plt.show()


# Lets check the most commmon words now just by printing them.
# 

# In[34]:


with open('filtered_text.txt', 'r') as f:
  text = f.read()

words = text.split()

from collections import Counter
word_counts = Counter(words)

most_common_words = word_counts.most_common(400)


# In[35]:


print(most_common_words)


# We could use the 1000 most common words but as mentioned prior we could run into some risks as we see here that some of those are key areas for this podcast
# 
# 

# In[44]:


import codecs

with codecs.open('tweets.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    
import re

# Lowercase the text and remove punctuation
text = text.lower()
text = re.sub(r'[^\w\s]', '', text)

# Split the text into words
words = text.split()

from sklearn.feature_extraction.text import TfidfVectorizer

# Create a TfidfVectorizer object
vectorizer = TfidfVectorizer()

# Compute the TF-IDF of the text
tfidf = vectorizer.fit_transform([text])

# Get the words with the highest TF-IDF scores
scores = zip(vectorizer.get_feature_names(), tfidf.data)
sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)

# Print the top 100 key topics
for word, score in sorted_scores[:100]:
    print(word)


# and then for fun lets do some training for language modeling. I'm not going to run this because it will take too long frankly. convert this to a python cell and run it if curious. 
# 

# # Open the text file and read its contents into a string
# with open('text_file.txt', 'r') as f:
#     text = f.read()
# 
# # Preprocess the text by lowercasing, removing punctuation, and splitting the text into individual words
# import re
# 
# # Lowercase the text and remove punctuation
# text = text.lower()
# text = re.sub(r'[^\w\s]', '', text)
# 
# # Split the text into words
# words = text.split()
# 
# # Tokenize the text
# tokenizer = Tokenizer(num_words=MAX_WORDS)
# tokenizer.fit_on_texts([text])
# sequences = tokenizer.texts_to_sequences([text])
# 
# # Flatten the list of sequences into a single list of words
# words = [item for sublist in sequences for item in sublist]
# 
# # Split the list of words into sequences of SEQUENCE_LENGTH words
# sequences = []
# for i in range(SEQUENCE_LENGTH, len(words)):
#     # Get the current sequence of words
#     seq = words[i-SEQUENCE_LENGTH:i]
#     # Append the sequence to the list of sequences
#     sequences.append(seq)
# 
# # Convert the sequences into a NumPy array
# import numpy as np
# sequences = np.array(sequences)
# 
# # Split the sequences into input and output
# X = sequences[:,:-1]
# y = sequences[:,-1]
# 
# # Fit the model to the data
# model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE)
# 
# import tensorflow as tf
# 
# # Set the number of words in the vocabulary
# VOCAB_SIZE = MAX_WORDS + 1
# 
# # Set the number of dimensions for the embedding layer
# EMBEDDING_DIM = 100
# 
# # Set the number of units in the LSTM layer
# LSTM_UNITS = 128
# 
# # Set the batch size
# BATCH_SIZE = 64
# 
# # Set the number of epochs
# EPOCHS = 10
# 
# # Define the model
# model = tf.keras.Sequential()
# model.add(tf.keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=SEQUENCE_LENGTH))
# model.add(tf.keras.layers.LSTM(LSTM_UNITS, return_sequences=True))
# model.add(tf.keras.layers.LSTM(LSTM_UNITS))
# model.add(tf.keras.layers.Dense(VOCAB_SIZE, activation='softmax'))
# 
# # Compile the model
# model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# 
# # Train the model
# model.fit(sequences, epochs=EPOCHS, batch_size=BATCH_SIZE)
# 

# Something to note here is i did make a mistake I should have preprocessed the data way earlier I wasn't being attentive for it as I thought the place I had collected this data had removed all unique characters. That mistake cost me some time and it was a mistake i should have caught.
# 

# In[ ]:




