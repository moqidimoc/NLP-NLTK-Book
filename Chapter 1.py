# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 11:42:09 2021

@author: Asus
"""

import nltk
from nltk.book import *

# CONCORDANCE
text1.concordance('monstrous')

# SIMILAR
text1.similar('monstrous')

text1.common_contexts(['monstrous', 'loving'])

# GENERATE
text4.generate()

# FREQUENCY DICTIONARIES
fdist1 = FreqDist(text1)
fdist2 = FreqDist(text2)
fdist7 = FreqDist(text7)
 	
fdist2.plot(50, cumulative=True)

nltk.chat.chatbots()

# Using list addition, and the set and sorted operations, compute the vocabulary of the sentences sent1 ... sent8.
sent = sent1 + sent2 + sent3 + sent4 + sent5 + sent6 + sent7 + sent8
sorted(set(sent))

 	
len(sorted(set(w.lower() for w in text1))) # in this case we avoid repetitions like 'This' and 'this'
len(sorted(w.lower() for w in set(text1)))

# Write the slice expression that extracts the last two words of text2.
text2[-2:]

# Find all the four-letter words in the Chat Corpus (text5). With the help of a frequency distribution (FreqDist), show
# these words in decreasing order of frequency.
fdist5_four = FreqDist(w for w in text5 if len(w)==4) # this are the word frequencies
fdist5_four.plot()

# Review the discussion of looping with conditions in 4. Use a combination of for and if statements to loop over the words
# of the movie script for Monty Python and the Holy Grail (text6) and print all the uppercase words, one per line.
for w in set(text6):
    if w.isupper():
        print(w, end=" ")
        
# Write expressions for finding all words in text6 that meet the conditions listed below. The result should be in the form
# of a list of words: ['word1', 'word2', ...].
    # a. Ending in ise
for w in set(text6):
    if w.endswith('ise'): print(w)
    # b.Containing the letter z
for w in set(text6):
    if 'z' in w.lower(): print(w)
    # c. Containing the sequence of letters pt
for w in set(text6):
    if 'pt' in w.lower(): print(w)
    # d. Having all lowercase letters except for an initial capital (i.e., titlecase)
for w in set(text6):
    if w.istitle(): print(w)
    
# Define sent to be the list of words ['she', 'sells', 'sea', 'shells', 'by', 'the', 'sea', 'shore']. Now write code to
# perform the following tasks:
sent = ['she', 'sells', 'sea', 'shells', 'by', 'the', 'sea', 'shore']
    # a. Print all words beginning with sh
for w in sent:
    if w.startswith('sh'): print(w)
    # b. Print all words longer than four characters
for w in sent:
    if len(w)==4: print(w)
    
# What does the following Python code do? sum(len(w) for w in text1) Can you use it to work out the average word length of
# a text?
            # This piece of code counts the number of characters in text1 (Moby Dick). You can use it to calculate
            # the average word length of a text like this:
n_characters = sum(len(w) for w in text1)
n_words = len(text1)
avg_word_length = n_characters/n_words # 3.830411128023649

# Define a function called vocab_size(text) that has a single parameter for the text, and which returns the vocabulary size
# of the text.
def vocab_size(text):
    return len(set(text)) # the vocabulary is composed by the number of unique words

# Define a function percent(word, text) that calculates how often a given word occurs in a text, and expresses the result
# as a percentage.
def percent(word, text):
    return 100*text.count(word)/len(text)