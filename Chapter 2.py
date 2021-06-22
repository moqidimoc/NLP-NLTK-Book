# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 12:06:37 2021

@author: Asus
"""
import nltk
from nltk.corpus import gutenberg

# nltk.corpus.gutenberg.fileids() # to see what corpus are in this library

emma = gutenberg.words('austen-emma.txt') # Here we set Emma, from Austen, to this variable

# Here we print some general information about all the corpus in this corpora	
for fileid in gutenberg.fileids():
    num_chars = len(gutenberg.raw(fileid))
    num_words = len(gutenberg.words(fileid))
    num_sents = len(gutenberg.sents(fileid))
    num_vocab = len(set(w.lower() for w in gutenberg.words(fileid)))
    print("The corpus {} has {} characters per word, {} words per sentence, and each word appears {} times on average\n".format(
        fileid, round(num_chars/num_words), round(num_words/num_sents), round(num_words/num_vocab)))

# We can select only sentences:
macbeth_sentences = gutenberg.sents('shakespeare-macbeth.txt')
print(macbeth_sentences)
longest_len = max(len(s) for s in macbeth_sentences)
print([s for s in macbeth_sentences if len(s) == longest_len]) # longest sentences

# There are also more informal texts (Firefox discussion forum, conversations overheard in New York, the movie script of Pirates of the Carribean, 
# personal advertisements, and wine reviews)
from nltk.corpus import webtext
for fileid in webtext.fileids():
    print(fileid, webtext.raw(fileid)[:65], '...')

# Or a corpus of instant messaging chat sessions, originally collected by the Naval Postgraduate School for research on automatic detection of Internet predators
from nltk.corpus import nps_chat
chatroom = nps_chat.posts('10-19-20s_706posts.xml')
chatroom[123]

# Or the Brown Corpus, which contains text from 500 sources, and the sources have been categorized by genre, such as news, editorial, and so on.
from nltk.corpus import brown
brown.categories()
news_text = brown.words(categories='news')

# This corpora is really useful to compare texts depending on its style (stylistics)
fdist = nltk.FreqDist(w.lower() for w in news_text)
modals = ['can', 'could', 'may', 'might', 'must', 'will']
for m in modals:
    print(m + ':', fdist[m], end=' ')

# And for the 'wh' words (what, when, who, where, why) and the category of mystery for example:
mystery_text = brown.words(categories='mystery')
fdist = nltk.FreqDist(w.lower() for w in mystery_text)
wh = ['what', 'when', 'where', 'who', 'why']
for m in wh:
    print(m + ':', fdist[m], end=' ')

# A conditional frequency distribution is a collection of frequency distributions, each one for a different "condition". The condition will often be the
# category of the text. Whereas FreqDist() takes a simple list as input, ConditionalFreqDist() takes a list of pairs.
# For each genre [2], we loop over every word in the genre [3], producing pairs consisting of the genre and the word [1]:

# Now, we obtain counts for each genre of interest
cfd = nltk.ConditionalFreqDist(
          (genre, word) # [1]
          for genre in brown.categories() # [2]
          for word in brown.words(categories=genre)) # [3]

# A ConditionalFreqDist provides some useful methods for tabulation and plotting. In the plot() and tabulate() methods, we can optionally specify which 
# conditions to display with a conditions=parameter, when we omit it, we get all the conditions (plot). Similarly, we can limit the samples to display with a 
# samples= parameter:
    # TABULATING: creating tables
genres = ['news', 'religion', 'hobbies', 'science_fiction', 'romance', 'humor']
modals = ['can', 'could', 'may', 'might', 'must', 'will']
cfd.tabulate(conditions=genres, samples=modals)
cfd.tabulate(conditions=genres, samples=wh)

    # PLOTTING
# We also have texts from the inaugural address corpus
from nltk.corpus import inaugural
cfd = nltk.ConditionalFreqDist(
          (target, fileid[:4]) # the first four elements of fileids are the years
          for fileid in inaugural.fileids()
          for w in inaugural.words(fileid)
          for target in ['america', 'citizen'] # the words we want to pay attention to
          if w.lower().startswith(target))
cfd.plot() # Evolution over time of the words 'America' and 'citizen'

# Some of the Corpora and Corpus Samples Distributed with NLTK: For information about downloading and using them, please consult the NLTK website.
# https://www.nltk.org/book/ch02.html

# There is also texts in other languages:
spanish = nltk.corpus.cess_esp.words()
floresta = nltk.corpus.floresta.words()
indian = nltk.corpus.indian.words('hindi.pos')
udhr_fileids = nltk.corpus.udhr.fileids() # universal declaration of human rights in over 300 languages

# Let's use a conditional frequency distribution to examine the differences in word lengths for a selection of languages included in the udhr corpus.
from nltk.corpus import udhr
languages = ['Chickasaw', 'English', 'German_Deutsch', 'Greenlandic_Inuktikut', 'Hungarian_Magyar', 'Ibibio_Efik', 'Spanish_Espanol']
cfd = nltk.ConditionalFreqDist(
        (lang, len(word))
        for lang in languages # Now the condition is the name of the language
        for word in udhr.words(lang + '-Latin1')) # It exploits the fact that the filename for each language is the language name followed by '-Latin1'
cfd.plot(cumulative=True)

# For example, we can tabulate the cumulative frequency data just for two languages, and for words less than 10 characters long, as shown below. 
cfd.tabulate(conditions=["English", "German_Deutsch"],
             samples=range(10), cumulative=True)
# We interpret the last cell on the top row to mean that 1,638 words of the English text have 9 or fewer letters.

# Now we plot a frequency distribution of the letters of the text using nltk.FreqDist(raw_text).plot() of spanish
raw_text = udhr.raw('Spanish-Latin1')
nltk.FreqDist(raw_text).plot()

# Basic Corpus Functionality defined in NLTK: more documentation can be found using help(nltk.corpus.reader) and by reading the online Corpus HOWTO at 
# http://nltk.org/howto.

# YOUR TURN: Working with the news and romance genres from the Brown Corpus, find out which days of the week are most newsworthy, and which are most romantic.
# Define a variable called days containing a list of days of the week, i.e. ['Monday', ...]. Now tabulate the counts for these words using
# cfd.tabulate(samples=days). Now try the same thing using plot in place of tabulate. You may control the output order of days with the help of an extra
# parameter: samples=['Monday', ...].
genres = ['news', 'romance']
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
# Now, we obtain counts for each genre of interest
cfd = nltk.ConditionalFreqDist(
          (genre, word)
          for genre in genres
          for word in brown.words(categories=genre))
cfd.tabulate(samples=days)
cfd.plot(samples=days)

# Generating Random Text with Bigrams: the function generate_model() contains a simple loop to generate text.
def generate_model(cfdist, word, num=15):
    for i in range(num):
        print(word, end=' ')
        word = cfdist[word].max()

text = nltk.corpus.genesis.words('english-kjv.txt')
bigrams = nltk.bigrams(text)
cfd = nltk.ConditionalFreqDist(bigrams)

# This program obtains all bigrams from the text of the book of Genesis, then constructs a conditional frequency distribution to record which words are most 
# likely to follow a given word; e.g., after the word living, the most likely word is creature; the generate_model() function uses this data, and a seed word,
# to generate random text.
print("Frequencies of the word 'living' and its bigrams: \n{}".format(cfd['living']))
generate_model(cfd, 'living')


# WORDLIST CORPORA
def unusual_words(text):
    text_vocab = set(w.lower() for w in text if w.isalpha()) # here we set the vocabulary of a text
    english_vocab = set(w.lower() for w in nltk.corpus.words.words()) # here we set the vocabulary of the English dictionary
    unusual = text_vocab - english_vocab # here we set the words that are in the text but not in the English dictionary
    return sorted(unusual)

print(unusual_words(webtext.words('singles.txt'))) # unusual words from the singles.txt corpus
print(unusual_words(gutenberg.words('shakespeare-macbeth.txt'))) # unusual words from Macbeth
print(unusual_words(nltk.corpus.nps_chat.words())) # unusual words from the chat corpus

# Function to define how many words aren't stop words
def content_fraction(text):
    stopwords = nltk.corpus.stopwords.words('english')
    content = [w for w in text if w.lower() not in stopwords]
    return len(content)/len(text)

print(content_fraction(nltk.corpus.reuters.words())) # fraction of words in the reuters corpora that aren't stopwords

# Word Puzzle
wordlist = nltk.corpus.words.words() # words in the English dictionary
puzzle = nltk.FreqDist('egivrvonl') # the letters we can use
obligatory = 'r'
minimum = 4
words = [w for w in wordlist if len(w) >= minimum and obligatory in w and nltk.FreqDist(w) <= puzzle]


# Names list and ambiguous names for male and female
male_names = nltk.corpus.names.words('male.txt')
female_names = nltk.corpus.names.words('female.txt')
ambiguous = [w for w in male_names if w in female_names]

# Seeing whether a name is female or male depending on its last letter
cfd = nltk.ConditionalFreqDist(
            (sex, name[-1])
            for sex in nltk.corpus.names.fileids()
            for name in nltk.corpus.names.words(sex))
cfd.plot()

# USE BY SPEECH: not interested for the moment

# 2.  Use the corpus module to explore austen-persuasion.txt. How many word tokens does this book have? How many word types?
persuasion = gutenberg.words(gutenberg.fileids()[1])
n_tokens = len(persuasion) # number of tokens
n_word_types = len(set(persuasion)) # number of word types (unique words)

# 8. Define a conditional frequency distribution over the Names corpus that allows you to see which initial letters are more frequent for males vs. females
male_names = nltk.corpus.names.words('male.txt') # 2943 names
female_names = nltk.corpus.names.words('female.txt') # 5001 female names
# Seeing whether a name is female or male depending on its first letter
cfd = nltk.ConditionalFreqDist(
            (sex, name[0])
            for sex in nltk.corpus.names.fileids()
            for name in nltk.corpus.names.words(sex))
cfd.plot()
# As it is really imbalances, lets do the same but dividing by the total number of names to see it in the same scale
cfd_normalised = cfd
for sex in nltk.corpus.names.fileids():
    for key in cfd[sex].keys():
        cfd_normalised[sex][key] /= len(nltk.corpus.names.words(sex))
cfd_normalised.plot()

# 9. Pick a pair of texts and study the differences between them, in terms of vocabulary, vocabulary richness, genre, etc. Can you find pairs of words which
# have quite different meanings across the two texts, such as monstrous in Moby Dick and in Sense and Sensibility?
# For this exercise I will choose Macbeth (I saw it recently) and Hamlet. Two works from Shakespear that are about very different things
macbeth = gutenberg.words('shakespeare-macbeth.txt')
hamlet = gutenberg.words('shakespeare-hamlet.txt')
    # a) Vocabulary
print("Macbeth has {} words and Hamlet has {} words".format(len(macbeth), len(hamlet)))
    # b) Vocabulary Richness
print("Macbeth has {} unique words and Hamlet has {} unique words".format(len(set(macbeth)), len(set(hamlet))))
# Now, I will remove the stopwords from both texts and check it again
stopwords = nltk.corpus.stopwords.words('english')
clean_macbeth = [w for w in macbeth if w.lower() not in stopwords]
clean_hamlet = [w for w in hamlet if w.lower() not in stopwords]
    # a) Vocabulary
print("Macbeth has {} words and Hamlet has {} words".format(len(clean_macbeth), len(clean_hamlet)))
    # b) Vocabulary Richness
print("Macbeth has {} unique words and Hamlet has {} unique words".format(len(set(clean_macbeth)), len(set(clean_hamlet))))























































