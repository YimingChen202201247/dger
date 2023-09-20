import nltk
from nltk.corpus import gutenberg
from nltk.probability import FreqDist
import matplotlib.pyplot as plt

nltk.download('gutenberg')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Read Moby Dick from the Gutenberg dataset
moby_dick = gutenberg.raw('melville-moby_dick.txt')

# Tokenization
tokens = word_tokenize(moby_dick)

# Stopwords filtering
stop_words = set(stopwords.words('english'))
filtered_tokens = [word.lower() for word in tokens if word.isalnum() and word.lower() not in stop_words]

# Parts-of-Speech (POS) tagging
pos_tags = nltk.pos_tag(filtered_tokens)

# POS frequency
pos_counts = FreqDist(tag for (word, tag) in pos_tags)
top_pos = pos_counts.most_common(5)

print("Top 5 Parts of Speech and their Frequencies:")
for tag, count in top_pos:
    print(f"{tag}: {count}")

# Lemmatization
lemmatizer = WordNetLemmatizer()
top_lemmas = [lemmatizer.lemmatize(word, pos) for word, pos in pos_tags[:20]]

print("\nTop 20 Lemmas:")
print(top_lemmas)

# Plotting frequency distribution
pos_counts.plot(20, title="POS Frequency Distribution")
plt.show()
