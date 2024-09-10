import pandas as pd
import requests
import nltk
import re
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from textblob import TextBlob

input_file="Input.xlsx"
data=pd.read_excel(input_file)

def extract_article_text(url):
    try:
        response=requests.get(url)
        soup=BeautifulSoup(response.content, 'html.parser')

        title=soup.title.get_text().strip()

        article_body=''
        for p in soup.find_all('p'):
            article_body+=p.get_text().strip()+''

        return title,article_body.strip()

    except Exception as e:
        print("Error Extracting:")
        return None, None

for index, row in data.iterrows():
    url=row['URL']
    url_id=row['URL_ID']

    title, article_text= extract_article_text(url)

    if article_text:

        with open(f"{url_id}.txt","w", encoding="utf-8") as file:
            file.write(f"{title}\n{article_text}")

print("Data Extraction Complete")

#sentiment analysis
def analyze_sentiment(text):
    blob = TextBlob(text)


    polarity_score = blob.sentiment.polarity
    subjectivity_score = blob.sentiment.subjectivity


    positive_score = max(polarity_score, 0)
    negative_score = -min(polarity_score, 0)

    return positive_score, negative_score, polarity_score, subjectivity_score

#Readability & Other Matrix
def calculate_readability_metrics(text):
    words = text.split()
    num_sentences = text.count('.') + text.count('!') + text.count('?')
    num_words = len(words)

    if num_sentences == 0:
        num_sentences = 1  # Avoid division by zero

    # Calculate average sentence length
    avg_sentence_length = num_words / num_sentences

    # Count complex words
    complex_words = [word for word in words if count_syllables(word) >= 3]
    complex_word_count = len(complex_words)

    # Calculate percentage of complex words
    percentage_complex_words = (complex_word_count / num_words) * 100

    # Calculate FOG Index
    fog_index = 0.4 * (avg_sentence_length + percentage_complex_words)

    # Calculate average number of words per sentence
    avg_words_per_sentence = num_words / num_sentences

    # Calculate syllable per word
    total_syllables = sum(count_syllables(word) for word in words)
    syllable_per_word = total_syllables / num_words

    # Calculate average word length
    total_chars = sum(len(word) for word in words)
    avg_word_length = total_chars / num_words

    return avg_sentence_length, percentage_complex_words, fog_index, avg_words_per_sentence, complex_word_count, num_words, syllable_per_word, avg_word_length

def count_syllables(word):
    word = word.lower()
    if not word:
        return 0

    word = re.sub(r'[^\w\s]', '', word)  # Remove punctuation

    vowels = 'aeiou'
    syllables = 0
    prev_char = ''

    for char in word:
        if char in vowels:
            if prev_char not in vowels:
                syllables += 1
        prev_char = char

    # Handle cases where syllable count might be zero
    if syllables == 0:
        syllables = 1
    return syllables

def count_personal_pronouns(text):
    pronouns=re.findall(r'\b(I|we|my|ours|us)\b',text,re.I)
    return len(pronouns)

output_data=[]

for index, row in data.iterrows():
    url_id=row['URL_ID']

    with open(f"{url_id}.txt", "r", encoding="utf-8") as file:
        article_text=file.read()

        positive_score, negative_score, polarity_score, subjectivity_score = analyze_sentiment(article_text)

        avg_sentence_length, percentage_complex_words, fog_index, avg_words_per_sentence, complex_word_count, word_count, syllable_per_word, avg_word_length = calculate_readability_metrics(article_text)

        personal_pronouns = count_personal_pronouns(article_text)

        output_data.append([
            row['URL_ID'], row['URL'],
            positive_score, negative_score, polarity_score, subjectivity_score,
            avg_sentence_length, percentage_complex_words, fog_index,
            avg_words_per_sentence, complex_word_count, word_count,
            syllable_per_word, personal_pronouns, avg_word_length
        ])

        output_df = pd.DataFrame(output_data, columns=[
            "URL_ID", "URL",
            "POSITIVE SCORE", "NEGATIVE SCORE", "POLARITY SCORE", "SUBJECTIVITY SCORE",
            "AVG SENTENCE LENGTH", "PERCENTAGE OF COMPLEX WORDS", "FOG INDEX",
            "AVG NUMBER OF WORDS PER SENTENCE", "COMPLEX WORD COUNT", "WORD COUNT",
            "SYLLABLE PER WORD", "PERSONAL PRONOUNS", "AVG WORD LENGTH"
        ])

        output_df.to_csv("Output.csv", index=False)






