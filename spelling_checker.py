import nltk
from nltk.tokenize import word_tokenize
import Levenshtein
import string
    

def closest_words(word, word_list):
    same_start = [w for w in word_list if w.startswith(word[0])]
    suggestions = same_start if same_start else word_list
    ranked = sorted(suggestions, key=lambda w: Levenshtein.distance(word, w))
    return ranked[:3]


def spell_check(in_text):
    with open('words_final.txt', 'r') as f:
        words = [w.strip().lower() for w in f if w.strip()]

    highlighted = ""
    final_sen = []
    #Tokenizes
    tokens = word_tokenize(in_text)

    for word in tokens:

        # punctuation
        if all(ch in string.punctuation for ch in word):
            final_sen.append(word)
            highlighted += word + " "
            continue

        word_lower = word.lower()

        # correct word
        if word_lower in words:
            final_sen.append(word)
            highlighted += word + " "
            continue

        # incorrect → find suggestions
        close_word = closest_words(word_lower, words)
        best_word = close_word[0]

        # maintain capitalization
        if word[0].isupper():
            best_word = best_word.capitalize()

        final_sen.append(best_word)
        highlighted += f"<mark style='background-color:#DABFE0'>{best_word}</mark> "

    correct_sentence = ' '.join(final_sen)
    print(correct_sentence)
    return correct_sentence


# RUN THE FUNCTION
if __name__ == "__main__":
    text = input("Enter sentence: ")
    spell_check(text)
