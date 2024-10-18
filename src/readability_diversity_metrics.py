import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
import syllapy

# Download required NLTK resources
nltk.download('punkt')


# Function to count syllables in a word
def count_syllables(word):
    return syllapy.count(word)


# Function to compute all metrics
def compute_diversity_and_readability_metrics(texts):
    """The definitions are taken from Yael Moros-Daval's thesis, available at:
    https://www.europeana.eu/en/item/355/https___hispana_mcu_es_lod_oai_riunet_upv_es_10251____196727_ent0
    """

    results = []

    for text in texts:
        # Tokenize sentences and words
        sentences = sent_tokenize(text)
        words = word_tokenize(text)

        # Remove punctuation and filter non-alphabetic words
        words = [word for word in words if word.isalpha()]

        # Calculate the number of words (nw), sentences (nst), and syllables (nsy)
        n_w = len(words)
        n_st = len(sentences)
        n_sy = sum(count_syllables(word) for word in words)
        ASL = n_w / n_st if n_st > 0 else 0  # Avoid division by zero for ASL

        # Calculate the number of words with 3 or more syllables (nwsy>=3)
        nwsy_ge_3 = sum(1 for word in words if count_syllables(word) >= 3)

        # Calculate the number of one-syllable words (nwsy=1)
        nwsy_eq_1 = sum(1 for word in words if count_syllables(word) == 1)

        # Compute Flesch's Reading Ease Score
        flesch = 206.835 - (1.015 * ASL) - (84.6 * (n_sy / n_w)) if n_w > 0 else 0

        # Compute Gunning's Fog Index
        fog = 0.4 * (ASL + 100 * (nwsy_ge_3 / n_w)) if n_w > 0 else 0

        # Compute SMOG Regression Equation C
        smog_c = 0.9986 * np.sqrt(nwsy_ge_3 * (30 / n_st) + 5) + 2.8795 if n_st > 0 else 0

        # Compute FORCAST readability grade level
        forecast = 20 - (nwsy_eq_1 * 150 / (n_w * 10)) if n_w > 0 else 0

        # Compute Scrabble Measure (mean Scrabble letter values)
        scrabble_values = {'a': 1, 'b': 3, 'c': 3, 'd': 2, 'e': 1, 'f': 4, 'g': 2, 'h': 4,
                           'i': 1, 'j': 8, 'k': 5, 'l': 1, 'm': 3, 'n': 1, 'o': 1, 'p': 3,
                           'q': 10, 'r': 1, 's': 1, 't': 1, 'u': 1, 'v': 4, 'w': 4, 'x': 8,
                           'y': 4, 'z': 10}

        scrabble_score = sum(scrabble_values.get(char.lower(), 0) for word in words for char in word)
        scrabble_measure = scrabble_score / n_w if n_w > 0 else 0

        # Compute Type-Token Ratio (TTR) and Yule's K
        word_counts = nltk.FreqDist(words)
        N = sum(word_counts.values())
        V = len(word_counts)
        TTR = V / N if N > 0 else 0  # To avoid division by zero

        freq_of_freq = nltk.FreqDist(word_counts.values())
        K = (-1 / N + sum((f / N) * ((i / N) ** 2) for i, f in freq_of_freq.items() if i > 0)) if N > 0 else 0

        # Append the results to the list
        results.append({
            "Flesch Reading Ease Score": flesch,
            "Gunning's Fog Index": fog,
            "SMOG.C": smog_c,
            "FORCAST": forecast,
            "Scrabble Measure": scrabble_measure,
            "Number of Words": n_w,
            "Average sentence length": ASL,
            "Type-Token Ratio (TTR)": TTR,
            "Yule's K": K
        })

    return results


if __name__ == "__main__":
    # Example usage
    texts = ["This is a simple text.", "Quanteda is a great library for text analysis text.", "bla bla bla", ""]
    metrics = compute_diversity_and_readability_metrics(texts)
    for idx, metric in enumerate(metrics):
        print(f"Document {idx + 1} Metrics:")
        for key, value in metric.items():
            print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
        print()
