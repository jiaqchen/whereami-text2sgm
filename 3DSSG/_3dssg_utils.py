# Python suppress warnings from spaCy
import warnings
warnings.filterwarnings("ignore", message=r"\[W095\]", category=UserWarning)

import spacy
nlp = spacy.load("en_core_web_md")

def noun_in_list_of_nouns(noun, nouns, threshold=0.5):
    # Get word2vec of noun
    noun_vec = nlp(noun)[0].vector

    # Find the noun in nouns with the highest similarity, spacy similarity
    max_sim = 0
    max_sim_noun = None
    for n in nouns:
        # oun_vec = nlp(n)[0].vector
        sim = nlp(noun).similarity(nlp(n))
        if sim > max_sim:
            max_sim = sim
            max_sim_noun = n
    return max_sim_noun, max_sim > threshold