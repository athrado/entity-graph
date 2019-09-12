"""
Entity Graph for German
Author: Julia Suter, 2018/19
-----------------------------

syntactic_feature_extraction.py

    - Extract and compute linguistic (baseline) features
    - Get parsed sentences
    - Get corpus information (ngrams)
    - Compute lexical features
    - Compute character features
    - Compute syntactic features
    - Merge feature sets

"""


# Import Statements

from __future__ import division

import parse_information as easy_parse

import os
import codecs
import string
import numpy as np

import nltk
from nltk.probability import FreqDist


# How many ngrams should be collected
N_NGRAMS = 100

def get_sentences_from_parsed_text(doc):
    """Return sentences of a document with already parsed text in CoNNL format."""    

    # Open file and split at sentence boarders    
    with codecs.open(doc,"r",'l1') as infile:
        infile = infile.read()
        sentences = infile.split('\n\n')[:-1]

    return sentences

def get_sentence_token_information(sentences):
    """Return parsed as nested sentence-token-information list."""
    
    parsed_text = []
    
    # For each sent, get all tokens
    for sent in sentences:
        parsed_sentence = []
    
        # Split sentence into tokens 
        tokens = sent.split('\n')
        for token in tokens:
            
            # Split token string into token information
            parsed_token = [t for t in token.split('\t')]
            parsed_sentence.append(parsed_token)
            
        parsed_text.append(parsed_sentence)
        
    return parsed_text


def get_text(parsed_text):
    """Return complete sentence as string without parsing information."""
    
    full_text = ''
    
    # For each sent, get all tokens and connect correctly
    for sent in parsed_text:
        tokens = [easy_parse.Token(t) for t in sent]
        
        for token in tokens:
            full_text += token.word
            
            # Get prev and next token
            next_token = easy_parse.get_next_token(tokens, token)
            
            # Insert white space unless there is punctuation mark
            if ((not (token.position != len(sent)
                            and next_token.sim_pos.startswith('$')
                            and next_token.word != '('))  
                            and token.word != '('):  

                            full_text += ' '   
                            
    return full_text
    
    
def get_basic_info(parsed_text):
    """Collect basic information for parsed texts, such as all tokens and lemmas."""
    
    # Info
    n_words = 0
    all_words = []
    all_lemmas = []
    all_tokens = []
    
    # For each sentence
    for sentence in parsed_text:
    
        # Transform tokens into Token class instances
        tokens = [easy_parse.Token(k) for k in sentence]
        
        # Get words and lemmas, add them to all words and lemmas
        words = [t.word for t in tokens]    
        lemmas = [t.lemma for t in tokens]
        
        all_words += words
        all_lemmas += lemmas
        
        # Get number of words
        n_words += len(words)
        all_tokens += tokens
        
    full_text = get_text(parsed_text)
        
    return all_tokens, all_lemmas, full_text

def get_corpus_information(directory, ngrams=100):
    """Get feature information for all files in directory:
       words, lemmas, characters, part-of-speech, bigrams, trigrams."""
    
    # Number of ngrams
    N_NGRAMS = ngrams

    # Get files from direcory
    all_files = os.listdir(directory)
    
    token_collection = []
    lemma_collection = []
    all_texts_as_string = ''
    
    # For each file, get parsed sentences and text information
    for file in all_files:
        
        # Get sentences and parsed texts
        sentences = get_sentences_from_parsed_text(directory+file)
        parsed_text = get_sentence_token_information(sentences)

        all_tokens, all_lemmas, full_text = get_basic_info(parsed_text)

        # Update collections
        token_collection += all_tokens
        lemma_collection += all_lemmas
        all_texts_as_string += full_text

    
    # Get part-of-speech information
    pos_collection = [t.sim_pos for t in token_collection]
    
    # Get bigrams
    pos_bigram_collection = nltk.bigrams(pos_collection)
    pos_bigram_fdist = FreqDist(pos_bigram_collection)
    
    # Get most frequent part-of-speech bigrams
    mf_pos_bigrams = sorted(pos_bigram_fdist, key=pos_bigram_fdist.get, reverse=True)[:20]

    # Get all characters
    char_collection = list(all_texts_as_string.lower())

    # Get character bigrams
    char_bigram_collection = nltk.bigrams(char_collection)
    char_bigram_fdist = FreqDist(char_bigram_collection)
    
    # Get most frequent bigrams
    mf_char_bigrams = sorted(char_bigram_fdist, key=char_bigram_fdist.get, reverse=True)[:N_NGRAMS]

    # Get character trigrams
    char_trigram_collection = nltk.trigrams(char_collection)
    char_trigram_fdist = FreqDist(char_trigram_collection)
    
    # Get most frequent character trigrams
    mf_char_trigrams = sorted(char_trigram_fdist, key=char_trigram_fdist.get, reverse=True)[:N_NGRAMS]

    return mf_pos_bigrams, mf_char_bigrams, mf_char_trigrams
    
    
def get_character_features(all_tokens, full_text, mf_char_bigrams, mf_char_trigrams):
    """Collect character features, such as frequency of characters or character types
    as well as bigrams and trigrams."""
    
    # Get n chars and tokens
    n_chars = len(full_text)
    n_tokens = len(all_tokens)
    
    # Get character type lists
    alpha_chars = [char for char in full_text if char.isalpha()]
    digit_chars = [char for char in full_text if char.isdigit()]
    upper_chars = [char for char in full_text if char.isupper()]
    lower_chars = [char for char in full_text if char.islower()]
    punct_chars = [char for char in full_text if char in ['.',',','?','!',':',';','(',')','[',']','-','"',"'",'»','«']]
    
    # Get character type frequencies
    alpha_char_freq = len(alpha_chars)/n_chars
    digit_char_freq = len(digit_chars)/n_chars
    upper_char_freq = len(upper_chars)/n_chars
    lower_char_freq = len(lower_chars)/n_chars
    punct_char_freq = len(punct_chars)/n_chars
    
    # Set character features
    char_features = [punct_char_freq, alpha_char_freq, digit_char_freq, upper_char_freq, lower_char_freq]
           
    # Set characters of interest    
    chars_of_interest = list(string.ascii_lowercase)
    chars_of_interest += ['ä','ö','ü','ß']        
    
    # Get alphabetic character frequency
    alpha_char_freq_dict = {char:0 for char in chars_of_interest}
    alpha_char_freq_dict['_ foreign char'] = 0
    
    # Get bigrams and trigram frequncies
    bigram_freq_dict = {bigram:0 for bigram in mf_char_bigrams}
    trigram_freq_dict = {trigram:0 for trigram in mf_char_trigrams}
    
    # Get full lowercased text
    lower_full_text = full_text.lower()
    
    # Get character frequencies
    all_chars =  [char for char in list(lower_full_text)]
    all_alpha_chars = [char for char in all_chars if char.isalpha()]
    n_alpha_chars = len(all_alpha_chars)
    fdist = FreqDist(all_alpha_chars)
    
    # Save frequencies in dict, specal case: foreign characters
    for key in fdist.keys():        
        if key in alpha_char_freq_dict.keys():
            alpha_char_freq_dict[key] = fdist[key]
        else:
            alpha_char_freq_dict['_ foreign char'] += fdist[key]
    
    # Get bigrams and trigrams of characters
    bigrams = nltk.bigrams(all_chars)
    trigrams = nltk.trigrams(all_chars)

    # Get bigram and trigram frequencies
    bigram_fdist =  FreqDist(bigrams)
    trigram_fdist = FreqDist(trigrams)
    
    # Save bigrams in dict
    for key in bigram_freq_dict.keys():        
        if key in bigram_fdist.keys():            
            bigram_freq_dict[key] = bigram_fdist[key]
    
    # Save trigrams in dict            
    for key in trigram_freq_dict.keys():        
         if key in trigram_fdist.keys():            
            trigram_freq_dict[key] = trigram_fdist[key]         
            
    # Normalize frequencies
    alpha_char_freq = [alpha_char_freq_dict[k]/n_alpha_chars for k in sorted(alpha_char_freq_dict)]
    bigram_freq = [bigram_freq_dict[k]/len(bigram_fdist) for k in sorted(bigram_freq_dict)]
    trigram_freq = [trigram_freq_dict[k]/len(trigram_fdist) for k in sorted(trigram_freq_dict)]
    
    # Compose final character feature array
    char_feature_array = np.array(char_features+alpha_char_freq+bigram_freq+trigram_freq)
        
    return char_feature_array
    
    
def get_freqs_of_function_words(all_tokens):
    """Get frequences of function words and specifiy part-of-speech words."""
    
    # Word types
    prepositions = [t for t in all_tokens if t.sim_pos_full == 'PREP']
    pronouns = [t for t in all_tokens if t.sim_pos_full == 'PRO']
    determiners = [t for t in all_tokens if t.full_pos == 'ART']
    conjunctions = [t for t in all_tokens if t.sim_pos_full in ['KON','KOUS','KOUI']]
    primary_verbs = [t for t in all_tokens if t.sim_pos == 'V' and t.lemma in ['sein','haben','werden']]
    adverbs = [t for t in all_tokens if t.sim_pos_full == 'ADV']
    #punctuations = [t for t in all_tokens if t.sim_pos.startswith('$')]
    
    zu = [t for t in all_tokens if t.sim_pos_full == 'PTKZU' and t.lemma == 'zu']
    
    # Get normalizesd frequencies
    freq_prepositions = len(prepositions)/len(all_tokens)
    freq_pronouns = len(pronouns)/len(all_tokens)
    freq_determiners = len(determiners)/len(all_tokens)
    freq_conjunctions = len(conjunctions)/len(all_tokens)
    freq_primary_verbs = len(primary_verbs)/len(all_tokens)
    freq_adverbs = len(adverbs)/len(all_tokens)
    freq_zu = len(zu)/len(all_tokens)
    #freq_punctuations = len(punctuations)/len(all_tokens)
    
    # Cmpose final feature list
    all_freqs = (freq_zu,  freq_primary_verbs,  freq_prepositions, freq_pronouns, freq_determiners, freq_conjunctions,
                 freq_adverbs)
                 
    return all_freqs
    
def get_hapax(all_lemmas):
    """Get frequencies of hapax legomena and dislegomena."""
    
    # Get frequenc distribution
    fdist = FreqDist(all_lemmas)
    
    # Collect words that only appear once or twice
    hapax_legomena = [k for k,v in fdist.items() if v == 1]
    hapax_dislegomena = [k for k,v in fdist.items() if v == 2]
    
    # Get normalized frequencies
    freq_hapax_legomena = len(hapax_legomena)/len(all_lemmas)
    freq_hapax_dislegomena = len(hapax_dislegomena)/len(all_lemmas)
    
    return freq_hapax_legomena, freq_hapax_dislegomena


def get_lexical_features(all_tokens, all_lemmas):
    """Get lexical features, such as frequencies of function words and hapax (dis)legomena."""
    
    # Get frequencies
    function_words = get_freqs_of_function_words(all_tokens)
    hapax = get_hapax(all_lemmas)
    
    # Set lexical feature array
    lexical_feature_array = np.array(function_words+hapax)
    
    return lexical_feature_array
    


def get_pos_entropy(all_tokens):    
    """Get part-of-speech entropy."""
    
    # Get all pos tags
    pos = [t.sim_pos for t in all_tokens]
    
    # Get frequencies
    pos_dist = FreqDist(pos)
    values = list(pos_dist.values())  

    # Get probability array
    prob_array = np.array(values)    
    prob_array_norm = prob_array/sum(prob_array)
    
    # Compute entropy
    entropy = np.sum(-1*(prob_array_norm)*np.nan_to_num(np.log2(prob_array_norm)))
    
    return entropy
    
    
def get_frequent_pos_bigrams(all_tokens, mf_pos_bigrams):
    """Get frequent part-of-speech bigrams."""
    
    # Get all part-of-speech tags
    all_pos = [t.sim_pos_full for t in all_tokens]
    
    # Get bigrams and frequencies
    pos_bigrams = nltk.bigrams(all_pos)
    pos_fdist = FreqDist(pos_bigrams)
    
    # Set dict
    pos_bigram_freq_dict = {bigram:0 for bigram in mf_pos_bigrams}
    
    # Fill dict 
    for key in pos_bigram_freq_dict.keys():        
         if key in pos_fdist.keys():            
            pos_bigram_freq_dict[key] = pos_fdist[key]   
            
    # Normalize frequencies
    pos_bigram_freq = [pos_bigram_freq_dict[k]/len(pos_fdist) for k in sorted(pos_bigram_freq_dict)]
    
    return pos_bigram_freq
    
    
    
def get_syntactic_features(all_tokens, mf_pos_bigrams):
    """Get syntactic features, such as part-of-speech frequencies and entropy."""
    
    # Get pos entropy
    entropy = get_pos_entropy(all_tokens)
    # Get pos bigrams
    pos_bigrams = get_frequent_pos_bigrams(all_tokens, mf_pos_bigrams)
    
    # Set syntactic features
    syntactic_feature_array = np.array([entropy]+pos_bigrams)
    
    return syntactic_feature_array
    
    
def extract_features(all_tokens, all_lemmas, full_text, mf_pos_bigrams, mf_char_bigrams, mf_char_trigrams):
    """Extract all types of features: character, lexical and syntactic features. """
    
    lex_features = get_lexical_features(all_tokens, all_lemmas)
    char_features = get_character_features(all_tokens, full_text, mf_char_bigrams, mf_char_trigrams)
    synt_features = get_syntactic_features(all_tokens, mf_pos_bigrams)
    
    # Concatenate feature lists
    feature_array = np.concatenate((lex_features, char_features, synt_features))
    
    return feature_array
            

def main():
    """Main function used for testing and demonstration."""
        
    # Set directory: set of parsed texts
    DIRECTORY = '/home/jsuter/Documents/03_HITS/Entity_Graph/GitHub/Entity_Graph_GitHub/entity-graph/datasets/GUTENBERG_parsed_texts/'

    # Get corpus features, such as part-of-speech bigrama snad character features
    mf_pos_bigrams, mf_char_bigrams, mf_char_trigrams = get_corpus_information(DIRECTORY)

    # Set test file
    test_file = os.listdir(DIRECTORY)[0]
    print('Test file:\t', test_file)

    # Get sentences, parsed texts and basic information
    sentences = get_sentences_from_parsed_text(DIRECTORY+test_file)
    parsed_text = get_sentence_token_information(sentences)    
    all_tokens, all_lemmas, full_text = get_basic_info(parsed_text)
    
    # Get all features
    feature_array = extract_features(all_tokens, all_lemmas, full_text, mf_pos_bigrams, mf_char_bigrams, mf_char_trigrams)
    print('# Features:\t', feature_array.shape[0])
    

# Call main function
if __name__ == '__main__':
    main() 
      