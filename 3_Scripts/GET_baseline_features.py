"""
Entity Graph for German
Author: Julia Suter, 2018/19
-----------------------------

GET_baseline_features.py

    - Get and save text collection ngram information
    - Collect set of linguistic (baseline) features (full or reduced)
    - Save the baseline features for each sample

"""


# ----------------------------------------
# Imports and Settings
# ----------------------------------------

# Import Statements
import numpy as np
import syntactic_feature_extraction as fe
import string
import os
import shutil
import re

# Settings
N_SENTS = 50
lang_levels = False
MAX = 50

# True: load_ngrams (if already saved, set to False)
load_ngrams = True

# Number of ngrams
ngrams = 100

# Directory with parsed texts
DIRECTORY = '../0_Datasets/GUTENBERG_parsed_texts/'

# Set output directory
TARGET_DIR = '../2_Features/baseline_features/'

# Use only reduced feature set (default)
reduced_f_set = True

if reduced_f_set:
    TARGET_DIR = '../2_Features/baseline_features_reduced/'
    
# Remove and create target dir
if os.path.exists(TARGET_DIR):
    shutil.rmtree(TARGET_DIR)
os.makedirs(TARGET_DIR)
    
# ----------------------------------------
# Compute and/or load corpus information
# ----------------------------------------

def get_full_corpus_information(ngrams=100):
    """Get required information of full corpus: most frequent POS bigrams, char bigrams and trigrams"""

    # Get pos bigrams, and character bigrams and trigrams
    mf_pos_bigrams, mf_char_bigrams, mf_char_trigrams = fe.get_corpus_information(DIRECTORY, ngrams)

    # Save results
    np.save('../2_Features/info/synt_features/mf_pos_bigrams.npy', mf_pos_bigrams)
    np.save('../2_Features/info/synt_features/mf_char_bigrams.npy', mf_char_bigrams)
    np.save('../2_Features/info/synt_features/mf_char_trigrams.npy', mf_char_trigrams)

if load_ngrams:
    get_full_corpus_information(ngrams)

# Load corpus information
mf_pos_bigrams = [(a,b) for a,b in np.load('../2_Features/info/synt_features/mf_pos_bigrams.npy')]
mf_char_bigrams = [(a,b) for a,b in  np.load('../2_Features/info/synt_features/mf_char_bigrams.npy')]
mf_char_trigrams = [(a,b,c) for a,b,c in np.load('../2_Features/info/synt_features/mf_char_trigrams.npy')]

# ----------------------------------------
# Set and save feature names
# ----------------------------------------

# Set lexical feature names
lex_f_names = ['zu','primary verbs',
               'prepositions', 'pronouns', 'determiners', 
               'conjunctions', 'adverbs',                
               'hapax legomena','hapax dislegomena']

# Set character feature names
char_f_names = (['punctuation','alpha chars', 'digits', 'upper case', 'lower case']
                + ['_foreign char'] + list(string.ascii_lowercase) + ['ä','ö','ü','ß'] 
                + [' + '.join([a,b]) for (a,b) in mf_char_bigrams] + [' + '.join([a,b,c]) for (a,b,c) in mf_char_trigrams])
    
# Set syntactic feature names
synt_f_names = [' + '.join([a,b]) for (a,b) in mf_pos_bigrams] + ['POS entropy']

# All feature names
feature_names = lex_f_names + char_f_names + synt_f_names

# Feature names for reduced set
if reduced_f_set:
    feature_names = lex_f_names + char_f_names + synt_f_names
    
# Save feature names
np.save('baseline_feature_names.npy', feature_names)

# ----------------------------------------
### Prepare feature and solution array
# ----------------------------------------

# Set nr features, sample, feature and solution array
n_features = len(feature_names)
n_samples = len(os.listdir(DIRECTORY))  #n_samples = 32622

# Initiate feature array
feature_array = np.zeros((n_samples, len(feature_names)))

print('Number features',n_features)
print(feature_names)

if reduced_f_set:
    feature_array = np.zeros((n_samples, 31))
    
# Prepare solution array
solution_array = []

# ----------------------------------------
# Get features for chunks of N sentences for every file
# ----------------------------------------

# For each file in dir...
for i, file in enumerate(os.listdir(DIRECTORY)):
    
    # Set new filename
    new_filename = file[:-4]
    new_filename = re.sub(r'\..$','',new_filename)       

    # Get parsed sentences
    sentences = fe.get_sentences_from_parsed_text(DIRECTORY+file)
    n_sentences = len(sentences)
     
    # Discard if less than N sents
    if n_sentences<N_SENTS:
        continue
        
    # Get nummber of N sentences chunks
    splits = n_sentences//N_SENTS

    # For each part
    for i in range(splits):
                
        # Stop if already more than MAX samples created from this file
        if i >= MAX:
            break
            
        # Get next chunk of sentences
        sents = sentences[i*N_SENTS:(i+1)*N_SENTS]

        # Set split filename
        split_filename = new_filename+'_'+str(i+1)+'.npy'

        print('\t'+split_filename)
        
        # Get author tag (for controlling only -- not used)
        author = re.sub('_.*','',split_filename)     
        solution_array.append(author)
        
        # Get parsed sentences and basic information
        parsed_text = fe.get_sentence_token_information(sents)    
        all_tokens, all_lemmas, full_text = fe.get_basic_info(parsed_text)

        # Extract all features for sample
        features = fe.extract_features(all_tokens, all_lemmas, full_text, mf_pos_bigrams, mf_char_bigrams, mf_char_trigrams)
        
        # Set target dir
        file_path = TARGET_DIR + split_filename
                
        print(file_path)
        
        # For reduced set, only use 31 features
        if reduced_f_set:

            x_features = features.tolist()
            
            # Take first 10 and last 21 features only
            reduced_features = x_features[:10]
            reduced_features += x_features[-21:]
            features = reduced_features

        # Save sample feature array
        np.save(file_path, features)
        
        # Add to complete feature array for all samples
        feature_array[i,:] = features
        
# ----------------------------------------
# Save feature arrays
# ----------------------------------------

# Print feature array shape
print(feature_array.shape)

# Save feature and solution arrays
np.save('../2_Features/info/misc/complete_baseline_feature_array.npy', feature_array)
np.save('../2_Features/info/misc/complete_baseline_solution_array.npy', solution_array)

# Save feature names
if reduced_f_set:
    np.save('../2_Features/info/feature_names/baseline_feature_names_reduced.npy', feature_names)
else:
    np.save('../2_Features/info/feature_names/baseline_feature_names.npy', feature_names)
