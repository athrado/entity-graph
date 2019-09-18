"""
Entity Graph for German
Author: Julia Suter, 2018/19
-----------------------------

RUN_classification.py

- Set settings for classification task (authors/genres)
- Load samples and features
- Train and test Support Vector Classifier
- Here: for author and genre categories, but can be adapted to any text classification task
- Compute confusion matrix to analyse performance
- Print and save results
- Analyse features: PCA (boxplot), correlation

"""


# ------------------------------------------------
# Import Statements
# ------------------------------------------------

import os
import sys
import pickle

import random
import re
import pprint
from operator import itemgetter

import numpy as np
import pandas as pd

import sklearn
from sklearn import svm
from sklearn.decomposition import PCA
from scipy.stats import linregress
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore', message=".*is ill-defined and being set to 0.0 in labels with no predicted samples.*")
warnings.filterwarnings('ignore', message="From version 0.21, test_size will always complement train_size unless both are specified")
warnings.filterwarnings('ignore', message="UndefinedMetricWarning.*")

# ------------------------------------------------
# Settings
# ------------------------------------------------

# Settings: select one from every list
feature_set =  ['eg', 'baseline', 'combined'][0]
class_type =  ['authors', 'genres'][1]
projection_type = ['pu','pw','pacc'][2]

baseline_f_dir  = "../2_Features/baseline_features_reduced/"
EG_f_dir = "../2_Features/EG_features/"

# Select projection type
EG_f_dir = EG_f_dir[:-1] + '_'+projection_type+'/'

# Number of authors
n_authors = 10

# True: most frequent category is considered solution for all samples
most_freq_cat = False

# Create a boxplot
boxplot = False
# Compute correlation between text length and PC1
correlation = False
# Balance sets (same number of texts for each author/genre; reduces sample size)
balanced = False
# Use PCA transformed data
use_transformed_data = False

# Correct boxplot settings if neccessary
if correlation:
    boxplot=True
    print("\nNote: Boxplot function has been activated.")
if feature_set == 'combined':
    boxplot = False
if boxplot:
    print("Note: Number of authors was automatically set to 20 due to boxplot computation.\n")
    n_authors = 20
    
# Set texttype version according to input
texttype_version = True if class_type == 'genres' else False
if texttype_version:
    n_authors = 30

# Only selected authors or texttypes
only_selected_authors = False

## If you want to use defined set of authors
authors_of_interest = []

# ------------------------------------------------
# Functions for plotting and PCA
# ------------------------------------------------

def plot_confusion_matrix(cm, title, categories):
    """Create and plot confusion matrix for categories."""

    plt.figure(figsize=(7.6,7.6))
    plt.imshow(cm, interpolation='none',cmap='Blues')
    for (i,j), z in np.ndenumerate(cm):
        plt.text(j, i, z, ha='center', va='center')
    plt.xlabel("prediction")
    plt.ylabel("ground truth")
    plt.title(title+' set')
    plt.gca().set_xticks(range(len(categories)))
    plt.gca().set_xticklabels(categories, rotation=45)
    plt.gca().set_yticks(range(len(categories)))
    plt.gca().set_yticklabels(categories)
    plt.gca().invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('./2_Results/plots/confusion_matrix_'+title+'_set.png', dpi=100, format='png', trasparent=True)

    plt.close()

def get_pandas_df(feature_array):
    """Return feature array as pandas dataframe."""
    
    # Load feature array as pandas df
    df = pd.DataFrame(data=feature_array)

    # Assign column names and authors
    df.columns = eg_feature_names if feature_set == 'eg' else baseline_feature_names
    df['author'] = solution_array
    
#    # Save principal components
#    df['PC1'] = transformed_data[:,0]
#    df['PC2'] = transformed_data[:,1]
#    df['PC3'] = transformed_data[:,2]
    
    return df

def principal_component_analysis(feature_array, full_analysis=False):
    """Principal component analysis for feature array."""
    
    # fit PCA
    pca = PCA()
    pca.fit(feature_array.astype(np.float))
    
    # principal components
    first_pc  =  pca.components_[0]
    second_pc =  pca.components_[1]
    third_pc  =  pca.components_[2]
    
    # Transform data
    transformed_data = pca.transform(feature_array)
    
    # if not full analysis necessary, return transformed data here
    if not full_analysis:
        return transformed_data

    # Plot principal components
    for i,j in zip(transformed_data, feature_array):
        plt.scatter(first_pc[0]*i[0], first_pc[1]*i[0], color='r')
        plt.scatter(second_pc[0]*i[1], second_pc[1]*i[1], color='c')
        plt.scatter(third_pc[0]*i[2], third_pc[1]*i[2], color='y')
        plt.scatter(j[0],j[1], color='b')

    plt.show()

    target_colors = {}
    for i in range(len(set_of_authors)):
        target_colors[set_of_authors[i]] = i

    color_array = np.linspace(0,1,num=len(set_of_authors))
    for i in range(color_array.shape[0]):
        target_colors[set_of_authors[i]] = color_array[i]


    plt.scatter(transformed_data[:,0],transformed_data[:,1], 
                c=[target_colors[key] for key in solution_array], cmap='rainbow',
                alpha=0.5, edgecolor='none')
    plt.show()

    # Plot explained variance ratio
    plt.plot(pca.explained_variance_ratio_)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.show()
    
    # Get pandas df
    df = get_pandas_df(feature_array)
    
    df['PC1'] = transformed_data[:,0]
    df['PC2'] = transformed_data[:,1]
    df['PC3'] = transformed_data[:,2]
    
    return transformed_data

def create_boxplot(feature_array, default_metric='PC1'):
    """Create boxplot for classes and metrics."""
    
    df = get_pandas_df(feature_array)
    
    variable = default_metric

    # Error for variable "author"
    if variable == 'author' or variable not in df.columns.values:
        raise IOError(variable+" is not a valid column for this plot!")

    # Get boxplot
    axes, bp = df.boxplot(column=variable, by='author', 
                                               figsize=(13,8), rot=50, fontsize=30, grid=False,
                                               return_type='both',sym='')[variable]

    for item in ['boxes', 'whiskers']:
        plt.setp(bp[item], color='k')

    for item in ['medians']:
        plt.setp(bp[item], color='k')
        
    # Set figure 
    fig = axes.get_figure()
    fig.suptitle('')
    plt.title("Boxplot of "+variable, fontsize=24)
    plt.title('')
    plt.xlabel('')
    plt.ylabel('PC1', fontsize=30)
    plt.yticks(fontsize=25)
    plt.xticks(ha='right')
    
    # Add jittered data
    authors = []
    x_labels = axes.get_xticklabels()
    for i,x_label in enumerate(x_labels):
            y = df.loc[:,variable][df.author==x_label.get_text()]  # Value
            authors.append((x_label.get_text()))
            x = np.random.normal(i+1, 0.06, size=len(y))   # Jitter
            plt.plot(x, y, 'k.',color='0.65', alpha=0.25)

    # Done
    plt.tight_layout()
    plt.savefig('./2_Results/plots/boxplot_'+variable+'.png', dpi=100, format='png', transparent=True)
    plt.savefig('./2_Results/plots/boxplot_'+variable+'.svg', dpi=100, format='svg', transparent=True)
    
    plt.close()
    

def compute_correlation(feature_array, file_list, file_length, ind_list):
    """Compute correlation between PC1 and total document length."""
    
    # Get pandas df
    df = get_pandas_df(feature_array)    
    
    # Add columns
    df['filename'] = file_list
    f_length_array = np.array(file_length)
    df['f_length'] = f_length_array

    #print(f_length_array)
    new_array = np.zeros((2, transformed_data[:,0].shape[0]))
    new_array[0,:] = transformed_data[:,0]
    new_array[1,:] = file_length
    
    grouped_means = df.groupby('filename').mean()
    slope, intercept, rvalue, pvalue, sterr = linregress(grouped_means['PC1'], 
                                                         grouped_means['f_length'])
    
    plt.figure(figsize=(8,8))
    plt.scatter(grouped_means['PC1'], grouped_means['f_length'], s=10, alpha=0.5)

    plt.plot([grouped_means['PC1'].min(), -intercept/slope],
             [grouped_means['PC1'].min() * slope + intercept, 
              - intercept / slope * slope + intercept], 'r')

    plt.tight_layout()
    plt.savefig('./2_Results/plots/correlation_plot.png', dpi=100, format='png', transparent=True)
    plt.savefig('./2_Results/plots/correlation_plot.svg', dpi=100, format='svg', transparent=True)
    
    plt.close()
    
    print('Correlation\n****************')
    print('r:      {:.3f}'.format(rvalue))
    print('p-value:', pvalue)
    print()
    
    return rvalue, pvalue
    
# ------------------------------------------------        
# Load files
# ------------------------------------------------

# Get all array files
allfiles = os.listdir(EG_f_dir)
array_files = [f for f in allfiles if f.endswith('.npy')]
        
# Get author array files
array_kafka_files =  [f for f in array_files if f.startswith('KA')]
array_kleist_files = [f for f in array_files if f.startswith('KL')]
array_schnitzler_files = [f for f in array_files if f.startswith('SCHN')]                   
array_zweig_files = [f for f in array_files if f.startswith('ZW')]
array_hoffmann_files = [f for f in array_files if f.startswith('HOFF')]
array_twain_files = [f for f in array_files if f.startswith('TWA')]

array_tieck_files = [f for f in array_files if f.startswith('TCK')]
array_gotthelf_files = [f for f in array_files if f.startswith('GTTH')]
array_eichendorff_files = [f for f in array_files if f.startswith('EICH')]
array_keller_files = [f for f in array_files if f.startswith('KEL')]
array_spyri_files = [f for f in array_files if f.startswith('SPY')]

array_bierbaum_files = [f for f in array_files if f.startswith('BIE')]
array_busch_files = [f for f in array_files if f.startswith('BUS')]
array_dauthendey_files = [f for f in array_files if f.startswith('DAUT')]
array_fontane_files = [f for f in array_files if f.startswith('FON')]
array_ganghofer_files = [f for f in array_files if f.startswith('GANG')]

array_gerstaecker_files = [f for f in array_files if f.startswith('GER')]
array_gleim_files = [f for f in array_files if f.startswith('GLE')]
array_grimm_files = [f for f in array_files if f.startswith('GRI')]
array_haltrich_files = [f for f in array_files if f.startswith('HAL')]
array_hebbel_files = [f for f in array_files if f.startswith('HEB')]
                
array_hofmannsthal_files = [f for f in array_files if f.startswith('HOFS')]
array_jeanpaul_files = [f for f in array_files if f.startswith('JEA')]
array_may_files = [f for f in array_files if f.startswith('MAY')]
array_novalis_files = [f for f in array_files if f.startswith('NOV')]
array_pestalozzi_files = [f for f in array_files if f.startswith('PES')]
                
array_poe_files = [f for f in array_files if f.startswith('POE')]
array_raabe_files = [f for f in array_files if f.startswith('RAA')]
array_scheerbart_files = [f for f in array_files if f.startswith('SCHE')]
array_schwab_files = [f for f in array_files if f.startswith('SCHW')]
array_stifter_files = [f for f in array_files if f.startswith('STI')]
                
array_storm_files = [f for f in array_files if f.startswith('STO')]
array_thoma_files = [f for f in array_files if f.startswith('THO')]
array_volkmann_files = [f for f in array_files if f.startswith('VLK')]

# Put all arrays together
all_array_files = [array_kafka_files, array_kleist_files, array_schnitzler_files, 
                   array_zweig_files, 
                   array_hoffmann_files, array_twain_files,                   
                   array_tieck_files, array_gotthelf_files, array_eichendorff_files, 
                   array_keller_files, array_spyri_files,                   
                   array_bierbaum_files, array_busch_files, array_dauthendey_files, 
                   array_fontane_files, array_ganghofer_files,                   
                   array_gerstaecker_files, 
                   array_grimm_files, 
                   array_haltrich_files, array_hebbel_files,                   
                   array_hofmannsthal_files, array_jeanpaul_files, array_may_files, 
                   array_novalis_files,                     
                   array_poe_files, array_raabe_files, array_scheerbart_files, 
                   array_schwab_files, array_stifter_files,                   
                   array_storm_files, array_thoma_files, array_volkmann_files]

# Set author names
all_author_names = ['Kafka','Kleist','Schnitzler','Zweig',
                    'Hoffmann', 'Twain', 
                'Tieck', 'Gotthelf', 'Eichendorff','Keller','Spyri', 
                'Bierbaum','Busch','Dauthendey','Fontane','Ganghofer', 
                'Gerstaecker',
                'Grimm','Haltrich','Hebbel', 
                'Hofmannsthal','JeanPaul','May','Novalis',  
                'Poe','Raabe','Scheerbart','Schwab','Stifter',
                'Storm','Thoma','Volkmann'] 

# How many samples per author
author_sample_dict = {all_author_names[i]:len(author) for (i,author) in enumerate(all_array_files)}

# Sort by number of samples and select authors with most samples
sorted_author_sample_dict = sorted(author_sample_dict, key=author_sample_dict.get, reverse=True)
selected_authors = sorted_author_sample_dict[:n_authors]

# Get sample number for author with least samples
MIN_NR_SAMPLES = author_sample_dict[selected_authors[-1]]

# If only defined set of authors is used
if only_selected_authors:
    selected_authors = sorted(authors_of_interest, key=lambda x: author_sample_dict[x], reverse=True)

# If sample set should be balanced (same number of samples for each author)
if balanced:    
    # Get files per author (balanced set)
    file_names = [[(f, all_author_names[i]) for f in random.Random(42).sample(author_files, MIN_NR_SAMPLES)] for i, author_files in enumerate(all_array_files) if all_author_names[i] in selected_authors]
    file_names = np.concatenate(file_names)   
    
else:
    # Get files per author
    file_names = [[(f, all_author_names[i]) for f in author_files] for i, author_files in enumerate(all_array_files) if all_author_names[i] in selected_authors]    
    file_names = np.concatenate(file_names)

# If texttypes are used instead of authors (discard Twain texts because they are not labeled)
if texttype_version:
    file_names = np.array([(file,author) for (file,author) in file_names if author != 'Twain'])
    

# Authors used in this experiment
set_of_authors = (list(set([author for (file, author) in file_names])))

# Get example (for n_features)
eg_sample_f = np.load(EG_f_dir+file_names[0][0])

all_samples = []
sample_author_dict = {}

# Shorten filename to text name and save text/author pairs
for sample, author in file_names:
    textname = re.search('[A-Z]+\_(.*?)(0*1)?\_',sample).group(1)
    all_samples.append((textname,author))
    
#    print(sample)
    
# Set of all text/author pairs, sort by author
all_samples = list(set(all_samples))
all_samples = [(text,author) for (text,author) in all_samples]
sorted_samples = sorted(all_samples,key=itemgetter(1))

# Print statements
print('Samples for authors:\n')
pprint.pprint(author_sample_dict)
print()
print('Selected authors:')
print(set_of_authors)
print()
print('# Files:', file_names.shape[0])
print('Min number of samples:', MIN_NR_SAMPLES)
print() 

# ------------------------------------------------ 
# Text types and text lengths
# ------------------------------------------------

# Load (text, author): texttype information
with open('./1_Info/misc/texttype_dict.pkl','rb') as infile:
    texttype_dict = pickle.load(infile, encoding='latin1')
    
# Load (text, author): file length
with open('./1_Info/misc/file_length_dict.pkl','rb') as infile:
    file_length_dict = pickle.load(infile, encoding='latin1')
        
# Set eg feature and solution arrays, and texttype solution
eg_feature_array = np.zeros((file_names.shape[0], eg_sample_f.shape[0]))
eg_solution_array = []
texttype_solution_array = []

# Unsuitable texttypes or genres
# e.g. because insufficient amount of samples or too general class
unsuitable_text_types = ['anthology', 'book', 'comics', 'inbook', 'letter', 
                         'poem', 'inbook','aphorism', 'autobio','report',
                         'satire', 'essay','tractate']
# Initialize
filtered_out_set = []
file_list = []
file_length = []
used_files = []
ind_list = []
counter = -1

# Limit number of samples per class
MAX_NR_SAMPLES = 300

# Initialize dict to count samples by class
n_samples_dict = {'comedy':0, 'drama':0, 'essay':0, 'fairy':0, 'fiction':0, 
                  'legend':0, 'narrative':0, 'novelette':0, 'tractate':0,'novel':0,
                  'preface':0, 'report':0, 'autobio':0, 'aphorism':0, 'satire':0, 'inbook':0, 'letter':0,
                  'comics':0, 'anthology':0, 'book':0, 'tragedy':0, 'poem':0}

# For (file, author) in set
for i, (file, author) in enumerate(file_names):
    
    textname_with_author = re.search('([A-Z]+\_.*?)(0*1)?\_',file).group(1)
    # If correlation between text length and PCA is measured
    if correlation:
        

        # Try to get file length by filename
        try:
            f_length = file_length_dict[textname_with_author]        
            if f_length<50:
                continue        
            file_length.append(f_length)
            file_list.append(textname_with_author)
            counter+=1        
        except KeyError:
            continue

        # Save index for documents not in sample list
        if textname_with_author not in used_files:
            ind_list.append(counter)
            
        # Save used files
        used_files.append(textname_with_author)
    else:
        used_files.append(file)
    
    # Get and save features
    features = np.load(EG_f_dir+file)
    eg_feature_array[i,:] = features
    
    # Save author as solution
    eg_solution_array.append(author)
    
    # Texttype/genre instead of author
    if texttype_version:
        
        # Get textname and texttype from dict
        textname = re.search('[A-Z]+\_(.*?)(0*1)?\_',file).group(1)
        textname_with_author = re.search(u'([A-Z]+\_.*?)(0*1)?\_',file).group(1)
        
        # Get texttype from dict
        texttype = texttype_dict[(textname, author)]

        # If not low-sample category
        if texttype not in unsuitable_text_types and n_samples_dict[texttype]<=MAX_NR_SAMPLES:

            # Save texttype
            texttype_solution_array.append(texttype)
            n_samples_dict[texttype] += 1
             
        else:
            # Save indices for discarded samples
            filtered_out_set.append(i)

# ------------------------------------------------
# Prepare and load data for SVM
# ------------------------------------------------

# Filter out all samples that only have 0-values 
eg_feature_array = eg_feature_array[~(eg_feature_array==0).all(1)]

# Load feature names
eg_feature_names = np.load('../2_Features/info/feature_names/EG_feature_names.npy')
#sorted_fnames = np.load('sorted_fnames.npy')
baseline_feature_names = np.load('../2_Features/info/feature_names/baseline_feature_names_reduced.npy')

# Get baseline samples
baseline_sample_f = np.load(baseline_f_dir+file_names[0][0])

# Set baseline array size (31 features)
baseline_feature_array = np.zeros((file_names.shape[0],31))

# Save syntactic features for each file
for i,(file,author) in enumerate(file_names):

    # Load
    features = np.load(baseline_f_dir+file)

    # Convert to list
    x_features = features.tolist()

    # Take first 10 and last 21 features
    reduced_features = x_features[:10]
    reduced_features += x_features[-21:]
    features = reduced_features

    # Save for each file
    baseline_feature_array[i,:] = features
    
# Select desired feature array, or combo
if feature_set == 'eg':
    feature_array = eg_feature_array
if feature_set == 'baseline':
    feature_array = baseline_feature_array
if feature_set == 'combined':
    feature_array = np.append(eg_feature_array, baseline_feature_array, axis=1)

# Solution array from entity graph (same as for baseline)
solution_array = np.array(eg_solution_array)

# If texttype version
if texttype_version:    
    # Discard samples from filtered out set
    feature_array = np.array([feature_array[i,:] for i in range(feature_array.shape[0]) if i not in filtered_out_set])    
    solution_array = np.array(texttype_solution_array)
    
    # Save all use classes
    set_of_texttypes = list(set(texttype_solution_array))

# Take care of NAN values
feature_array = np.nan_to_num(feature_array)

# Print statements
print('# Number of samples:', eg_feature_array.shape[0])
print('# Entity graph metrics features:', eg_feature_array.shape[1])
print('# Syntactic features:', baseline_feature_array.shape[1])
print() 

# Scale features
scaler = sklearn.preprocessing.StandardScaler()
feature_array = scaler.fit_transform(feature_array)  
    
# Get PCA for feature array and use transformed data if necessarys
transformed_data = principal_component_analysis(feature_array)
if use_transformed_data:
    feature_array = transformed_data[:,:1]
    
# Plot boxplot and/or correlation
# ------------------------------------------------

# If boxplot or correlation plot is to be computed
if boxplot:
    create_boxplot(feature_array)
    
if correlation:
    compute_correlation(feature_array, file_list, file_length, ind_list)
    print('Note: Correlation measure set is different from original set and will not yield same results.')

# ------------------------------------------------
# SVM settings
# ------------------------------------------------

RANDOM_STATE = 42
TRAIN_SIZE = 0.8

gamma_dict = {('authors','combined'):0.05,
              ('authors','baseline'):0.1, 
              ('authors', 'eg'):0.1,
              
              ('genres', 'combined'):0.01,
              ('genres', 'baseline'):0.1, 
              ('genres', 'eg'):0.05  }

C_dict = {('authors','combined'): 7.5, 
          ('authors','baseline'):5.0, 
          ('authors','eg'):7.5,
          
          ('genres','combined'): 5.0,
          ('genres','baseline'):5.0, 
          ('genres','eg'):2.5} 

gamma = gamma_dict[(class_type, feature_set)]
C = C_dict[(class_type, feature_set)]

# Split into train and test set
f_train, f_test, s_train, s_test = sklearn.model_selection.train_test_split(feature_array, solution_array,                                                             
                                                                             train_size=TRAIN_SIZE,
                                                                             stratify=solution_array,
                                                                             random_state=RANDOM_STATE)
# Print sample and class distribution for training set
print('Training set:', f_train.shape[0], 'samples\n')
for ttype, count in zip(np.unique(s_train,return_counts=True)[0],np.unique(s_train,return_counts=True)[1]):
    print(ttype,' ',count)

# Print sample and class distribution for training set
print()
print('Test set:', f_test.shape[0], 'samples\n')
for ttype, count in zip(np.unique(s_test,return_counts=True)[0],np.unique(s_test,return_counts=True)[1]):
    print(ttype,' ',count)
    
 
# ------------------------------------------------
# Train classifier and analyze results 
# ------------------------------------------------

# Train classifier
classifier = svm.SVC(random_state=RANDOM_STATE, cache_size=1000, gamma=gamma, C=C)
classifier.fit(f_train, s_train)

# Predict train and test set
pred_train = classifier.predict(f_train)
pred_test = classifier.predict(f_test)

# Get categories
categories = set_of_authors if not texttype_version else set_of_texttypes

# Get "baseline" results; guess most frequent class for all samples
if most_freq_cat:

    # Set prediction to most frequent class (results for guessing most frequent class)
    if class_type == 'authors':
        pred_test = len(pred_test)*['Fontane']
    else:
        pred_test = len(pred_test)*['narrative']

    # Print results
    print()
    print('Accuracy:\t',sklearn.metrics.accuracy_score(s_test, pred_test))
    print('Precision:\t', sklearn.metrics.precision_score(s_test, pred_test, average='macro'))
    print('Recall:\t\t',sklearn.metrics.recall_score(s_test, pred_test,average='macro'))
    print('F1:\t\t\t',sklearn.metrics.f1_score(s_test, pred_test,average='macro'))

    # Exit
    sys.exit()

## Confusion matrix info
cm_train = sklearn.metrics.confusion_matrix(s_train, pred_train, labels=categories)
cm_test = sklearn.metrics.confusion_matrix(s_test, pred_test, labels=categories)

# Plotting of confusion matrix
plot_confusion_matrix(cm_train, 'train', categories)
plot_confusion_matrix(cm_test, 'test', categories)

# ------------------------------------------------
# Cross Validation
# ------------------------------------------------

### C-SVC (non-linear)
cv_classifier = svm.SVC(random_state=RANDOM_STATE,cache_size=1000,gamma=gamma, C=C)
    
# Cross validation
cv_ = sklearn.model_selection.ShuffleSplit(n_splits=5, train_size=TRAIN_SIZE, 
                                           random_state=RANDOM_STATE)
# Get accuracies
accuracies = sklearn.model_selection.cross_val_score(cv_classifier, feature_array, 
                                                    solution_array, cv=cv_, 
                                                    n_jobs=5, scoring='accuracy')
# Get precision scores (macro)
precision_m = sklearn.model_selection.cross_val_score(cv_classifier, feature_array, 
                                                    solution_array, cv=cv_, 
                                                    n_jobs=5, scoring='precision_macro')
# Get recall scores
recall_m = sklearn.model_selection.cross_val_score(cv_classifier, feature_array, 
                                                    solution_array, cv=cv_, 
                                                    n_jobs=5, scoring='recall_macro')
# Get F1 scores (macro)
f1_m = sklearn.model_selection.cross_val_score(cv_classifier, feature_array, 
                                                    solution_array, cv=cv_, 
                                                    n_jobs=5, scoring='f1_macro')

# ------------------------------------------------
# Print results
# ------------------------------------------------

# Set feature set name
if feature_set == 'eg':
    feature_set_name = 'entity graph metrics'
elif feature_set == 'baseline':
    feature_set_name = 'syntactic'
else:
    feature_set_name = 'syntactic + entity graph'
    
# Print settings
print('\n\n**************\nSETTINGS\n**************\n')
print('Feature set:', feature_set_name)
print('Number of different classes:', n_authors)
print('Number of samples:', feature_array.shape[0])
print('Class types:', 'texttypes/genres' if texttype_version else 'authors')

# Print results
print('\nAccuracies:')
print('Mean:    {:.4f}'.format(np.mean(accuracies)))
print('St. Dev: {:.4f}'.format(np.std(accuracies)))
 
print('\nWeighted precision:')
print('Mean:    {:.4f}'.format(np.mean(precision_m)))
print('St. Dev: {:.4f}'.format(np.std(precision_m)))

print('\nWeighted recall:')
print('Mean:    {:.4f}'.format(np.mean(recall_m)))
print('St. Dev: {:.4f}'.format(np.std(recall_m)))

print('\nWeighted F1:')
print('Mean:    {:.4f}'.format(np.mean(f1_m)))
print('St. Dev: {:.4f}'.format(np.std(f1_m)))

# Save accuracies 
np.save('accuracies_'+feature_set_name+'.npy', accuracies)
