"""
Entity Graph for German
Author: Julia Suter, 2018/19
-----------------------------

RUN_evaluation.py

- Evaluate the entity graph on the sentence reordering task
- Compute performance for discrimination and insertion task
- Reproduce results from KONVENS paper (2018)
- Print and save the results

"""

# ------------------------------------------------
# Import Statements
# ------------------------------------------------


from __future__ import division

import numpy as np
import pandas as pd
import scipy.stats as stats

import copy
import os
import fnmatch

import re
import pickle

import time
import random
random.seed(42)

# Own modules
import config
import RUN_entity_graph as eg

# Multiprocessing
import multiprocessing
import multiprocessing.pool

activate_multiprocessing = True

GLOBALLOCK = multiprocessing.Lock()

# Load (text, author): texttype information for Literature texts
with open('1_Info/misc/texttype_dict.pkl','rb') as infile:
    texttype_dict = pickle.load(infile)
    
# author abbreviations    
author_dict = {'Kafka':'KA','Kleist':'KL', 'Keller':'KEL','Schnitzler':'SCHN',
               'Twain':'TWA','Eichendorff':'EICH','Hoffmann':'HOFF','Spyri':'SPY',
               'Gotthelf':'GTTH','Tieck':'TCK',
               'Dauthendey':'DAUT','Bierbaum':'BIE', 'Ganghofer':'GANG',
               'Gerstaecker':'GER','Fontane':'FON',
               'Gleim':'GLE','Haltrich':'HAL','Hebbel':'HEB','Novalis':'NOV',
               'Pestalozzi':'PES',
               'Poe':'POE','Hoffmannsthal':'HOFS','JeanPaul':'JEA','May':'MAY',
               'Raabe':'RAA',
               'Scheerbart':'SCHE','Schwab':'SCHW','Stifter':'STI','Storm':'STO',
               'Volkmann':'VLK',
               'Busch':'BUS','Wieland':'WIE','Thoma':'THO','Grimm':'GRI',
               'Zweig':'ZW'         }
    
inv_author_dict = {v: k for k, v in author_dict.items()}       

def discrimination(sentences,n, verbose=True):
    """Return accuracy of discrimination evaluation given parsed sentences."""

    # get entity grid seperately
    entity_grid = eg.get_entity_grid(sentences)
    
    # get avg out degree for original sentence order
    original_avg_out_degree = eg.get_avg_outdegree(entity_grid,len(sentences))
    
    # copy of entity grid for scrambling
    random_entity_grid = copy.deepcopy(entity_grid)

    successes = 0
    
    # shuffle sentences randomly and compare with original sentence order
    for i in range(1,n+1):
        np.random.shuffle(random_entity_grid)        
        avg_out_degree = eg.get_avg_outdegree(random_entity_grid,len(sentences))

        if original_avg_out_degree > avg_out_degree:
            successes += 1
    
    # accuracy    
    accuracy = successes/n
    
    if verbose:
        print('Coherence measure:  {:.4f}'.format(original_avg_out_degree))
        print('Disc Accuracy:      {:.4f}'.format(accuracy))
        
    return accuracy
    
    
def insertion(sentences, verbose=True, baseline=False):
    """Return insertion score and accuracy of insertion evaluation given parsed sentences."""
    
    successes = 0
    num_sentences = len(sentences)
    ins_score_list = []
    
    # get entity grid
    entity_grid = eg.get_entity_grid(sentences)
    
    # for each sentence: remove sentence, reinsert at all possible positions, select best reinsertion    
    for i in range(0, num_sentences):
        
        highest_avg_out_degree = 0
        best_inserted_position = 0     
        
        # write out reinserted sentence
        reinserted_sent = entity_grid[i,:]   

        # for each sentence
        for k in range(0, num_sentences):    
            
            # copy entity grid
            new_entity_grid = copy.deepcopy(entity_grid)
            # delete target sentence
            new_entity_grid = np.delete(new_entity_grid,i,0)
            # reinsert target sentence at new position
            new_entity_grid = np.insert(new_entity_grid,k,reinserted_sent,0)
            
            # compute avg out degree
            avg_out_degree = eg.get_avg_outdegree(new_entity_grid,num_sentences)
            
            # check if this insertion's average out degree is higher than currently highest degree
            if avg_out_degree >= highest_avg_out_degree:
                
                # if so, overwrite highest score and best position
                highest_avg_out_degree = avg_out_degree
                best_inserted_position = k
                
        if baseline:
            best_inserted_position = random.randint(0,num_sentences)
        
        # new variable assignment to match equation in paper by (Elsner and Charniak, 2007)
        p = best_inserted_position+1
        k = i+1
        n = num_sentences
     
        # insertion score as described by (Elsner and Charniak, 2007)
        ins_score = 1 - (((abs(p-k))*2*n)/((k*(k-1)+(n-k+1)*(n-k))))
        ins_score_list.append(ins_score)
                
        # count successes
        if i == best_inserted_position:
            successes += 1
        
    # average insertion score and accuracy
    avg_insertion_score = sum(ins_score_list)/num_sentences
    accuracy = successes/num_sentences
    
    if verbose:
            print('Ins score:          {:.4f}'.format(avg_insertion_score))
            print('Ins Accuracy:       {:.4f}'.format(accuracy))
      
    # return average insertion score and accuracy
    return avg_insertion_score, accuracy


def single_file_evaluation(arguments):
    """Get evaluation for single file; used in multiprocessing."""

    # unpacking of arguments
    file_arg, filter_on, filtered_corpus, directory, verbose, baseline = arguments
    
    counter = file_arg[0]+1
    file = file_arg[1]
  
    # if filtering, only use docs with IDs that were collected above
    if filter_on:
         
        # get file nr
        file_nr = int(re.search(r'_(\d+).txt', file).group(1))
        
        # discard doc if not in list
        if file_nr not in filtered_corpus:
            return None
          
    # print lock start
    GLOBALLOCK.acquire()
    
    # print document
    if verbose:
        print('\nDocument', counter, file)
 
    # create testfile name
    testfile = directory+"//"+file

    # get the parsed sentences from test doc   
    if config.parsed_eval_files or config.preparsed:
        test_sents = eg.get_sentences(testfile,True)

    else:
        test_sents = eg.get_sentences(testfile,False)
         

    # Discard if less than 10 sentences
    if len(test_sents) < 10:
        if verbose:
            print('Document was discared since it contains less than 10 sentences.')
        
        # print lock end
        GLOBALLOCK.release()
        return None
    
    # discrimination and insertion evaluation scores
    dis_acc = discrimination(test_sents, 50, verbose)
    ins_score, ins_acc = insertion(test_sents, verbose, baseline)
    
    # print lock end
    GLOBALLOCK.release()
    
    return (dis_acc, ins_score, ins_acc, len(test_sents))
        

def full_evaluation(verbose=False, directory=None, min_10_sents=False, min_5_entities=False, 
                    intersection=False, is_parsed=None, max_n_docs=None, only_first_part=False, baseline=False, 
                    tag='', texttype=None, normal_test=False):
    """Evaluate all documents found in directory.
    Filter option for documents with less than 10 sentences and/or less than 5 entities."""  
    
    # print settings
    
    print('\n***** SETTINGS *****\n')
    print('Version:\t', config.version)
    print('Data:\t\t','parsed files' if config.parsed_eval_files else 'unparsed files')
    
    print('\nCoref res.\t', 'ON' if config.coref_on else 'OFF')
    print('Passive res. \t','ON' if config.passive_on else 'OFF')
    print('Normalized\t', 'ON' if config.normalized else 'OFF')
    print('Poss. pron.\t','ON' if config.cat_p_on else 'OFF')
    print('Gen. mod.\t','ON' if config.cat_g_on else 'OFF')
    print('P and G merged\t', 'ON' if config.merge_p_and_g else 'OFF')
    print('\nReduced weights\t', 'ON' if config.reduce_weights else 'OFF')
    print('Reduction:\t', config.REDUCTION)
    print('\n********************\n\n') 

    # result lists
    dis_accuracies = []
    ins_scores = []
    ins_accuracies = []
    
    lengths = []   
    
    # get directories from config file
    if directory==None:
        directory = config.parsed_eval_docs_path if (config.parsed_eval_files) else config.unparsed_eval_docs_path
    
    # if unparsed but preparsed version available, use that!
    if (not config.parsed_eval_files) and config.preparsed:
        directory = config.preparsed_unparsed_eval_docs_path
    
    if is_parsed != None:
        directory = config.parsed_eval_docs_path if is_parsed else config.preparsed_unparsed_eval_docs_path
    
    # get number of docs
    n_docs = len(fnmatch.filter(os.listdir(directory), '*.txt'))
    
    # set to true if reduced set should be used
    filter_on = True if (min_10_sents or min_5_entities) else False
    
    filtered_corpus = None
    
    # for reduced datasets
    if filter_on:
        
        # total number of documents in dataset
        total = range(0,len(pickle.load(open("1_Info/misc/filenames.txt","rb"))))
         
        # IDs for docs with few ents and sents (including those filtered in "create_corpus_parsed")
        few_sents = pickle.load(open("1_Info/filter_IDs/few_sent_doc_IDs_all.txt", "rb" ))
        few_ents =  pickle.load(open("1_Info/filter_IDs/few_ent_doc_IDs_all.txt","rb"))

        # intersection and union of both versions
        few_intersection = [elem for elem in few_sents if elem in few_ents]
        few_union = list(set(few_sents+few_ents))
        
        # make sure total number is correct
        assert n_docs == len(total)  
        
        # full corpus (to start with)
        filtered_corpus = total
        
        # if both filters on
        if min_10_sents and min_5_entities:
            
            # intersection
            if intersection:
                filtered_corpus = [x for x in total if x not in few_intersection]
                
            # union
            else:
                filtered_corpus = [x for x in total if x not in few_union]
        
        # only sents
        if min_10_sents and not min_5_entities:
            filtered_corpus = [x for x in total if x not in few_sents]
            
        # only ents
        if min_5_entities and not min_10_sents:
            filtered_corpus = [x for x in total if x not in few_ents]
    
        # new number of documents
        n_docs = len(filtered_corpus)
    
    print(n_docs, 'docs are being processed...')
    
    # get starting time
    start_time = time.time()

    if activate_multiprocessing:

        # print number of CPU
        print(multiprocessing.cpu_count(), 'cpu count')

        # prepare multiprocessing pool
        my_pool = multiprocessing.Pool(processes=5)

    # list of all files
    files = os.listdir("./"+directory) if not directory.startswith('/home') else os.listdir(directory)

    # If only first part/chunk of each text is used 
    if only_first_part:
        
        # Only get files that end in _1        
        files = [f for f in files if f.endswith('_1.txt')]
        
    # if only specific text type / genre is used
    if texttype!=None:
        
        print('Searching for genre ',texttype, '...')
                
        final_files = []
        for file in files:

            # extract author from filename
            try:
                author = re.search('([A-Z]+)\_(.*)', file[:-4]).group(1)  
            except AttributeError:
                continue
            
            # extract filename
            try:
                filename = re.search('([A-Z]+)\_(.*)', file[:-4]).group(2) 
            except AttributeError:
                continue
            
            # get author by abbreviation
            author = inv_author_dict[author]
     
            # Use dict to look up texttype for given file and author
            try:
                ttype = texttype_dict[(filename, author)]
            except KeyError:
                continue
            
            # only use if it matches required texttype
            if ttype == texttype:
                final_files.append(file)
                
        files = final_files

    # If max n of docs is set, only use first n docs
    if max_n_docs != None:
        files = files[:max_n_docs]  

    if activate_multiprocessing:

        # create argument tuples
        arguments = [(f, filter_on, filtered_corpus, directory, verbose, baseline) for f in enumerate(files)]

        # start multiprocess and save results
        results = my_pool.map(single_file_evaluation, arguments)

    else:
        # Save results
        results = []

        # Iterate instead of multiprocess: useful for debugging
        for i,f in enumerate(files):
            args = ((i,f), filter_on, filtered_corpus, directory, verbose, baseline)
            out = single_file_evaluation(args)
            results.append(out)

    # get end time
    end_time = time.time()
    print('\n\n***\nRun time: %s minutes\n***' % str((end_time-start_time)/60),'\n')
    
    # unpack results into lists
    dis_accuracies = [r[0] for r in results if r != None]
    ins_scores     = [r[1] for r in results if r != None]
    ins_accuracies = [r[2] for r in results if r != None]
    lengths        = [r[3] for r in results if r != None]
        
    # Save results from sentence ordering tasks 
    np.save('./2_Results/sent_reordering_results/disc_accs'+tag+'.npy', dis_accuracies)
    np.save('./2_Results/sent_reordering_results/ins_scores'+tag+'.npy', ins_scores)
    np.save('./2_Results/sent_reordering_results/ins_accs'+tag+'.npy', ins_accuracies)
        
    # Use metrics
    metrics = ['Disc. accuracies', 'Insertion scores', 'Insertion accuracies']
    
    # assign number of docs
    if max_n_docs != None:
        n_docs = len(ins_scores)

    # make sure number of docs is correct
    assert len(ins_scores) == n_docs
    
    # Normal test only if more than 20 docs/results avaiable
    if normal_test and n_docs>=20: 
        
        # Normal test for results
        print('\nNormal test\n*************')
        for scores, metric in zip([dis_accuracies, ins_scores, ins_accuracies], metrics):
            k2, p = stats.normaltest(scores)
            print(metric, 'P:\t', p)
        print()
                 
    # print average accuracies and scores
    if verbose:
        print('Average Discrimination Acc: {:.4f}'.format(sum(dis_accuracies)/n_docs))
        print('Average Insertion Score:    {:.4f}'.format(sum(ins_scores)/n_docs))
        print('Average Insertion Acc:      {:.4f}'.format(sum(ins_accuracies)/n_docs))
    
    # create dataframe and fill with results
    df = pd.DataFrame()
    
    a= sum(dis_accuracies)/n_docs
    b= sum(ins_scores)/n_docs
    c= sum(ins_accuracies)/n_docs
    
    df['Average Discrimination Acc'] = [a]
    df['Average Insertion Score'] = [b]
    df['Average Insertion Acc'] = [c]
        
    # put in table format
    table_results = [['Average Discrimination Acc', a],['Average Insertion Score', b],  ['Average Insertion Acc', c]]
        
    if not verbose:
        
        print('\n\n'+50*'*')        
        print('Average Discrimination Acc: {:.4f}'.format(table_results[0][1]))
        print('Average Insertion Score:    {:.4f}'.format(table_results[1][1]))
        print('Average Insertion Acc:      {:.4f}'.format(table_results[2][1]))     
    
    return table_results, df
        

def evaluation_for_paper():
    """Call function for recreating results from paper Suter and Strube (2018)."""
    # config: pu, pw, full
    full_evaluation(verbose=False, min_10_sents=True, is_parsed=True)
     
     
    
if __name__ == '__main__':         
    if config.draw_graph:
        raise IOError('You cannot evaluate while draw_graph is set to True. Set to False in config.py and try again.')
    
    evaluation_for_paper()    

    