"""
Entity Graph for German
Author: Julia Suter, 2018/19
-----------------------------

GET_entity_graphs_for_corpus.py

    - Process (and if necessary parse) each file in source dir
    - Compute entity graph and information
    - Save entity graph for each sample in target dir

"""

# Import Statements

import RUN_entity_graph as eg
import config

import os
import re

def create_graphs_for_texts(source_dir, target_dir):
    """Create entity graphs for all texts contained in given directory."""
        
    # Get directory
    directory = source_dir   
    
    print('Source file:', source_dir)
    
    # Get list of all files in dir
    files = os.listdir("./"+directory)
     
    # Create target dir if it does not exist yet     
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    
    is_parsed = True
    
    # Set cunks of N sentences
    N_SENTS = 50
    
    # Max number of chunk samples per file
    MAX_SAMPLES = 50
    
    
    for f in files:
                            
            # Remove file ending (.txt)
            short_filename = target_dir+f[:-4]
            short_filename = re.sub(r'\..$','', short_filename)           
      
            # Full filename with directory
            filename = directory+f
           
            print('\t'+f)
                        
            # Get all sentences from path
            sentences = eg.get_sentences(filename, is_parsed)
            n_sentences = len(sentences)

            # If sentence chunk number is not met, discard
            if n_sentences<N_SENTS:
                continue
   
            # Number of possible chunk splits 
            splits = n_sentences//N_SENTS
            
            # For each split
            for i in range(splits):
                
                # Break if max number of samples is reached
                if i >= MAX_SAMPLES:
                    break
                
                # Sentences for this chunk 
                chunk_sents = sentences[i*N_SENTS:(i+1)*N_SENTS]

                # Create filename for this chunk
                split_filename = short_filename+'_'+str(i+1)

                # Compute coherence measure and save entity graph as numpy array
                eg.get_coherence_measure(chunk_sents, return_details=False, filename=split_filename)

 
# main function
if __name__ == '__main__':
    
    source_dir = '../0_Datasets/GUTENBERG_parsed_texts/'
    target_dir = '../1_Processed_texts/graphs_per_text/GUTENBERG_'+config.version+'/'
    
    create_graphs_for_texts(source_dir, target_dir)       