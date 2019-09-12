"""
Entity Graph for German
Author: Julia Suter, 2018/19
-----------------------------

RUN_entity_graph.py

- Entity graph module
- Read and parse input files or directory
- Pass parsed text to entity grid and graph modules
- Compute average out-degree
- Compute and display entity graph visualization

"""

# ------------------------------------------------
# Import Statements
# ------------------------------------------------

from __future__ import division

import os
import sys
import shutil
import re
import pickle
import codecs

import subprocess
from subprocess import PIPE

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import itemfreq

# Own modules
import create_entity_grid as egrid
import create_entity_graph as egraph

import config
import RUN_evaluation

import warnings
warnings.filterwarnings('ignore', message="The is_string_like function.*")
# warnings.filterwarnings('ignore', message="QApplication: invalid style override passed, ignoring it.*")
# warnings.filterwarnings('ignore', message="MatplotlibDeprecationWarning.*")
# warnings.filterwarnings('ignore', message="The iterable function was deprecated in Matplotlib 3.1.*")


def get_sentences_from_unparsed_text(doc, save_in_dir):
    """Return ParZu and CorZu parsed sentences for given document. Save parsed document if required."""

    # Delete previous parsing results (if existing)
    if os.path.exists(config.PARSE_RESULTS_PATH+'/coref.conll'):
        os.remove(config.PARSE_RESULTS_PATH+'/coref.conll')
    if os.path.exists(config.PARSE_RESULTS_PATH+'/coref.html'):
        os.remove(config.PARSE_RESULTS_PATH+'/coref.html')
        
    input_type = 'cat'
    CorZu_type = 'CorZu'
     
    # Command line string
    # Parse document and store results in /CorZu_results  
    cmd = "%(type)s %(filename)s | " \
          "python %(parse_path)s/ParZu_NEW/parzu -q -o conll > "\
          "%(parse_res_path)s/parsed.conll && "\
          "python %(parse_path)s/%(corzu)s/extract_mables_from_conll.py "\
          "%(parse_res_path)s/parsed.conll > "\
          "%(parse_res_path)s/markables.txt && "\
          "python %(parse_path)s/%(corzu)s/corzu.py "\
          "%(parse_res_path)s/markables.txt "\
          "%(parse_res_path)s/parsed.conll > "\
          "%(parse_res_path)s/coref.conll "\
          "&& python %(parse_path)s/%(corzu)s/conll_to_html.py "\
          "%(parse_res_path)s/coref.conll > "\
          "%(parse_res_path)s/coref.html" % {'corzu':CorZu_type, 'type':input_type, 'filename': doc,'parse_path':config.PARSER_PATH, 'parse_res_path':config.PARSE_RESULTS_PATH}
  

    # Execute
    process = subprocess.Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True)
    stdout, stderr = process.communicate()
    
    # Catch parsing errors from ParZu or CorZu by checking for output files
    if not os.path.isfile(config.PARSE_RESULTS_PATH+'/parsed.conll'):
        raise IOError('Sorry, CorZu failed. Coref.conll file does not exist. Try another document.')
    else:
        with open(config.PARSE_RESULTS_PATH+'/parsed.conll', "r") as infile:
            infile = infile.read()
            if len(infile)<1:
                raise IOError('Sorry, ParZu failed. No parsing results.')
        
    if not os.path.isfile(config.PARSE_RESULTS_PATH+'/coref.conll'):
        raise IOError('Sorry, CorZu failed. Coref.conll file does not exist. Try another document.')
    
    
    # Open the parsed result file, split at sentence boarders and get single sentences
    with open(config.PARSE_RESULTS_PATH+'/coref.conll', "r") as infile:
        infile = infile.read()
        sentences = infile.split('\n\n')[:-1]
     
    # If filename for saving is given, save parsing results
    if save_in_dir:
        shutil.copy2(config.PARSE_RESULTS_PATH+'/coref.conll', save_in_dir)
        
    return sentences
    
def get_sentences_from_parsed_text(doc):
    """Return sentence list of a document with already parsed text in CoNNL format."""
    
    # Open file and split at sentence boarders    
    with open(doc,"r") as infile:
        infile = infile.read()
        sentences = infile.split('\n\n')[:-1]

    return sentences
    
    
def get_sentences(filename, is_parsed=True, save_in_dir=None):
    """Direct to corresponding function for parsed or unparsed documents."""    

    if is_parsed:   
        sentences = get_sentences_from_parsed_text(filename)
    else:
        sentences = get_sentences_from_unparsed_text(filename, save_in_dir)
             
    # Rewrite the coref tags for easier processing
    sentences = egrid.rewrite_coref_tags(sentences)    
    
    return sentences

def draw_graph(entity_graph, filename=None, edge_weights_factor=20):
    """Draw and save entity graph."""
       
    # Add a zero row/column to end of array (for empty node in circle to indicate end of text)
    entity_graph_zero = np.zeros((entity_graph.shape[0]+1, entity_graph.shape[1]+1))
    entity_graph_zero[:-1,:-1] = entity_graph
    
    # Create a network graph from array
    D = nx.from_numpy_matrix(entity_graph_zero)  
    
    # Position of nodes
    pos = nx.shell_layout(D) # or spring_layout
    
    # Values for node color list and weights
    integer_list = range(0,entity_graph.shape[0]) 
    weights = np.array([i[2]['weight']*edge_weights_factor for i in D.edges(data=True)])

    # Label dictionary (+1 so first sentence == index 1, last node without label)
    label_dict = {i:i+1 for i in range(0, entity_graph.shape[0])}
    label_dict[entity_graph.shape[0]] = ''
    
    # Save labels and weights as attributes (for easy access later)
    D.my_label = label_dict
    D.my_weights = weights
    D.my_integers = integer_list    
    D.my_nodelist = range(0,entity_graph_zero.shape[0]-1)
    
    # Alpha values based on degree
    alpha_values = [D.degree()[i] for i in range(entity_graph.shape[0]+1)]
    norm_alpha_values = [(((x - min(alpha_values))/(max(alpha_values) - min(alpha_values))))*10 for x in alpha_values[:-1]]
    
    # Set figure size  
    plt.figure(figsize=(10.0,10.0))
    
#    # Draw edges 
    nx.draw_networkx_edges(D,pos,width=weights)
    
#    # Draw rainbow nodes
#    nx.draw_networkx_nodes(D,pos,cmap=plt.get_cmap('gist_rainbow'), node_color=integer_list, node_size=400, nodelist = range(0,entity_graph_zero.shape[0]-1))
#    
#    # Draw gray nodes    
#    nx.draw_networkx_nodes(D,pos, node_color='gray', edge_color='k', node_size=400, nodelist = range(0,entity_graph_zero.shape[0]-1))
    
    # Draw orange nodes
    nx.draw_networkx_nodes(D,pos, node_color=norm_alpha_values, edge_color='k',
                            cmap=plt.cm.Oranges,vmax=15, node_size=400, 
                            nodelist = range(0,entity_graph_zero.shape[0]-1))
     
    nx.draw_networkx_labels(D,pos,labels=label_dict,font_size=10)
    
    # Graph display settings
    plt.axis('off')
    plt.gca().set_aspect('equal')
    plt.tight_layout()

    # Save entity graph as numpy array if filename is given
    if filename:
        
        # Save and show graph
        plt.savefig(filename+'.png', dpi=50)
        plt.savefig(filename+'.eps', format='eps')
        plt.savefig(filename+'.svg', format='svg')
             

    plt.show()
    plt.close()


def get_entity_grid(sentences):
    """Compute the entity grid given list of parsed sentences."""
    
    # Get parsed text with accessible token information
    parsed_text = egrid.get_sentence_token_information(sentences)
 
    # Get entities with syntactic roles
    entities_with_roles = egrid.get_entities_with_roles(parsed_text)
    
    # Get clean entity set and entity counter, with resolved coref
    entities_by_sent,entity_counter  = egrid.get_entity_collection_and_counter(entities_with_roles)

    # Get the entity grid and list of all entities
    entity_grid, entity_list = egrid.get_entity_grid(entities_by_sent,entity_counter)

    # Write entity grid to csv outfile
    if config.write_out:
        egrid.write_to_outfile(entity_grid, entity_list, entity_counter)
        
    # Get entity grid as numpy array  
    grid_array = egrid.get_grid_array(entity_grid, entity_list)    
    
    
    return grid_array
    
    
def get_avg_outdegree(entity_grid,len_sentences, return_details=False, filename=None, draw_single_graph=False):
    """Compute the average outdegree for entity graph given entity grid."""
    
    # Get entity graph and avg out degree
    entity_graph = egraph.get_entity_graph(entity_grid)
    avg_out_degree = egraph.get_avg_out_degree(entity_graph,len_sentences)
       
    # Number of all sentence pairs and the non zero cells
    all_pairs = sum(range(entity_graph.shape[0]))
    nonzeros = (np.where(entity_graph!=0)[0]).shape[0]
  
    # Get percentage of connected sent pairs
    percentage_connected_sent_pairs = nonzeros/all_pairs*100 if all_pairs != 0 else 0
    
    # Draw graph
    if config.draw_graph or draw_single_graph:
        if not filename and draw_single_graph:
            filename = "./2_Results/entity_graphs/entity_graph"
        draw_graph(entity_graph,filename=filename)
      
    # Save entity graphy
    if filename:
        pickle.dump(entity_graph, codecs.open(filename+'_entgraph.npy', 'wb'))
        
    return (avg_out_degree, percentage_connected_sent_pairs) if return_details else avg_out_degree
    
def get_coherence_measure(sentences, return_details=False, filename=None, draw_single_graph=False):
    """Compute entity grid and graph and return average out-degree for given sentences (= coherence measure).
    If details are required, return details about the entity grid: # entities, avg frequecies of entities, avg number of ents/sent, 
    percentage of unconnected sent pairs."""
    
    # Get entity graph and avg out degree    
    entity_grid = get_entity_grid(sentences)

    avg_out_degree = get_avg_outdegree(entity_grid,len(sentences), filename=filename, draw_single_graph=draw_single_graph)   

    # Compute frequencies and other details about entity grid
    if return_details:
        
        # Get avg_out_degree and unconntected_sent_pairs
        avg_out_degree, percentage_unconnected_sent_pairs = get_avg_outdegree(entity_grid,len(sentences), return_details=True)  
           
        # Get avg freq of discourse entities
        number_entities = entity_grid.shape[1] 
        freqs_of_disc_ents = itemfreq(np.nonzero(entity_grid)[1])
        avg_freqs_of_disc_ents = np.sum(freqs_of_disc_ents[:,1])/number_entities   
        
        # Get avg number of entities per sent         
        number_of_ents_per_sent = itemfreq(np.nonzero(entity_grid)[0])
        avg_number_of_ents_per_sent = np.sum(number_of_ents_per_sent[:,1])/len(sentences)
    
        return (avg_out_degree, len(sentences), number_entities, avg_freqs_of_disc_ents, avg_number_of_ents_per_sent, percentage_unconnected_sent_pairs)

    return avg_out_degree
       


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
            sentences = get_sentences(filename, is_parsed)
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
                get_coherence_measure(chunk_sents, return_details=False, filename=split_filename)

        
def parse_texts_in_dir(directory, save_dir, nested=False):
    """Parse and save texts in given (nested) directory."""    
    
    # If nested directory, repeat procedure for every subdir
    if nested:    
        
        # For each subdir
        for subdir in os.listdir(directory):     
            
            # For each file
            for file in os.listdir(directory+subdir):    
                                   
                      # Get file path and new file name
                      file_path = directory+subdir+'/'+file                                       
                      new_file_name = save_dir+'/'+subdir+'_'+file
                      
                      # Get sentences and parse file in the process
                      get_sentences(file_path, is_parsed=False, save_in_dir=new_file_name)
                      print(new_file_name)
                      
                      
    else:   
        # For each fiile
        for file in os.listdir(directory):  
            
                   # Get file path and new file name
                   file_path = directory+'/'+file                    
                   new_file_name = save_dir+'/'+file
                   
                   # Get sentences and parse file in the process
                   get_sentences(file_path, is_parsed=False, save_in_dir=new_file_name)
                   print(new_file_name)
 

 
def test_driver(filename, is_parsed=False, to_be_evaluated=False, draw_single_graph=True):
    """Read file and compute entity graph and coherence measure."""

    # Catch if file does not exist
    if not os.path.isfile(filename):        
        raise IOError('File %s does not exist'  % filename)
            
    # Check if file is parsed
    with open(filename,'r') as infile:
        infile = infile.read()     
        if infile.startswith('1\t'):   
            is_parsed = True   
            
    # Get parsed sentences
    sentences = get_sentences(filename, is_parsed)
       
    # Get average out degree
    original_avg_out_degree = get_coherence_measure(sentences, draw_single_graph=draw_single_graph)
    print('Original: {:f}'.format(original_avg_out_degree))
    

    # Evaluation
    if to_be_evaluated or config.evaluation:
        evaluation.insertion(sentences)
        evaluation.discrimination(sentences,100)
    
# ------------------------------------------------
# Main function
# ------------------------------------------------

if __name__ == '__main__':
    
    # If no file is given use default
    if len(sys.argv)==1:

        # Set filename and version
        filename = './1_Info/single_docs/KAFKA_Tuersteher.txt'
        test_driver(filename, draw_single_graph=True)

    # If file is given
    else:
        # Get filename
        filename = sys.argv[1]
        
        # Is text pre-parsed?
        is_parsed = True if len(sys.argv)>2 and sys.argv[2].startswith('-p') else False
        
        # Call test driver
        test_driver(filename, is_parsed=is_parsed, draw_single_graph=True)  