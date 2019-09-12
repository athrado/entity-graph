"""
Entity Graph for German
Author: Julia Suter, 2018/19
-----------------------------

create_entity_graph.py

    - Entity graph module
    - Compute entity graph for each sample text
    - Compute coherence measure (average out-degree)
    - Compute for three different projections (pu, pw, pacc)
    - Save and write out entity graph

"""


# Import Statements

from __future__ import division


import numpy as np
import config

def safe_division_by_zero(term_a, term_b):
    """Safe division (does not error when division by 0). 
    Replace NaN with 0. Use with care!"""
    
    with np.errstate(divide='ignore', invalid='ignore'):           
            
            # divison
            result = term_a/term_b
            
            # correct division by zero
            result = np.nan_to_num(result)
                        
    return result
    
def get_entity_graph(grid_array):
    """Compute entity graph given grid array."""
      
    # NORMALIZATION of the entity graph (does not affect pu version)     
    if config.normalized and config.version != 'pu':
        
        # get weighted edge
        W_norm = safe_division_by_zero(grid_array, np.sum(grid_array, axis=1, keepdims=True))    
        sum_E = np.sum(grid_array!=0 , axis=1, keepdims=True)
        
        # sum of sentences in common divided by all sentences
        E_norm = safe_division_by_zero(sum_E, np.add(sum_E,sum_E.T))
                
        # create empty new array
        grid_array_new = np.zeros_like(E_norm)  
        
        # number of sentences
        n_sents = W_norm.shape[0]
        
        # normalize graphW
        for i in range(n_sents):
            
                a1 = (W_norm * W_norm[i]) > 0        
                tmp = ( (W_norm[i]*E_norm[i,:].reshape(-1,1)) + (W_norm[:]*E_norm[:,i].reshape(-1,1)))        
                tmp[~a1] = 0.0
                grid_array_new[i,:] = np.sum(tmp, axis=1)
    
        # delete lower triangular part
        grid_array_new = np.triu(grid_array_new,k=1)  
        
        grid_array = grid_array_new

    else:    
        # dot product grid array 
        grid_array = np.dot(grid_array,grid_array.T)
        grid_array = np.triu(grid_array,k=1)
        
        
#    # scale between 1 and 0 (not used)
#    grid_array =  safe_division_by_zero(((grid_array - grid_array.min())), (grid_array.max() - grid_array.min()))

    # binary values for pu
    if config.version == 'pu':
        grid_array = (grid_array != 0).astype(np.float)

    # distance of sentences k and i as array
    ki = np.arange(grid_array.shape[0])
    ki = np.expand_dims(ki,1)
    ki_distance = np.abs(ki-ki.T)
           
    # divide by k-i distance
    if config.dist_on:
        
        # normalize by k-i distance
        entity_graph_array = safe_division_by_zero(grid_array, ki_distance)   
        
    else:
        entity_graph_array = grid_array
    

    return entity_graph_array
    
def get_avg_out_degree(entity_graph,num_of_sents):
    """Get average out degree of projection graph"""
    
    # sum of all entity graph cells divided by number of sents
    avg_out_degree = np.sum(entity_graph)/num_of_sents

    return avg_out_degree
    

# main function
if __name__ == '__main__':       
 
     grid_array = np.load('1_Info/example_array.npy')
     get_entity_graph(grid_array)  