"""
Entity Graph for German
Author: Julia Suter, 2018/19
-----------------------------

config.py

- Config file for settings
- Set paths, weights and other entity graph settings
"""

# ------------------------------------------------
# Import Statenents
# ------------------------------------------------

import os 
from warnings import warn

# ------------------------------------------------
# Path settings
# ------------------------------------------------

PARSER_PATH = '/home/jsuter/repositories/'
    
ENTITY_GRID_FILE = 'entity_grid.csv'
PARSE_RESULTS_PATH = './0_CorZu_results'

# Make repository only if it does not exist
if not os.path.exists(PARSE_RESULTS_PATH):        
    os.makedirs(PARSE_RESULTS_PATH)

# TueBa/DZ dataset path
parsed_eval_docs_path = '../0_Datasets/Tueba_DZ/gold_docs'
unparsed_eval_docs_path = '../0_Datasets/Tueba_DZ/unparsed_docs'
preparsed_unparsed_eval_docs_path = '../0_Datasets/Tueba_DZ/parzu_parsed_docs'

# ------------------------------------------------
# Entity Graph Settings
# ------------------------------------------------

# Entity graph version:  [pu, pw, pacc]
version = 'pacc'

# pu = P unweighted
# pw = P weighted
# pacc/full = P accumulated

### syntacic roles for P and G
cat_p_on = True
cat_g_on = True
merge_p_and_g = True

# weight reduction
reduce_weights = True

# possession modifiers
possession_modifiers = True

if possession_modifiers != None:
    cat_p_on = possession_modifiers
    cat_g_on = possession_modifiers
    merge_p_and_g = possession_modifiers

# parsed or unparsed eval
parsed_eval_files = True
preparsed = True
   
# execute evaluation in main entity graph script
evaluation = False

# Default settings
passive_on = True
coref_on = True
dist_on = True
normalized = True

# ------------------------------------------------
# Syntactic roles
# ------------------------------------------------

# Weights for syntactic roles
synt_role_dict = {'S':3,
                  'O':2,
                  'P':3,
                  'G':3,                 
                  'X':1,
                  '_':0 }

# Reduction degree
REDUCTION = 0.75

# True =  acc obj, dat obj, gen obj, prep obj
# False = acc obj only 
all_obj_on = True

# ------------------------------------------------
# Settings depending on other settings
# ------------------------------------------------

# redirect from "full" to "pacc"
version = 'pacc' if version == 'full' else version

# Syntactic roles only for "full" version
synt_roles_on = True if version == 'pacc' else False

# Do not merge p and g if not both are active
merge_p_and_g = False if not (cat_p_on and cat_g_on) else merge_p_and_g

# ------------------------------------------------
# Writing out and graph drawing settings
# ------------------------------------------------

draw_graph = False
write_out = False

# Do not write single entities into entity grid to increase readability
reduced_entity_grid = False

# ------------------------------------------------
#  Drawing Graph with Care: For safety reasons...
# ------------------------------------------------

# Do not allow drawing graphs when evaluating (too many graphs)
if evaluation and draw_graph:
    draw_graph = False
    warn("Draw graph deactivatedy.")

# In general, ask if graphs really should be created and shown
if draw_graph:
    ans = input('Do you really want to draw the graph(s)? ')
    if not ans.lower().startswith('y'):
        draw_graph = False
