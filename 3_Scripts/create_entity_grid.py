"""
Entity Graph for German
Author: Julia Suter, 2018/19
-----------------------------

create_entity_grid.py

    - Entity grid module
    - Compute entity grid based on parsed text
    - Compute for three different projections (pu, pw, pacc)
    - Save and write out entity grid.

"""


# Import Statements

from __future__ import division

import itertools
from itertools import chain
from operator import itemgetter
from collections import defaultdict

import numpy as np

import parse_information as parse_info
import config


def get_sentence_token_information(sentences):
    """Return parsed sentence as nested sentence-token-information list."""
    
    parsed_text = []
    
    for sent in sentences:
        parsed_sentence = []
    
        # split sentence into tokens 
        tokens = sent.split('\n')
        for token in tokens:
            
            # split token string into token information
            parsed_token = [t for t in token.split('\t')]
            parsed_sentence.append(parsed_token)
            
        parsed_text.append(parsed_sentence)
        
    return parsed_text
    
    
def rewrite_coref_tags(sentences):
    """Adjust the coreference tags given by CorZu. 
    E.g. remove brackets, resolved nested/multile coreference tags."""
    
    parsed_text = get_sentence_token_information(sentences)   
    new_sentences = []
              
    for sentence in parsed_text:
        
         # transform tokens into Token class instances
        tokens = [parse_info.Token(k) for k in sentence]
        
        # adjust the corefence information for easier processin
        parse_info.adjust_coref(tokens)
        
        sentence_string = parse_info.sentenceToString(tokens)
        new_sentences.append(sentence_string)
    
    return new_sentences
    
    
def get_entities_with_roles(parsed_text):
    """Get entities with syntactic role S,O,X,P or G.
    If required, reduce weights for embedded entities."""
    
    # Initializing
    entities_with_roles = []
    passive_counter = 0
        
    for sentence in parsed_text:          
        
        # transform tokens into Token class instances
        tokens = [parse_info.Token(k) for k in sentence]
        
        # resolve and count passive constructions if necessary
        if config.passive_on:
            passives = parse_info.adjust_passive(tokens)
            if passives == True:
                passive_counter += 1
        
        # transform tokens into Sentence class instance
        sent = parse_info.Sentence(tokens)
        
        # get all subjects and objects of Sentence
        # get subject and object lemma if subj/obj is a noun or is marked as coreferent entity
        
        subjs = sent.subj()
        subjs_lemma = [t for t in subjs if (t.sim_pos == 'N' or t.coref != '_')]
       
        objs = sent.obj()
        objs_lemma = [t for t in objs if (t.sim_pos == 'N' or t.coref != '_')]
        
        # get all words from full subj and obj noun phrases (for excluding words later in the 'other' category)
        full_subjs = [t for t in list(chain.from_iterable([parse_info.get_full_phrase(tokens,subj) for subj in subjs])) if (t.sim_pos == 'N' or t.coref != '_')]
        full_objs  = [t for t in list(chain.from_iterable([parse_info.get_full_phrase(tokens,obj) for obj in objs])) if (t.sim_pos == 'N' or t.coref != '_')]
        
        # get all possessive pronouns (category 'P')  
        poss_pronouns = [t for t in tokens if (t.coref != '_' and (t.full_pos == 'PPOSAT'))]

        # get all genitive modifiers (category 'G')
        genitive_mods = [t for t in tokens if ((t.coref != '_' or t.sim_pos=='N') and t.function == 'gmod')] 
 
        # get all nouns that are not contained in the subj or obj noun phrase, or genitive modifiers 
        others = [t for t in tokens if ((t.sim_pos == 'N') and t not in subjs_lemma+objs_lemma)] 
        
        # get prepositions
        preps = [t for t in tokens if t.function == 'pp']
                       
        # if genitive cat is on, remove genitives from 'others'
        if config.cat_g_on:
            others = [t for t in others if t.function != 'gmod']
            
            # assign cat G to genitive modifiers; or merge with category P into X
            for g in genitive_mods:
                if not config.merge_p_and_g:
                    g.tag = 'G'
                
                # if category P and G are merged into one (X)
                else:
                   
                    if g in full_subjs:
                        subjs_lemma.append(g)
                    elif g in full_objs:
                        objs_lemma.append(g)
                    else:
                        others.append(g)
                        g.tag = 'X'
                        
        # Assign tag X to "other" category tokens
        for x in others:
            x.tag = 'X'   
            
        # if possessive category is on,
        if config.cat_p_on:
            
            # assign cat G to possessive pronouns, or merge with category G
            for p in poss_pronouns:    
                if not config.merge_p_and_g:
                    p.tag = 'P'
                    
                # if category P and G are merged into one (X)
                else:
                    if p in full_subjs:
                        subjs_lemma.append(p)                    
                    elif p in full_objs:
                        objs_lemma.append(p)
                    else:
                        others.append(p)
                        p.tag = 'X'
                      
        # Assign tag O to objects
        for o in objs_lemma:
            o.tag = 'O'
                
        # Assign tag S to subjects
        for s in subjs_lemma:
            s.tag = 'S'

        # get prepositional phrases   
        prep_phrase = [(p_ent) for (p_ent, prep, ent) in itertools.product(tokens, preps, tokens)  
                         if p_ent.function == 'pn' and 
                         p_ent in subjs_lemma+objs_lemma+others+poss_pronouns+genitive_mods + full_subjs+full_objs and
                         p_ent.dependency == prep.position and
                         prep.dependency == ent.position and
                         (ent.function == 'pn')] 
                         
        # get rel pronouns
        rel_prons = [t for t in tokens if t.full_pos == 'PRELS']
        
        # get rel clauses
        rel_clauses = [(k,j) for (k,j) in itertools.product(rel_prons, tokens)     
                        if j.function in ['rel','cj', 'objc']
                        and j.full_pos.endswith('FIN') and j.position > k.position]
            
        # mark relative clause tokens
        for (rel_pron, rel_pred) in rel_clauses:
            for token in tokens:
                if token.position >= rel_pron.position and token.position <= rel_pred.position:                
                    token.rel = True   
               
                    
        # get conjunction candidates
        conjunctions = [t for t in tokens if t.full_pos == 'KOUS' and t.function == 'konj']
        
        # get conjunctions and predicates
        conj_pred = [(k,j) for (k,j) in itertools.product(conjunctions, tokens) if j.full_pos.startswith('V') 
                                                                       and j.full_pos.endswith('FIN')
                                                                       and j.function in  ['root','neb'] 
                                                                       and j.position == k.dependency ]
                                                                       
        # Mark all tokens within subjunctional clause
        for k,j in conj_pred:
                                                                
            for t in tokens:
                if t.position >= k.position and t.position <= j.position:
                    t.subj = True
                    
        # get part presense and past
        part_pres = [t for t in tokens if t.full_pos == 'ADJD' and t.morph.part == '<PPRES' and t.function in  ['root','pn']]
        part_praet = [t for t in tokens if t.full_pos == 'VVPP' and t.function == 'neb']
        
        # for each participle
        for part in part_pres+part_praet:
            
            # get full participle construction
            part_con = parse_info.get_dependent_tokens(tokens, part) + [part]
            part_con = parse_info.get_all_tokens(part_con, tokens)
            
            # set initial comma positions
            first_comma_position = None
            sec_comma_position = None
            
            # find comma positions
            for comma in [t for t in part_con if t.lemma == ',']:
                if comma.position < part.position:
                    first_comma_position = comma.position
                if comma.position > part.position:
                    sec_comma_position = comma.position
        
            # cut participle construction at commas (only in-between)
            part_con = [k for k in part_con 
                        if (first_comma_position == None or first_comma_position < k.position) 
                        and (sec_comma_position == None or sec_comma_position > k.position)]
    
            # mark token in participle construction
            for token in part_con:
                token.part = True
                   
        # Reduce weights for tokes in prepositional phrases, relative and 
        # subjunctive clauses and participle constructions
        if config.reduce_weights:
                            
            for p in prep_phrase:                
                if p.tag!= '':                    
                    p.reduce_tag()
                
            for t in tokens:
                if t.rel and t.tag != '':
                    t.reduce_tag()          
                if t.part and t.tag != '':
                    t.reduce_tag()                             
                if t.subj and t.tag != '':
                    t.reduce_tag()
            

        # list of all entities
        all_entities  = subjs_lemma + objs_lemma + others 
        
        if not config.merge_p_and_g:
            # append cat p and g entities
            if config.cat_p_on:
                all_entities = all_entities + poss_pronouns
            if config.cat_g_on:
                all_entities = all_entities + genitive_mods
            
        entities_with_roles.append(all_entities)
        

    return entities_with_roles
    
    
def get_entity_collection_and_counter(entities_with_roles):
    """Collect entity and role information per sentence. 
    Resolve coreference if necessary.
    Count entities occurences over sentences."""
    
    entity_counter = defaultdict(int)
    coref_dict = {}
    entity_collection = []
    
    # iterate through sentences and entities

    for sent in entities_with_roles:
        entity_set = []    
        entity_info = []
        
        for word in sent:
            
            # get entity and syntactic role  
            entity = word
            synt_role = entity.tag
            reduct_deg = entity.red_degree
          
            # if coreferent entity
            if entity.coref != '_':
                
                # if unseen noun, add lemma to coref_dict
                # save entity information
                if entity.coref not in coref_dict.keys() and entity.sim_pos == 'N':
                    
                    coref_dict[entity.coref] = entity.lemma
                    entity_set.append(entity.lemma)                
                    entity_info.append((entity.lemma,synt_role,reduct_deg))
                    
                # if seen and coreference feature is on, get corresponding lemma from coref_dict 
                # save entity information
                if entity.coref in coref_dict.keys() and config.coref_on:
                    entity_name = coref_dict[entity.coref]                    
                    entity_info.append((entity_name,synt_role, reduct_deg))
                    entity_set.append(entity_name)
             
            # if no coreferent entity but noun, save entity information
            else: 
                if entity.sim_pos == 'N':
                        entity_set.append(entity.lemma)                  
                        entity_info.append((entity.lemma,synt_role, reduct_deg))
    
        # count entity occurences over sentences
        for e in set(entity_set):
            entity_counter[e] += 1
        
        # save set of entities per sentence (no multiples with same synt role)    
        sorted_entity_info = sorted(list(set(entity_info)), key=itemgetter(0))
        entity_collection.append(sorted_entity_info)


    return entity_collection, entity_counter


def get_weight(role, degree):
    """Compute weight based on category and reduction degree."""
    
    if config.synt_roles_on:
        weight = config.synt_role_dict[role]
                   
        if config.reduce_weights:

            weight = weight*(config.REDUCTION**degree)
              
        return weight
    
    return 0

def get_entity_grid(clean_entities_by_sent, entity_counter):
    """Return the entity grid and entity list.
    Filter duplicates in sentences, assign highest level syntactic role 
    and filter out entities that occur only in one sentence."""
    
    # Initialize result lists
    entity_grid = []
    entity_list = []

    # Iterate over sentences
    for sent in clean_entities_by_sent:
    
        # collection of all entities in sent
        all_ents_in_sent = [e[0] for e in sent]
        
        # controle which entities were already processed to avoid double processing and counting
        proc_entities = []
        entity_infos = []
        for entity in sent:
            if entity[0] in proc_entities:
                continue
            proc_entities.append(entity[0])
        
            # if entity occurs more than once in sentence (with different syntactic roles)    
            # select the "highest level" syntactic role: S > 0 > (X|P)
            if all_ents_in_sent.count(entity[0])>1:            
                
                multiples = [e for e in sent if e[0] == entity[0]]                
             
                ents_with_weights = [(e, e[1], get_weight(e[1],e[2])) for e in multiples]
                ents_with_weights = sorted(ents_with_weights,key=itemgetter(2), reverse=True)
                
                highest_role = ents_with_weights[0]
                entity = highest_role[0]
                    
#            # only consider entity if it occurs in more than one sentence
#            if entity_counter[entity[0]]>1:       
#                entity_infos.append(entity)
#                entity_list.append(entity[0])         

            # Append entity
            entity_infos.append(entity)
            entity_list.append(entity[0])

        entity_grid.append(entity_infos)
        
    # list with all entities (first row of entity grid)
    entity_list = sorted(set(entity_list), key=entity_list.index)
       
    return entity_grid, entity_list
      

def write_to_outfile(entity_grid, entity_list, entity_counter):
    """Write entity grid to csv file."""

    # create outfile
    with open(config.ENTITY_GRID_FILE,'w') as outfile:
        sent_counter = 0
        
        # write first row (with entities)
        for word in entity_list:
            
            # do not write singles into Entity Grid
            if config.reduced_entity_grid and entity_counter[word] <= 1:
                continue
            
            outfile.write(' ,'+word)
        outfile.write('\n')
            
        # write sentence number
        for sent in entity_grid:
            sent_counter += 1
            outfile.write(str(sent_counter))
            
            all_ent_in_sent = [e[0] for e in sent]
            
            # if entity is not in sentence, write _
            # otherwise write syntactic role
            
            for entity in entity_list:
                
                # do not write singles in Entity Grid
                if config.reduced_entity_grid and entity_counter[entity] <= 1:
                    continue
                
                if entity not in all_ent_in_sent:
                    outfile.write(',_')
                else:
                    role = next(e[1] for e in sent if e[0] == entity)
                    outfile.write(','+role)
                    
                    if config.reduce_weights:
                        degree = next(e[2] for e in sent if e[0] == entity)            
                        if degree>0:
                            outfile.write('-'+str(degree))
                        
            outfile.write('\n')
     

def get_grid_array(entity_grid, entity_list):
    """Create array from entity grid with integer values for syntactic roles."""
   
    entity_array = []
    
    # for each sentence
    for sent in entity_grid:        
        array_row = []       
        all_ent_in_sent = [e[0] for e in sent]
        
        # if entity is not in sentence, take 0 as syntactic role value
        for entity in entity_list:
            if entity not in all_ent_in_sent:
                array_row.append(0)
            
            # if semantic roles are considered, select corresponding integer (1,2 or 3)
            # otherwise use 1 
            else:
                if config.synt_roles_on:
                    role = next(e[1] for e in sent if e[0] == entity)
                    weight = config.synt_role_dict[role]
                   
                    if config.reduce_weights:
                        degree = next(e[2] for e in sent if e[0] == entity)
                        weight = weight*(config.REDUCTION**degree)
                 
                    array_row.append(weight)
                  
                else:
                    array_row.append(1)
                     
        entity_array.append(array_row)
        
    # save data as numpy array
    grid_array = np.zeros((len(entity_grid),len(entity_list)))
    for i in range(0,len(entity_array)):
        grid_array[i:,] = entity_array[i]

    return grid_array