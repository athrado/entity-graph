"""
Entity Graph for German
Author: Julia Suter, 2018/19
-----------------------------

parse_information.py

    - Classes and methods for easy processing of CoNNL formatted output
    - Output given by ParZu and CorZu
    - Classes for Sentence and Token
    - Compute and store syntactic information from parsed text

"""

# Import Statement
import config


# --- Classes for Sentence and Token --- 

class Sentence(object):
    
    """Sentence Class for sentences parsed by ParZu.
    Offers many methods for returning syntatical functions of Sentence like subject, predicate or prepositional phrase
    """
    
    def __init__(self,data):
            self.data = data
    
    def subj(self):
        """Return (first) subject head of Sentence."""
        try:
            subj = [k for k in self.data if k.function == 'subj']
            return subj
        except StopIteration:
            return None 
            
    def predicate(self):
        """Return (first) predicate of Sentence."""
        try:
            # Predicate is finite verb
            pred = [k for k in self.data if k.full_pos.startswith('V') and k.full_pos.endswith('FIN')]
            return pred    
        except StopIteration:
            return None 

    def agens(self, participle):
        """Return agens of Sentence."""
        pp_heads = [k for k in self.data if k.function == 'pn']
        for pn in pp_heads:
            for k in self.data:
                if (k.position == pn.dependency and (k.lemma == 'von') and (k.function == 'pp') and (k.dependency == participle.position)):
                    return pn

    def pass_verb(self,verb_pos):
        """Return passive verb (action verb) Sentence."""      
        try:
            # The passive verb (action verb) is the past participle with function "aux" (or "cj")
            verb = next(k for k in self.data if ((k.function == 'aux' or k.function == 'cj') and (k.full_pos == 'VVPP') and (k.dependency == verb_pos)))
            return verb
        except StopIteration:
            return None
            
    def obj(self):
        """Return direct object's head of Sentence."""
        try:
            obj = [k for k in self.data if (k.function == 'obja' or k.function == 'objd')]
            
            if config.all_obj_on:
                obj = [k for k in self.data if (k.function == 'objd' or k.function == 'obja' or k.function == 'objp' or k.function == 'objg')]
            return obj
        except StopIteration:
            return None   
            
    def obj_d(self):
        """Return direct object's head of Sentence."""
        try:
            obj = [k for k in self.data if (k.function == 'objd')]
            return obj
        except StopIteration:
            return None   
            
    def genitive(self):
        """Return all genitive attributes of Sentence."""
        genitives = [k for k in self.data if (k.function() == 'gmod')]
        return genitives
        
    def prep_phrase(self):
        """Return all prepositiona phrases of Sentence."""
        prepositional_phrases = [k for k in self.data if k.function == 'pp']
        return prepositional_phrases
        

class Token(object):
    
    """Token Class for representing Tokens in sentences parsed by ParZu.
    Offers many functions for returning linguistic information on token
    and changing it.
    """
    def __init__(self,token_data):    

            self.data = token_data
            self.position = int(token_data[0])
            self.word = token_data[1]
            self.lemma = token_data[2]
            self.sim_pos_full = token_data[3]
            self.sim_pos = token_data[3][0]
            self.full_pos = token_data[4]
            if self.full_pos == '_':
                self.full_pos = token_data[3]
            
            self.mo = token_data[5]
            
            if self.full_pos in ['ADJD', 'ADJA']:
                self.morph = Adj_Morphology(token_data[5])            
            
            self.dependency = int(token_data[6])
            self.function = token_data[7].lower()
            self.coref = token_data[9]
            
            self.tag = ''
            self.rel = False
            self.subj = False
            self.part = False
            self.red_degree = 0
            

    def reduce_tag(self):
        """Increase the reduction degree and thus reduce the weight."""
        
        self.red_degree += 1
        
    def morph(self):
        """Return morphological information of Token in sentence.
        Depending on part-of-speech, morphology is safed in different class"""        
                
        if self.sim_pos == 'V':
            return Verb_Morphology(self.mo)
     
    def change_word(self,new_word):
        """Change word of Token to new_word."""
        self.word = new_word
        
        
class Noun_Morphology(object):
    """Class for morphology of nouns."""
    
    def __init__(self, morphdata):        
                
        self.morphdata = morphdata    
        self.genus = '_' if self.morphdata == '_' else self.morphdata.split('|')[0]
        self.casus = '_' if self.morphdata == '_' else self.morphdata.split('|')[1]
        self.numerus =  '_' if self.morphdata == '_' else self.morphdata.split('|')[2]
        
         
class Adj_Morphology(object):
    """Class for morphology of adjectives."""
    
    def __init__(self, morphdata):
        
        self.morphdata = morphdata    
        self.comp = '_' if self.morphdata == '_' else self.morphdata.split('|')[0]
        self.part = '_' if (self.morphdata == '_' or len(self.morphdata.split('|'))<=1)  else self.morphdata.split('|')[1]   
 
    
class Verb_Morphology(object):    
    """Class for Morphology information for Verbs as given by ParZu.
    Offers many methods for returning morphological information"""


    def __init__(self,morphdata):
        self.morphdata = morphdata 
     
    def person(self):
        """Return person."""
        return int(self.morphdata.split('|')[0])
        
    def numerus(self):
        """Return number."""
        return self.morphdata.split('|')[1]
    
    def temp(self):
        """Return tense."""
        return self.morphdata.split('|')[2]
    
    def mod(self):
        "Return mode."""
        if self.morphdata == '_':
            return '_'
        else:
            return self.morphdata.split('|')[3]

def adjust_passive(tokens):
    """Change subject and agent to object and logical subject if there is passive construction."""
    
    passive_found = False
    
    # transform tokens to Sentence class instance 
    sent = Sentence(tokens)
    
    # get predicates
    preds = sent.predicate()
    
    # get preds that involve passive
    preds = [t for t in preds if (t.lemma.startswith('werden') and t.morph().mod().lower() != 'subj' and (t.function == 'root' or t.function == 'neb'))]

    for pred in preds:
        # get particple (action verb)
        participle =  sent.pass_verb(pred.position)
        # get subj (logical object)
        try:
            subj = next(t for t in sent.subj() if (t.dependency == pred.position))
        except AttributeError:
            continue
        except StopIteration:
            continue
        # get agent (logical subject)
        if participle == None:
            continue
        agens = sent.agens(participle)
        if agens == None:
            continue
        
        # change functions
        subj.function = 'obja'
        agens.function = 'subj'
        passive_found = True
 
    return passive_found
    
def toString(t):
    """Put all token information into one string."""
    
    string = str(t.position)+'\t'+t.word+'\t'+t.lemma+'\t'+t.sim_pos+'\t'+t.full_pos+'\t'+t.mo+'\t'+str(t.dependency)+'\t'+t.function+'\t'+t.coref+'\t'+t.coref
    return string
     
def sentenceToString(tokens):
    """Put the token information for all tokens into one string."""
    
    sentence_string = []    
    for t in tokens:
        string = toString(t)
        sentence_string.append(string)
    
    return "\n".join(sentence_string)
            

def coref_cleaning(t, new_coref_np, current_coref):
        """Handle coreference tags provided by CorZu. 
        Remove brackets and handle relative clauses."""
    
        if t.coref.startswith('(') and t.coref.endswith(')'):
            current_coref = t.coref[1:-1]   
            t.coref = current_coref
            new_coref_np = False            
       
        # nested coref       
        elif t.coref.startswith('('):
            current_coref = t.coref[1:]  
            t.coref = current_coref

            new_coref_np = True
            if t.full_pos == 'PRELS':
                new_coref_np = False
           
        # if value is - , get last found integer
        elif new_coref_np and t.coref.startswith('-'):
            t.coref = current_coref
            
            if t.full_pos == 'PRELS':
                new_coref_np = False
           
        # end of nested coref 
        elif new_coref_np == True and t.coref.endswith(')'):

            current_coref = t.coref[:-1]  
            if t.sim_pos=='V':
                current_coref = '_'
            t.coref = current_coref
            new_coref_np = False
           
        # no coref entity
        elif new_coref_np == False:
            t.coref = '_'
            
            
        return (new_coref_np, current_coref)
        
def adjust_coref(tokens):
    
    """Assign a single integer coreference tag to tokens based on CorZu tag.
    For coreference tags with brackets, apply tag to all tokens within brackets.
    Handle multiple coref tags cases. """
    
    new_coref_np = False   
    current_coref = None
    
    # change coref value to integer between brackets     
    for t in tokens:

        if '|' in t.coref:            
            first_part = t.coref.split('|')[0]
            second_part = t.coref.split('|')[1]
            
            t.coref = first_part
            
            new_coref_np, current_coref = coref_cleaning(t,new_coref_np, current_coref)
        
            if second_part.startswith('('):    
                new_coref_np = True
                current_coref = second_part[1:]
                
        else:
            new_coref_np, current_coref = coref_cleaning(t,new_coref_np, current_coref)
        

def get_all_tokens(phrase, tokens):
    """Get first and last token of phrase and return all tokens in between.
    Useful to catch full relative and subjunctive clauses. """
    
    from operator import itemgetter
    
    positions = sorted([(t, t.position) for t in phrase], key=itemgetter(1))

    first_tok = positions[0][1]
    last_tok = positions[-1][1]
    
    new_phrase = [t for t in tokens if t.position >= first_tok and t.position <= last_tok]
    
    return new_phrase
    

def get_dependent_tokens(sent,head,already_processed = []):
    """ Return all words in sentence that are dependent on head.
    Args:       sent (list), head (Token)
    Returns:    dependent_tokens (list of Tokens)
    """
    
    for k in sent:
            dependent_tokens = [k for k in sent if (k.dependency == head.position) and k not in already_processed] 
    if len(dependent_tokens)<1:
        return []        
    #   for all dependent tokens, get their dependent tokens (recursion)
    else:
        for k in dependent_tokens:             
                new_dep_tokens = get_dependent_tokens(sent,k, dependent_tokens)
                dependent_tokens += new_dep_tokens
 
        return dependent_tokens
        
           
def get_full_phrase(sent,head):

    """ Return the full phrase for a given head (usually for noun phrases).
    Args:       sent (list), head (Token)
    Returns:    phrase (list of Tokens)
    """

    phrase = get_dependent_tokens(sent,head)
    phrase.append(head)
    phrase = sorted(set(phrase), key=lambda x: x.position)
    return phrase


def get_next_token(tokens, head):
    """ Returns next token
    Args:       tokens (list), head (Token)
    Returns:    next_t (Token)
    """
    
    try:
        next_t = next(t for t in tokens if t.position == head.position+1)
        return next_t
    except StopIteration:
        return None
    

def print_current_sent(sent):
    
    """ Print sentence at current processing step.
    Only used for implementing and debugging.
    Args: sent (list)
    """

    for a in sent:
        print(a.word())        
    
