"""
Entity Graph for German
Author: Julia Suter, 2018/19
-----------------------------

GET_entgraph_features.py

    - Set which files to process (including files to skip)
    - Compute entity graph for sample
    - Compute entity graph metrics
    - Save the entity graph features

"""
# ------------------------------------------------
# Import Statenents
# ------------------------------------------------

import os
import shutil
import pickle
import re

import config

import numpy as np
from scipy.stats import skew, kurtosis
from scipy.linalg import norm

import graph_tool.all as gt
import networkx as nx

import timeout_decorator

# ------------------------------------------------
# Graph Creation and Metric Computation
# ------------------------------------------------

class Graph(object):
    """Graph class which computes a number of metrics."""
        
    # Get graph metrics and measures
    def __init__(self, g, author, filename, entity_graph):  
                
        # Basics
        self.author_ = author
        self.edge_weights_ = g.edge_properties["weights"] 
        self.pos_ = g.vertex_properties["pos"]
        self.v_count_ = g.vertex_properties["v_count"]
        self.gt_graph_ = g
        self.filename_ = re.sub('_entgraph','',filename)
               
        # Number of edges and verties, density
        self.num_edges_ = g.num_edges()
        self.num_edges  = g.num_edges()
        self.num_nodes_ = g.num_vertices()
        
        self.num_poss_edges_ = (self.num_nodes_*(self.num_nodes_-1))/2
        self.density_ = self.num_edges_/self.num_poss_edges_        
        self.density_norm = self.density_/self.num_edges_

        # Degree
        self.vertex_avg_, self.vertex_avg_var = gt.graph_tool.stats.vertex_average(g, "total")
        self.vertex_avg_in_, self.vertex_avg_in_var_ = gt.graph_tool.stats.vertex_average(g, "in")
        self.vertex_avg_out_, self.vertex_avg_out_var_ = gt.graph_tool.stats.vertex_average(g, "out")
        self.edge_avg, self.edge_avg_var = gt.graph_tool.stats.edge_average(g, eprop=self.edge_weights_)
        
        self.vertex_avg_norm = self.vertex_avg_/self.num_edges_
        self.edge_avg_norm = self.edge_avg/self.num_edges_

        # Vertex and edge histograms
        self.vertex_hist_ = gt.graph_tool.stats.vertex_hist(g, deg='total', )
        self.vertex_hist_in_ = gt.graph_tool.stats.vertex_hist(g, deg='in', bins=range(0,self.num_nodes_))
        self.vertex_hist_out_ = gt.graph_tool.stats.vertex_hist(g, deg='out', bins=range(0,self.num_nodes_))
        self.edge_hist_ = gt.graph_tool.stats.edge_hist(g,eprop=self.edge_weights_, bins=np.arange(0.0,1.0,0.01))
        
        self.degrees_ = get_values_from_histo(self.vertex_hist_)
        self.degrees_mean, self.degrees_var, self.degrees_skew, self.degrees_kurtosis = get_moments(self.degrees_)

        self.degrees_in_ = get_values_from_histo(self.vertex_hist_in_)
        self.degrees_in_mean_, self.degrees_in_var, self.degrees_in_skew, self.degrees_in_kurtosis = get_moments(self.degrees_in_)

        self.degrees_out_ = get_values_from_histo(self.vertex_hist_out_)
        self.degrees_out_mean_, self.degrees_out_var, self.degrees_out_skew, self.degrees_out_kurtosis = get_moments(self.degrees_out_)

        self.weights_ = get_values_from_histo(self.edge_hist_)
        self.weights_mean, self.weights_var, self.weights_skew, self.weights_kurtosis = get_moments(self.weights_)
        
        self.degrees_mean_norm = self.degrees_mean/self.num_edges_
        self.weights_mean_norm = self.weights_mean/self.num_edges_
        
        self.edge_weights_mean_, self.edge_weights_var, self.edge_weights_skew, self.edge_weights_kurtosis = get_moments(self.edge_weights_.a)
        self.edge_weights_mean_norm = self.edge_weights_mean_/self.num_edges_                    
            
        # Distance metrices
        self.dist_histogram_ = gt.graph_tool.stats.distance_histogram(g, bins = range(0,10))
        self.avg_shortest_path = np.mean(get_values_from_histo(self.dist_histogram_))

        self.diameter = np.max(get_values_from_histo(self.dist_histogram_))
        self.pseudo_diameter_ = gt.pseudo_diameter(g)[0] 
        
        self.diameter_norm = self.diameter/self.num_edges_
        self.avg_shortest_path_norm = self.avg_shortest_path/self.num_edges_
        
        # Centrality measures
        self.max_eigen_, self.eigenvectors_ = gt.eigenvector(g, weight=self.edge_weights_)
        self.eigenvectors_ = self.eigenvectors_.a
        self.katz_ = gt.graph_tool.centrality.katz(g, weight=self.edge_weights_).a
        self.pageranks_ = gt.graph_tool.centrality.pagerank(g, weight=self.edge_weights_).a
        
        self.eigenvectors_mean, self.eigenvectors_var, self.eigenvectors_skew, self.eigenvectors_kurtosis = get_moments(self.eigenvectors_)
        self.katz_mean, self.katz_var, self.katz_skew, self.katz_kurtosis = get_moments(self.katz_)
        self.pageranks_mean, self.pageranks_var, self.pageranks_skew, self.pageranks_kurtosis = get_moments(self.pageranks_)

        self.eigenvectors_mean_norm = self.eigenvectors_mean/self.num_edges_
        self.katz_mean_norm = self.katz_mean/self.num_edges_
        self.pageranks_mean_norm = self.pageranks_mean/self.num_edges_
        
        # HITS: authority centrality, hub centrality
        self.hits_eig, self.auth_centr_, self.hub_centr_ = gt.graph_tool.centrality.hits(g, weight=self.edge_weights_)
        self.hits_eig = self.hits_eig
        self.auth_centr_ = self.auth_centr_.a
        self.hub_centr_ = self.hub_centr_.a    

        self.auth_centr_mean, self.auth_centr_var, self.auth_centr_skew, self.auth_centr_kurtosis = get_moments(self.auth_centr_)
        self.hub_centr_mean, self.hub_centr_var, self.hub_centr_skew, self.hub_centr_kurtosis = get_moments(self.hub_centr_)

        self.hits_eig_norm = self.hits_eig/self.num_edges_
        self.auth_centr_mean_norm = self.auth_centr_mean/self.num_edges_
        self.hub_centr_mean_norm = self.hub_centr_mean/self.num_edges_

        # Closeness and betweenness
        self.closeness_ = gt.graph_tool.centrality.closeness(g, weight=self.edge_weights_)
        self.closeness_ = self.closeness_.a

        self.vertex_betweenness_ , self.edge_betweenness_ = gt.graph_tool.centrality.betweenness(g, weight=self.edge_weights_)
        self.vertex_betweenness_ = self.vertex_betweenness_.a
        self.edge_betweenness_ = self.edge_betweenness_.a

        self.closeness_mean_, self.closeness_var_, self.closeness_skew_, self.closeness_kurtosis_ = get_moments(self.closeness_)
        self.vertex_betweenness_mean, self.vertex_betweenness_var, self.vertex_betweenness_skew, self.vertex_betweenness_kurtosis = get_moments(self.vertex_betweenness_)
        self.edge_betweenness_mean, self.edge_betweenness_var, self.edge_betweenness_skew, self.edge_betweenness_kurtosis = get_moments(self.edge_betweenness_)
        
        self.vertex_betweenness_mean_norm = self.vertex_betweenness_mean/self.num_edges_
        self.edge_betweenness_mean_norm = self.edge_betweenness_mean/self.num_edges_            
            
        # Reciprocity
        self.edge_reciprocity_ = gt.graph_tool.topology.edge_reciprocity(g)
        self.edge_reciprocity_norm = self.edge_reciprocity_/self.num_edges_

        # Components
        self.largest_component = gt.graph_tool.topology.label_largest_component(g, directed=False).a
        self.fraction_largest_component_ =  np.sum(self.largest_component)/self.largest_component.shape[0]
        self.largest_component = np.sum(self.largest_component)
        
        self.largest_component_norm = self.largest_component/self.num_edges_
        
        # Booleans
        self.is_bipartite_ = gt.graph_tool.topology.is_bipartite(g)
        self.is_DAG_ = gt.graph_tool.topology.is_DAG(g)
        #self.is_planar = gt.graph_tool.topology.is_planar(g)
        
        # Clustering 
        self.local_clustering_coefficient_ = gt.graph_tool.clustering.local_clustering(g).a
        self.global_clustering_coefficient, self.global_clustering_coefficient_var = gt.graph_tool.clustering.global_clustering(g)
        self.local_clustering_coefficient_mean, self.local_clustering_coefficient_var_, self.local_clustering_coefficient_skew, self.local_clustering_coefficient_kurtosis = get_moments(self.local_clustering_coefficient_)

        self.k_core_ = gt.graph_tool.topology.kcore_decomposition(g).a
        self.k_core_mean = np.mean(self.k_core_)
        self.k_core_mean_norm = self.k_core_mean/self.num_edges_
        
        self.local_clustering_coefficient_mean_norm = self.local_clustering_coefficient_mean/self.num_edges_
        self.global_clustering_coefficient_norm = self.global_clustering_coefficient/self.num_edges_

        # Assortivity
        self.assortivity, self.assortivity_var = gt.graph_tool.correlations.assortativity(g, deg="total")
        self.scalar_assortivity, self.scalar_assortivity_var = gt.graph_tool.correlations.scalar_assortativity(g, deg="total")

        self.assortivity_norm = self.assortivity/self.num_edges_
        self.scalar_assortivity_norm = self.scalar_assortivity/self.num_edges_
        
        ## MAX FLOW
        
        # The capacity will be defined as the inverse euclidean distance
        cap = g.new_edge_property("double")
        pos = self.pos_
        edges = list(g.edges())
        for e in edges:
            cap[e] = min(1.0 / norm(pos[e.target()].a - pos[e.source()].a), 10)
        g.edge_properties["cap"] = cap

        cap = g.edge_properties["cap"]
        cap = self.edge_weights_
        
        # Max flow 
        src, tgt = g.vertex(0), g.vertex(self.num_nodes_-1)
        res = gt.graph_tool.flow.edmonds_karp_max_flow(g, src, tgt, cap)
        res.a = cap.a - res.a  # the actual flow
        self.max_flow = sum(res[e] for e in tgt.in_edges())
        
        self.min_st_cut_partition = np.sum(gt.graph_tool.flow.min_st_cut(g, src, cap, res).a)
        self.min_st_cut_partition_norm = self.min_st_cut_partition/self.num_edges_
        self.max_flow_norm = self.max_flow/self.num_edges_
        
        # First vertex features        
        self.fv_degree_ = self.degrees_[0]
        self.fv_eigenvector_ = self.eigenvectors_[0]
        self.fv_katz_ = self.katz_[0]
        self.fv_pagerank_ = self.pageranks_[0]
        self.fv_auth_centr_ = self.auth_centr_[0]
        self.fv_hub_centr_ = self.hub_centr_[0]
        self.fv_closeness_ = self.closeness_[0]
        self.fv_betweenness_ = self.vertex_betweenness_[0]
        self.fv_local_clustering_coeff_ = self.local_clustering_coefficient_[0]
        
        # Min cut       
        g.set_directed(False)        
        self.min_cut, self.partition = gt.graph_tool.flow.min_cut(g, weight=self.edge_weights_)
        self.partition = np.sum(self.partition.a)
        
        self.min_cut_norm = self.min_cut/self.num_edges_
        self.partition_norm = self.partition/self.num_edges_
        
        self.ent_graph_ = entity_graph
              

def get_values_from_histo(histogram):
    """Get values from graph-tool histogram."""

    value_array = np.concatenate([np.asarray(np.array(histogram[0][i])*[histogram[1][i]])
                    for i in range(histogram[0].shape[0])
                    if histogram[0][i] != 0])
    
    return value_array

def get_moments(array):
    """Return moments of array (mean, var, skewness, kurtosis)."""

    # Compute moments    
    mean = np.mean(array)
    var = np.var(array)
    skewness = skew(array)
    kurtosis_ = kurtosis(array)

    # Return
    return mean, var, skewness, kurtosis_


@timeout_decorator.timeout(2)  # Time out after 2 seconds (for each graph)
def create_graph(filename, author):
    """Get entity graph data from file and create graph with networkx."""
    
    # Get filename (of array) and load
    with open(DIRECTORY+filename,'rb') as f:       
        entity_graph = pickle.load(f, encoding='latin1')

    # Compute entity graph sum to verify graph is not empty.
    eg_sum = np.sum(entity_graph)

    # Discard empty graphs
    if eg_sum==0.0:
        return None
        
    ## Get graphs (add empty vertex)
    entity_graph_zero = np.zeros((entity_graph.shape[0]+1, entity_graph.shape[1]+1))
    entity_graph_zero[:-1,:-1] = entity_graph 
    #adj_matrix = entity_graph_zero
    
    # Get adjacency matrix
    adj_matrix = entity_graph
         
    # Get circular graph pos
    D = nx.from_numpy_matrix(entity_graph_zero)
    pos_x = nx.shell_layout(D)
    
    # Create graph, set nodes, edge type and pos
    g = gt.Graph(directed = True)
    g.add_vertex(adj_matrix.shape[0])
    edge_weights = g.new_edge_property("float")
    pos = g.new_vertex_property("vector<float>")
    v_count = g.new_vertex_property("int")
    
    # Assign pos by network x
    n_vertices = adj_matrix.shape[0]

    for i in range(adj_matrix.shape[0]):
        vert = n_vertices-i
        pos[g.vertex(i)] = pos_x[vert]
        v_count[g.vertex(i)] = i+1
        
    # Fill graph with weights
    for i in range(n_vertices):
        for j in range(n_vertices):            
                if adj_matrix[i,j] != 0.0:
                    e = g.add_edge(i,j)
                    edge_weights[e] = adj_matrix[i,j]
        
    # Prop sizing
    v_size_p = gt.prop_to_size(v_count, 5,10)
    edge_weights_p = gt.prop_to_size(edge_weights, mi=0, ma=10, power=1)

    # Save edge/vertex properties
    g.edge_properties["weights"] = edge_weights
    g.vertex_properties["pos"] = pos
    g.vertex_properties["v_count"] = v_count
    
    # Print graph
    #gt.graph_draw(g, pos=pos,vertex_text=v_count, vertex_fill_color='orange', edge_pen_width=edge_weights_p, output_size=(600, 600))
 
    # Generate Graph
    try:
        new_graph = Graph(g, author, filename, entity_graph)
    except ValueError:
        print("This file didn't work:", filename)
        return
    return new_graph


# ------------------------------------------------
# Settings
# ------------------------------------------------

# Load version (pu, pw, pacc)
version = config.version

# Source directory
DIRECTORY = '../1_Processed_texts/graphs_per_text/GUTENBERG_'+version+'/'

# Target directory
TARGET_DIR = '../2_Features/EG_features_'+version+'/'

# Delete and create target dir
if os.path.exists(TARGET_DIR):
    shutil.rmtree(TARGET_DIR)
os.makedirs(TARGET_DIR)

# Set minimum number of samples to use per category
MIN_NR_SAMPLES = 450

# ------------------------------------------------
# Set which samples or authors to process
# ------------------------------------------------

# Get all array files
allfiles = os.listdir(DIRECTORY)
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

# Used for language level version (not implemented here)
a1_text_files = [f for f in array_files if f.startswith('A1')]
a2_text_files = [f for f in array_files if f.startswith('A2')]
b1_text_files = [f for f in array_files if f.startswith('B1')]
b2_text_files = [f for f in array_files if f.startswith('B2')]


# Put all arrays together
all_array_files = [array_kafka_files, array_kleist_files, array_schnitzler_files, 
                   array_zweig_files, array_hoffmann_files, array_twain_files,
                   
                   array_tieck_files, array_gotthelf_files, array_eichendorff_files, 
                   array_keller_files, array_spyri_files,
                   
                   array_bierbaum_files, array_busch_files, array_dauthendey_files, 
                   array_fontane_files, array_ganghofer_files,
                   
                   array_gerstaecker_files, array_grimm_files, 
                   array_haltrich_files, array_hebbel_files,
                   
                   array_hofmannsthal_files, array_jeanpaul_files, array_may_files, 
                   array_novalis_files, 
                   
                   array_poe_files, array_raabe_files, array_scheerbart_files, 
                   array_schwab_files, array_stifter_files,
                   
                   array_storm_files, array_thoma_files, array_volkmann_files]

# Set author names
all_author_names = ['Kafka','Kleist','Schnitzler','Zweig','Hoffmann', 'Twain', 
                'Tieck', 'Gotthelf', 'Eichendorff','Keller','Spyri', 
                'Bierbaum','Busch','Dauthendey','Fontane','Ganghofer', 
                'Gerstaecker','Grimm','Haltrich','Hebbel', 
                'Hofmansthal','JeanPaul','May','Novalis',
                'Poe','Raabe','Scheerbart','Schwab','Stifter',
                'Storm','Thoma','Volkmann'] 


# Make sure nr of authors is right
assert (len(all_array_files) == len(all_author_names))

# # Print samples per author
# for i, author in enumerate(all_array_files):
#     print(all_author_names[i], len(author))
    
# Get files per author
graph_files_by_author = [[(f, all_author_names[i]) for f in author_files] for i, author_files in enumerate(all_array_files)]
graph_files_by_author = np.concatenate(graph_files_by_author)

# ------------------------------------------------
# Skip files: which files to avoid (because of length, corruption)
# ------------------------------------------------

# Trouble files that need to be skipped

skip_files = ['HEB_genoveva_18_entgraph.npy', 'HEB_muttkind_18_entgraph.npy',
              'HEB_kuh_4_entgraph.npy','HEB_schnock_5_entgraph.npy',
              'HEB_muttkind_25_entgraph.npy','HEB_genoveva_6_entgraph.npy',
              'HEB_muttkind_21_entgraph.npy', 'HEB_1nacht_8_entgraph.npy',
              'HEB_genoveva_22_entgraph.npy','SCHN_geronimo_5_entgraph.npy',
              'SCHN_casaheim_50_entgraph.npy', '901 SCHN_geronimo_20_entgraph.npy',
              'SCHN_reichtum_48_entgraph.npy', 'SCHN_rufleben_15_entgraph.npy',
              'SCHN_reigen_18_entgraph.npy', 'SCHN_gustl_25_entgraph.npy',
              'SCHN_zwspiel_27_entgraph.npy', 'SCHN_traumnov_53_entgraph.npy',
              'SCHN_reichtum_2_entgraph.npy', 'SCHN_casaheim_13_entgraph.npy',
              'SCHN_einsweg_94_entgraph.npy', 'SCHN_zwspiel_63_entgraph.npy',
              'SCHN_traumnov_72_entgraph.npy', 'SCHN_zwspiel_87_entgraph.npy',
              'SCHN_geronimo_22_entgraph.npy', 'SCHN_einsweg_133_entgraph.npy',
              'SCHN_wegfreie_11_entgraph.npy', 'SCHN_einsweg_155_entgraph.npy',
              'HEB_1nacht_3_entgraph.npy', 'HEB_magdalen_1_entgraph.npy',
              'HEB_genoveva_99_entgraph.npy', 'HEB_muttkind_10_entgraph.npy',
              'HEB_magdalen_83_entgraph.npy', 'HEB_genoveva_105_entgraph.npy',
              'HEB_genoveva_91_entgraph.npy', '2031 KEL_kleider_10_entgraph.npy',
              'THO_nachbar_43_entgraph.npy',
              'SCHE_tarub_32_entgraph.npy']

# More trouble files that need to be skipped
if version.endswith('pu'):
    more_skip_files = [
                  'THO_jagrlois_26_entgraph.npy','THO_ruepp_24_entgraph.npy',
                  'THO_mnchnrin_6_entgraph.npy','THO_moral_5_entgraph.npy',
                  'THO_muenchnr_14_entgraph.npy','THO_hies_2_entgraph.npy',
                  'THO_nachbar_37_entgraph.npy','THO_jagrlois_38_entgraph.npy',
                  'THO_mnchnrin_33_entgraph.npy','THO_jagrlois_27_entgraph.npy'
                  'THO_jagrlois_8_entgraph.npy','THO_jagrlois_27_entgraph.npy',
                  'THO_altaich_34_entgraph.npy']

    # Add skip files
    skip_files += more_skip_files

# ------------------------------------------------
# Compute and save graph metrics for each sample
# ------------------------------------------------

# Initialize 
all_graphs = []
author_names = []
file_names = []

author_list = []
file_list = []

long_files = []
empty_files = []

print('\nStarting...\n')

# For file in selected 2000 sampples
for k,(file, author) in enumerate(graph_files_by_author):

        # Print
        print(k, file)

        # Skip certain files
        if file in skip_files:
            continue
    
        try:
            # Create the graph and its metrics
            new_graph = create_graph(file, author)

        # Skip files that take too long to process (should not happen!)
        except RuntimeError:
            print('Failed, took too long: ', file)
            long_files.append(file)
            continue

        # Skip empty files (should not happen!)
        if new_graph==None:
            empty_files.append(file)
            continue
                
        # Save author and filename
        all_graphs.append(new_graph)
        author_names.append(author)

        # Save entity graph and filename
        filename = re.sub('_entgraph','',file)
        file_names.append(filename)


# Check correct number of graphs and solutions
assert len(author_names) == len(all_graphs)

# ------------------------------------------------
# Set category names, feature names and feature dict
# ------------------------------------------------

# Authors/genres that were used
used_author_names = list(set(author_names))

# Get feature dict from example graph
exp_graph = all_graphs[0]
feature_names = [attr for attr in dir(exp_graph) if not callable(getattr(exp_graph, attr)) and not attr.startswith("__")]
feature_dict = vars(exp_graph)

# Get feature names 
features_of_interest = [k for k in feature_dict.keys() if (not k.endswith('_') 
                                                           and not k.endswith('_norm')
                                                           and not k.endswith('_skew')
                                                           and not k.endswith('_kurtosis')
                                                           and not k.endswith('_var'))]

# Create sample by feature array
sample_feature_array = np.zeros((len(all_graphs),len(features_of_interest)))

# Transform into array
all_graphs = np.array(all_graphs)

# ------------------------------------------------
# Write out graph features for each text sample
# ------------------------------------------------

print('\nNOW WRITING OUT...\n')

# For each sample...
for i, sample in enumerate(all_graphs):

    # Get features as dict
    feature_dict = vars(sample)

    # Save author and filename
    author_list.append(feature_dict['author_'])
    file_list.append(feature_dict['filename_'])

    # Initialize feature array for this sample
    features = np.zeros((len(features_of_interest)))

    # Fill feature array
    for j, feature in enumerate(features_of_interest):
        features[j] = feature_dict[feature]

    # Set target directory for file
    FILE_TAR_DIR = TARGET_DIR+feature_dict['filename_']

    # Save feature array for each sample
    np.save(FILE_TAR_DIR, features)

    # NOT NECESSARY:

    # Add sample features to complete feature array
    sample_feature_array[i,:] = features

    # Entity graph as flat array (could be used as feature)
    ent_graph = feature_dict['ent_graph_'].flatten()

# Print # skipped files and samples
print('Skipped files:\t', len(skip_files))
print('Samples:\t', sample_feature_array.shape[0])

# Get solution array
solution_array = np.array(author_list)

# Save complete arrays and feature names
np.save('../2_Features/info/misc/complete_EG_feature_array.npy', sample_feature_array)
np.save('../2_Features/info/feature_names/EG_feature_names.npy',features_of_interest)
