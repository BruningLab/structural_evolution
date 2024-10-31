"""
Network analysis of sequence space

Author: Daniel P. McDougal

This code takes as an input an alignment of amino acid sequences in fasta format. There is the option to create
three graphs:
    - a mininum spanning tree
    - a network connecting nodes only if the Hamming distance is equal to 1 (used in manuscript)
    - a fully connected graph with edge weights proportional to Hamming distance. Used for calculating
      shortest mutational paths with Dijkstra's algorithm
      
This code also contains scripting to calculate sequence constraint AUCs using sequence counts and the
input alignment of non-redundant sequences.

"""



import numpy as np
import pandas as pd
from Bio import AlignIO
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm
import json
import re
from collections import Counter



#Process the alignment
alignment = AlignIO.read("non-redundant_alignment.fasta", "fasta")
seq_ids = [seq.id for seq in alignment]
sequence_list = []

for record in alignment:    
    sequence_list.append(str(record.seq.upper()))

amino_acids = "ACDEFGHIKLMNPQRSTVWY-"
aa_to_index = {aa: i for i, aa in enumerate(amino_acids)}    

large_aln = AlignIO.read("full_alignment.fasta", "fasta")



# count the number of times a sequence (genotype) is observed in the full alignment
def count_sequence_occurrences(alignment1, alignment2):

    sequence_counts = Counter()  
    for record1 in alignment1:  
        sequence1 = str(record1.seq)        
        for record2 in alignment2:            
            sequence2 = str(record2.seq)                        
            if sequence1 in sequence2:                
                sequence_counts[sequence1] += 1

    return sequence_counts

sequence_counts = count_sequence_occurrences(alignment, large_aln)

for sequence, count in sequence_counts.items():    
    print(f"Sequence '{sequence}' appears {count} times in the second alignment within the first alignment.")



# calculate the Hamming distance matrix (genetic distance) between each sequence
hamming_dist_matrix = np.zeros((len(alignment), len(alignment)))

for i, record1 in enumerate(alignment):    
    for j, record2 in enumerate(alignment):        
        if i == j:            
            continue        
        hamming_dist = sum(a != b for a, b in zip(record1.seq.upper(), record2.seq.upper()))       
        hamming_dist_matrix[i][j] = hamming_dist 
        
ham_mat = pd.DataFrame(hamming_dist_matrix, columns=seq_ids, index=seq_ids)       
        
        

# create a fully-connected graph        
g = nx.Graph()

labels = ham_mat.index.to_list()
counts_list = list(sequence_counts.values())
label_value_dict = dict(zip(labels, counts_list))
patterns = ["ER_alpha", "ER_beta", "ER_gamma", "amER", "pER", "wER"] #can be replaced with any string in header


for i in range(len(ham_mat)):    
    label = ham_mat.index[i]    
    substring = next((pattern for pattern in patterns if re.search(pattern, label)), "other")  
    count_value = label_value_dict.get(label, 0)    
    g.add_node(i, substring=substring, count=count_value, )

for node in g.nodes():
    index = node    
    sequence = str(alignment[index].seq)    
    g.nodes[node]["sequence"] = sequence


for i in range(len(ham_mat)):
  for j in range(len(ham_mat)):
    if i != j:
      g.add_edge(i, j, weight=ham_mat.iloc[i, j])



# create a dictionary to store alignment information
alignment_dict = {}

for seq in alignment:
    alignment_dict[seq.id] = {"seq": str(seq.seq), "description": seq.description}

   

# function to calculate pairwise shortest paths in the fully-connected graph using Dijkstra's algorithm
def shortest_paths_(start, end, alignment_dict):

    try:
        path = nx.dijkstra_path(g, start, end, weight="weight") 
        
    except nx.NetworkXNoPath:
        print("No path found between nodes", start, "and", end)        
        return None
    
    node_seqs = [alignment_dict[node]['seq'] for node in path]    
    path_length = nx.dijkstra_path_length(g, start, end, weight="weight") #uses the Hamming distance
    
    return path, path_length, node_seqs



# create a dictionary to story shortest path information
shortest_paths = {}

for i in tqdm(range(len(seq_ids))):
    
    for j in range(i+1, len(seq_ids)):
        
        start = seq_ids[i]
        end = seq_ids[j]
        path, path_length, node_seqs = shortest_paths_(start, end, alignment_dict=alignment_dict)
        shortest_paths.setdefault(start, {})[end] = {"path": path, "length": path_length, "node_seqs": node_seqs}



# extract the path lengths for all pairs
all_path_lengths = []

for start, end_dict in shortest_paths.items():
    for end, path_info in end_dict.items():
        path_length = path_info["length"]
        all_path_lengths.append(path_length)
        

# save all the information to a json file and also graph
with open("shortest_paths.json", "w") as f:
    json.dump(shortest_paths, f)
nx.write_graphml(g,"connected_graph.graphml") 

   

#plot the distribution of path lengths    
sns.set(rc={'figure.figsize':(3,3)})
sns.set(rc={"figure.dpi":1200, 'savefig.dpi':1200})
sns.set_style(style="ticks")
sns.set_context("paper", font_scale=1.3)


ax = plt.hist(all_path_lengths, bins=15, color="tab:gray")
plt.grid(axis="y")
plt.ylabel("Count")
plt.xlabel("Shortest path length")
sns.despine()    



# create a minimum spanning tree (MST) from the fully connected graph
mst = nx.minimum_spanning_tree(g, weight="weight", algorithm="prim")



def total_edge_weight(graph):
  return sum(edge[2]["weight"] for edge in graph.edges(data=True))



original_weight = total_edge_weight(g)
mst_weight = total_edge_weight(mst)



# do community analysis with Louvain clustering algorithm
partitions = nx.community.louvain_communities(mst, seed=42)
for community_label, nodes in enumerate(partitions):
    for node in nodes:
        mst.nodes[node]["community"] = community_label

nx.write_graphml(mst,"MST_network.graphml")



# create a graph where nodes are connected by an edge if the Hamming distance between them = 1
labels = ham_mat.index.to_list()
counts_list = list(sequence_counts.values())
label_value_dict = dict(zip(labels, counts_list))
graph = nx.Graph()
patterns = ["ER_alpha", "ER_beta", "ER_gamma", "amER", "pER", "wER"] #can be replaced with any string in header


for i in range(len(ham_mat)):    
    label = ham_mat.index[i]    
    substring = next((pattern for pattern in patterns if re.search(pattern, label)), "other")  
    count_value = label_value_dict.get(label, 0)    
    graph.add_node(i, substring=substring, count=count_value, )

    for j in range(i + 1, len(ham_mat)):        
        if ham_mat.iloc[i, j] <= 1:            
            graph.add_edge(i, j)



for node in graph.nodes():
    index = node    
    sequence = str(alignment[index].seq)    
    graph.nodes[node]["sequence"] = sequence



for i in range(len(ham_mat)):
    node = i
    attributes = graph.nodes[node]
    print(f"Node {node}: {attributes}")



partitions_ = nx.community.louvain_communities(graph, seed=42)
for community_label, nodes in enumerate(partitions_):
    for node in nodes:
        graph.nodes[node]["community"] = community_label



""" save stuff """

nx.write_graphml(graph,"_.graphml")
ham_mat.to_csv("_hamming_dist_matrix.csv") 

counts_seqs = [sequence_list, counts_list]
count_df = pd.DataFrame(sequence_list, index=ham_mat.index, columns=["Seq"])
count_df["Count"] = counts_list
count_df.to_csv("_seqs_count.csv")



# calculate the fraction of nodes reached from a source node for each mutational step
fractions_reached_all_nodes = []

for col_idx in range(ham_mat.shape[1]):
    distances = np.array(ham_mat.iloc[:, col_idx])
    fractions_reached = []

    max_ = int(distances.max() + 1)
    
    for step in range(max_ + 1):  # Start from step 0
        count = np.sum(distances <= step)
        fraction_reached = count / len(distances)
        fractions_reached.append(fraction_reached)

    fractions_reached_all_nodes.append(fractions_reached)



fractions_df = pd.DataFrame(fractions_reached_all_nodes).T
fractions_df.columns = [f'Node_{i+1}' for i in range(ham_mat.shape[1])]
fractions_df.to_csv("")



sns.set(rc={'figure.figsize':(3,3)})
sns.set(rc={"figure.dpi":1200, 'savefig.dpi':1200})
sns.set_style(style="ticks")
sns.set_context("paper", font_scale=1.4)



grey_palette = ["tab:grey"] * fractions_df.shape[1]
ax = sns.lineplot(fractions_df, legend=False, dashes=False, alpha=0.33, palette=grey_palette, lw=0.66)
ax = sns.lineplot(fractions_df.mean(axis=1), lw=2.25)
#adjust accordingly
#plt.xticks(ticks=[0,5,10,15,20])
plt.ylabel("Fraction reached")
plt.xlabel("Steps")



# calculate sequence constraint and AUCs for each functional region

#calculate LBP cumulative fraction and new tick coords
LBP = pd.read_csv("ER_LBD_LBP_seqs_count.csv", index_col=[0])
lbp = np.array(LBP["Count"])
lbp_cumm_frac = [sum(lbp[:i+1]) for i in range(len(lbp))]
lbp_cumm_percent = [fraction / lbp_cumm_frac[-1] for fraction in lbp_cumm_frac]
lbp_cumm_percent.insert(0, 0)
lbp_ticks = np.arange(1,59)
lbp_transformed = (lbp_ticks - 1)/(58-1)
lbp_transformed = np.append(lbp_transformed, 1)

#calculate AF2 cumulative fraction and new tick coords
AF2 = pd.read_csv("ER_LBD_AF2_seqs_count.csv", index_col=[0])
af2 = np.array(AF2["Count"])
af2_cumm_frac = [sum(af2[:i+1]) for i in range(len(af2))]
af2_cumm_percent = [fraction / af2_cumm_frac[-1] for fraction in af2_cumm_frac]
af2_cumm_percent.insert(0, 0)
af2_ticks = np.arange(1,95)
af2_transformed = (af2_ticks - 1)/(94-1)
af2_transformed = np.append(af2_transformed, 1)

#calculate AN cumulative fraction and new tick coords
AN = pd.read_csv("ER_LBD_allosteric_network_seqs_count.csv", index_col=[0])
an = np.array(AN["Count"])
an_cumm_frac = [sum(an[:i+1]) for i in range(len(an))]
an_cumm_percent = [fraction / an_cumm_frac[-1] for fraction in an_cumm_frac]
an_cumm_percent.insert(0, 0)
an_ticks = np.arange(1,164)
an_transformed = (an_ticks - 1)/(163-1)
an_transformed = np.append(an_transformed, 1)

#calculate DI cumulative fraction and new tick coords
DI = pd.read_csv("ER_LBD_dimer_interface_seqs_count.csv", index_col=[0])
di = np.array(DI["Count"])
di_cumm_frac = [sum(di[:i+1]) for i in range(len(di))]
di_cumm_percent = [fraction / di_cumm_frac[-1] for fraction in di_cumm_frac]
di_cumm_percent.insert(0, 0)
di_ticks = np.arange(1,343)
di_transformed = (di_ticks - 1)/(342-1)
di_transformed = np.append(di_transformed, 1)

#calculate FN cumulative fraction and new tick coords
FN = pd.read_csv("ER_LBD_folding_network_seqs_count.csv", index_col=[0])
fn = np.array(FN["Count"])
fn_cumm_frac = [sum(fn[:i+1]) for i in range(len(fn))]
fn_cumm_percent = [fraction / fn_cumm_frac[-1] for fraction in fn_cumm_frac]
fn_cumm_percent.insert(0, 0)
fn_ticks = np.arange(1,321)
fn_transformed = (fn_ticks - 1)/(320-1)
fn_transformed = np.append(fn_transformed, 1)



# calculate AUCs
from sklearn import metrics

lbp_AUC = metrics.auc(lbp_transformed, lbp_cumm_percent)
af2_AUC = metrics.auc(af2_transformed, af2_cumm_percent)
an_AUC = metrics.auc(an_transformed, an_cumm_percent)
di_AUC = metrics.auc(di_transformed, di_cumm_percent)
fn_AUC = metrics.auc(fn_transformed, fn_cumm_percent)



sns.set(rc={'figure.figsize':(3,3)}) #same dimensions as JSD plot
sns.set(rc={"figure.dpi":1200, 'savefig.dpi':1200})
sns.set_style(style="ticks")
sns.set_context("paper", font_scale=1.4)



plt.plot(lbp_transformed, lbp_cumm_percent, lw=2, ls="-")
plt.plot(af2_transformed, af2_cumm_percent, lw=2, ls="--")
plt.plot(an_transformed, an_cumm_percent, lw=2, ls="-")
plt.plot(di_transformed, di_cumm_percent, lw=2, ls="--")
plt.plot(fn_transformed, fn_cumm_percent, lw=2, ls="-")
plt.plot([0,1], [0,1], ls="--", color="black")

plt.xlabel("Fraction of genotypes", fontsize=12)
plt.ylabel("Fraction of population", fontsize=12)
plt.xticks(ticks=[0,0.25,0.5,0.75,1], labels=["0", "0.25", "0.5", "0.75", "1"])
plt.yticks(ticks=[0,0.25,0.5,0.75,1], labels=["0", "0.25", "0.5", "0.75", "1"])
#plt.grid(True)

patch1 = mpatches.Patch(color="tab:blue", label="LBP")
patch2 = mpatches.Patch(color="tab:orange", label="AF2")
patch3 = mpatches.Patch(color="tab:green", label="AN")
patch4 = mpatches.Patch(color="tab:red", label="DI")
patch5 = mpatches.Patch(color="tab:purple", label="FN")

plt.legend(handles=[patch1, patch2, patch3, patch4, patch5], frameon=False, loc="best",
           bbox_to_anchor=(1.04, 1), borderaxespad=0, fontsize=14)
