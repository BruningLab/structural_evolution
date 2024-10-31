"""
Code to perform contact analysis of microstate assignments to uncover evolutionarily conserved
allosteric and protein folding networks.

Author: Daniel McDougal

"""

import os
import mdtraj as md
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns



# load pcca samples in multi-pdb format from a directory
paths = os.listdir()
trajs = []

for traj in paths:
    trajs.append(md.load(traj))
    
samples = md.join(trajs)



# calculate contacts >> default setting and "closest heavy" atoms, include EST (last "residue")
contacts = md.compute_contacts(samples,ignore_nonprotein=False)
matrix = md.geometry.squareform(contacts[0],contacts[1])

filtered_matrix = np.where(matrix <= 0.42, matrix, 0) # less than or equal to 4.2 angstroms
binary_matrix = np.where(filtered_matrix != 0, 1, 0).sum(axis=0) / len(samples)

plt.imshow(binary_matrix)



# convert to dataframe and save all contact frequencies
res_list = []

for res in samples.topology.residues:
    res_list.append(res)
    
contact_matrix = pd.DataFrame(binary_matrix, index=res_list, columns=res_list) #add [:-1] to exclude LIG
contact_matrix.to_csv("_pcca_contact_frequency_matrix.csv")



# in excel, adjust matrices to accomodate gaps etc and load in as arrays
hERa = np.array(pd.read_csv("hERa_matrix.csv", index_col=[0]))
hERb = np.array(pd.read_csv("hERb_matrix.csv", index_col=[0]))
rfERa = np.array(pd.read_csv("rfERa_matrix.csv", index_col=[0]))
rfERb = np.array(pd.read_csv("rfERb_matrix.csv", index_col=[0]))
rfERg = np.array(pd.read_csv("rfERg_matrix.csv", index_col=[0]))


# merge the matrices and then calculate probability of each contact 
stacked = np.stack((hERa,hERb,rfERa,rfERb,rfERg))
final_matrix = np.sum(stacked, axis=0)/5


# plot the matrix
sns.set(rc={'figure.figsize': (5, 5)})
sns.set(rc={"figure.dpi": 600, 'savefig.dpi': 600})
sns.set_style(style="ticks")
sns.set_context("paper")
sns.set_context("paper", font_scale=1.6)


plt.imshow(final_matrix, cmap="BuPu")

plt.xticks(ticks=[0,50,100,150,200,239], labels=["310", "360", "410", "470", "520","548"])
plt.yticks(ticks=[0,50,100,150,200,239], labels=["310", "360", "410", "470", "520","548"])

plt.ylabel("Residue position")
plt.xlabel("Residue position")
cbar = plt.colorbar(shrink=0.66)
cbar.set_label("Contact occupancy")


# save the final matrix
np.savetxt("ER_LBD_pcca_contacts.csv", final_matrix, delimiter=",")


# filter to remove backbone contacts (within 5 residues)
lower = np.tril(final_matrix, k=-5)
np.savetxt("ER_LBD_pcca_contacts_filtered.csv", lower, delimiter=",")



# extract contacts occupied >= 0.9 of the time
filtered_lower = np.where(lower >= 0.9, lower, 0)
np.savetxt("ER_LBD_pcca_contacts_filtered_conserved.csv", filtered_lower, delimiter=",")


# create a network representation
num_residues = filtered_lower.shape[0]

g = nx.Graph()

node_names = range(310, 310 + num_residues)
g.add_nodes_from(node_names)

for i in range(num_residues):
    for j in range(i + 1, num_residues):
        weight = contact_matrix[j, i]  # Note the reversal of indices for lower triangular matrix
        if weight > 0:
            g.add_edge(node_names[i], node_names[j], weight=weight)

nx.write_graphml(g, "_.graphml")

