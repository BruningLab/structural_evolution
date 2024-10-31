"""
Calculation of Jensen-Shannon divergence (JSD)

Author: Daniel P. McDougal

This code takes as an input an alignment of amino acid sequences in fasta format. It then converts
all characters to uppercase and counts positional (p) and background frequencies (q) of amino
acids. The Jensen-Shannon divergence (JSD) is then calculated using the Scipy.stats.distance module.

The JSD of each resdiue position can then be plotted as a line plot, or alternatively, a barplot.
Additionally, given a csv file containing the JSD values of residues comprising each functional region,
a box plot can be made.

Included is also a code to replace the CA atoms of a protein structure with the JSD
value for easy visualisation. 

"""


import numpy as np
from scipy.spatial import distance
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
from Bio import AlignIO
import random



#Process the alignment
msa = AlignIO.read("alignment.fasta", "fasta")
alignment = [str(record.seq) for record in msa]

clean_msa = []
for ele in alignment:
    clean_msa.append(ele.upper())

amino_acids = "-ACDEFGHIKLMNPQRSTVWY"
aa_to_index = {aa: i for i, aa in enumerate(amino_acids)}



#Calculate the positional (p) and background frequencies (q) of amino acids
n_positions = len(clean_msa[0])
freqs = np.zeros((n_positions, len(amino_acids))) #create array to store frequency values

for position in range(n_positions):
    frequencies = np.zeros(len(amino_acids))
    
    for sequence in clean_msa:
        aa = sequence[position]
        if aa in aa_to_index:
            frequencies[aa_to_index[aa]] += 1
            
    probabilities = frequencies / len(clean_msa)
    freqs[position] = probabilities

freq_p = freqs#[:, 1:] uncheck to calculate without gaps
freq_q = np.sum(freq_p, axis=0)/n_positions



#Calculate Jensen-Shannon divergence of each residue position
js_scores = np.zeros((n_positions))

for position in range(n_positions):
    js_val = distance.jensenshannon(freq_p[position], freq_q, axis=0)
    js_scores[position] = js_val   

np.savetxt("JSD_values.csv", js_scores, delimiter=",")



#Plot the data using Matplotlib and Seaborn libraries
sns.set(rc={'figure.figsize':(8,3)})
sns.set(rc={"figure.dpi":1200, 'savefig.dpi':1200})
sns.set_style(style="ticks")
sns.set_context("paper", font_scale=1.6)



#Plot the data as a line plot
ax = plt.plot(js_scores)

plt.axhline(0.619, ls="--", lw="0.66", c="black")
plt.axhline(0.741, ls="--", lw="0.66", c="black")

plt.grid()

plt.ylabel("Conservation ($JSD$)")
plt.xlabel("Residue position")

#adjust xticks to number of residue positions and required spacing
plt.xticks(ticks=[0, 14, 29, 44, 59, 74, 89, 104, 119, 134, 149, 164, 179, 194, 209, 224, 238], 
               labels=("310", "", "340", "", "370", "", "400", "", "430", "", "460", "", "490", "", "510", "", "548"))
sns.despine()



#Plot the data as a boxplot per region
df = pd.read_csv("JSD_values_regions.csv")

lbp = np.array(df["LBP"])
af2 = np.array(df["AF2"])
an = np.array(df["AN"])
di = np.array(df["DIMER"])
mfn = np.array(df["FN"])
nf = np.array(df["NF"])

sns.set(rc={'figure.figsize':(3,3)}) #resize figure
sns.set_context("paper", font_scale=1.4)

melted = pd.melt(df).dropna()

sns.boxplot(data=melted, x="variable", y="value", saturation=1, showfliers=False, palette="BuPu")
sns.stripplot(data=melted, x="variable", y="value", jitter=True, color="silver",
              marker="$\circ$", ec="face", s=8, alpha=0.7)


plt.xticks(ticks=[0,1,2,3,4,5], labels=["LBP", "AF2", "AN", "DI", "FN", "NF"], rotation=45)
plt.ylabel("Conservation ($JSD$)")
plt.xlabel(None)
sns.despine()



#Reset bfactors of CA atoms in PDB file
from Bio.PDB import PDBParser, PDBIO

b_factors = list(js_scores)

parser = PDBParser()
structure = parser.get_structure("file", "structure.pdb") # ensure that number of residues = length js_scores

for i, model in enumerate(structure):
    for chain in model:
        for residue in chain:
            for atom in residue:
                if atom.get_name() == "CA":
                    atom.set_bfactor(b_factors[i])
                    i += 1
                
io = PDBIO()
io.set_structure(structure)
io.save("new_bfactors.pdb") 














