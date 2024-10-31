"""
Code to perform Markov modelling using the selected features.

Note: much of this code is directly inspired from the PyEMMA2 documentation and tutorials:
    
    https://emma-project.org/
    
    Scherer et al. (2015) PyEMMA 2: A Software Package for Estimation, Validation and Analysis
    of Markov Models. Journal of Chemical Theory and Computation. Volume 11(11). 5525-5542. DOI: https://doi.org/10.1021/acs.jctc.5b00743

"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import mdtraj as md
import pandas as pd
import pyemma
import networkx as nx




# load csv file with vector distances and split into arrays for each receptor
df = pd.read_csv("", index_col=[0])
target = "hESR1"
temp = df[df["label"] == target]
temp = temp.drop("label", axis=1)
data_concatenated = temp.values



# investigate suitable lag times for tICA discretisation
sns.set(rc={"figure.dpi":600, 'savefig.dpi':600})
sns.set_style(style="ticks")
sns.set_context("paper", font_scale=1.3)

lags = [1, 2, 5, 10, 20] #bc MD frames are 100ps steps, lag of 1 step = 100ps

fig, axes = plt.subplots(5, 3, figsize=(10, 18))
for i, lag in enumerate(lags):
    tica = pyemma.coordinates.tica(data_concatenated, lag=lag, dim=2) 
    tica_concatenated = np.concatenate(tica.get_output())
    
    pyemma.plots.plot_feature_histograms(
        tica_concatenated,
        ['IC {}'.format(i + 1) for i in range(tica.dimension())],
        ax=axes[i, 0])
    
    axes[i, 0].set_title("lag time = {} steps".format(lag))
    axes[i, 1].set_title(
        "Density, actual dimension = {}".format(tica.dimension()))
    pyemma.plots.plot_density(
        *tica_concatenated[:, :2].T, ax=axes[i, 1], cbar=False, cmap="RdBu")
    pyemma.plots.plot_free_energy(
        *tica_concatenated[:, :2].T, ax=axes[i, 2], legacy=False)
for ax in axes[:, 1:].flat:
    ax.set_xlabel("IC1")
    
    ax.set_ylabel("IC2")
axes[0, 2].set_title("Pseudo free energy")
fig.tight_layout()



# perform tICA analysis using the selected lag time
tica = pyemma.coordinates.tica(data_concatenated, lag=1, dim=2) 
tica_output = tica.get_output()
tica_concatenated = np.concatenate(tica_output)

sns.set(rc={"figure.dpi":600, 'savefig.dpi':600})
sns.set(rc={'figure.figsize':(5,3.8)})
sns.set_style(style="ticks")
sns.set_context("paper", font_scale=1.3)

pyemma.plots.plot_density(*tica_concatenated[:, :2].T, logscale=True, cmap="viridis")
plt.xlabel("tICA1")
plt.ylabel("tICA2")



#look at trajectories in space of top two TICA ICs
fig, axes = plt.subplots(2, 1, figsize=(12, 4), sharex=True)
x = 0.1 * np.arange(tica_output[0].shape[0])

for i, (ax, tic) in enumerate(zip(axes.flat, tica_output[0].T)):
    ax.plot(x, tic, c='tab:blue')
    ax.set_ylabel('IC {}'.format(i + 1))
axes[-1].set_xlabel('time / ns')
fig.tight_layout()



# identify the optimal number of clusters for k-means using VAMP2 score as a heuristic
# with cross-validation on five unvalidated MSMs 
n_clustercenters = [5, 10, 25, 50, 75, 100, 150, 200, 300, 500, 750, 1000, 1500, 2000, 3000]

scores = np.zeros((len(n_clustercenters), 5))
for n, k in enumerate(n_clustercenters):
    for m in range(5):
        with pyemma.util.contexts.settings(show_progress_bars=False):
            _cl = pyemma.coordinates.cluster_kmeans(
                tica_output, k=k, max_iter=500, stride=1)
            _msm = pyemma.msm.estimate_markov_model(_cl.dtrajs, 5)
            scores[n, m] = _msm.score_cv(
                _cl.dtrajs, n=1, score_method="VAMP2", score_k=min(10, k))

fig, ax = plt.subplots()
lower, upper = pyemma.util.statistics.confidence_interval(scores.T.tolist(), conf=0.9)
ax.fill_between(n_clustercenters, lower, upper, alpha=0.3)
ax.plot(n_clustercenters, np.mean(scores, axis=1), "-o")
ax.semilogx()
ax.set_xlabel("$K$ cluster centres")
ax.set_ylabel("VAMP-2 score")
fig.tight_layout()



# cluster the tica coordinates using kmeans agorithm with VAMP2 n clusters and best lag
cluster = pyemma.coordinates.cluster_kmeans(tica, k=500, max_iter=500, stride=1, fixed_seed=42)
dtrajs_concatenated = np.concatenate(cluster.dtrajs)



# calculate implied timescales over a range of lag times. 
its = pyemma.msm.its(
    cluster.dtrajs, lags=[1,2,5,10,25], nits=9, errors="bayes")

fig, axes = plt.subplots(1, 3, figsize=(10, 3))
pyemma.plots.plot_feature_histograms(tica_concatenated, feature_labels=["IC1", "IC2"], ax=axes[0])
pyemma.plots.plot_density(*tica_concatenated.T, ax=axes[1], cbar=False, alpha=0.3, cmap='viridis')
axes[1].scatter(*cluster.clustercenters.T, s=8, c='tab:orange')
axes[1].set_xlabel("IC1")
axes[1].set_ylabel("IC2")
axes[1].set_title(r"$\mathcal{k}$ = 500")
pyemma.plots.plot_implied_timescales(its, ax=axes[2], units="ns", dt=0.1, xlog=True)
fig.tight_layout()



# estimate a Bayesian MSM with n-states and validate with CK-test
msm = pyemma.msm.bayesian_markov_model(cluster.dtrajs, lag=1, dt_traj='0.1 ns', conf="0.95")

print('fraction of states used = {:f}'.format(msm.active_state_fraction))
print('fraction of counts used = {:f}'.format(msm.active_count_fraction))
#^ must both be 1.0


nstates = 4 #set to number of macrostates
pyemma.plots.plot_cktest(msm.cktest(nstates, mlags=5), units='ns', dt=0.1)



# visualise the stationary distribution, reweighted free-energy and eigenvectors of the validated MSM
print(msm.stationary_distribution)     
print('sum of weights = {:f}'.format(msm.pi.sum()))

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
pyemma.plots.plot_contour(
    *tica_concatenated[:, :2].T,
    msm.pi[dtrajs_concatenated],
    ax=axes[0],
    mask=True,
    cbar_label="stationary distribution",
    cmap="viridis")
pyemma.plots.plot_free_energy(
    *tica_concatenated[:, :2].T,
    weights=np.concatenate(msm.trajectory_weights()),
    ax=axes[1],
    legacy=False)
for ax in axes.flat:
    ax.set_xlabel("IC1")
axes[0].set_ylabel("IC2")
axes[0].set_title("Stationary distribution", fontweight="bold")
axes[1].set_title("Reweighted free energy surface", fontweight="bold")
fig.tight_layout()



eigvec = msm.eigenvectors_right()
print("first eigenvector is one: {} (min={}, max={})".format(
    np.allclose(eigvec[:, 0], 1, atol=1e-15), eigvec[:, 0].min(), eigvec[:, 0].max()))

fig, axes = plt.subplots(1, 5, figsize=(16, 3))
for i, ax in enumerate(axes.flat):
    pyemma.plots.plot_contour(
        *tica_concatenated.T, eigvec[dtrajs_concatenated, i + 1], ax=ax, cmap='PiYG',
        cbar_label='{}. right eigenvector'.format(i + 2), mask=True)
    ax.set_xlabel('IC1')
    ax.set_ylabel('IC2')
fig.tight_layout()



# save progress (example filenames below)
msm.save("hERa_bayesian.pyemma", model_name="hERa_bayesian_msm", overwrite=True)
tica.save("hERa_bayesian.pyemma", model_name="tICA", overwrite=True)
cluster.save("hERa_bayesian.pyemma", model_name="tICA_clusters", overwrite=True)



# Coarse-grain the MSM with Perron cluster cluster analysis (PCCA) and Transition path theroy (TPT)
msm.pcca(nstates)



#stationary probabilities of metstable states
for i, s in enumerate(msm.metastable_sets):
    print('π_{} = {:f}'.format(i + 1, msm.pi[s].sum()))
    
    
    
#visualise fuzzy assignments for metstable states
fig, axes = plt.subplots(1, 4, figsize=(16, 3))
for i, ax in enumerate(axes.flat):
    pyemma.plots.plot_contour(
        *tica_concatenated.T,
        msm.metastable_distributions[i][dtrajs_concatenated],
        ax=ax,
        cmap="Blues",
        mask=True,
        cbar_label="metastable distribution {}".format(i + 1))
    ax.set_xlabel("IC1")
axes[0].set_ylabel("IC2")
fig.tight_layout()



#visualise states (crisp assignments on the tICA representation)
metastable_traj = msm.metastable_assignments[dtrajs_concatenated]

fig, ax = plt.subplots(figsize=(5, 4))
_, _, misc = pyemma.plots.plot_state_map(
    *tica_concatenated[:, :2].T, metastable_traj, ax=ax, cmap="viridis")
ax.set_xlabel("IC1")
ax.set_ylabel("IC2")
misc["cbar"].set_ticklabels([r"$\mathcal{S}_%d$" % (i + 1)
                             for i in range(nstates)])
fig.tight_layout()



# extract and save the top n microstate contributions of the mestable macrostates
highest_membership = msm.metastable_distributions.argmax(1)
coarse_state_centers = cluster.clustercenters[msm.active_set[highest_membership]]
micro_state_assignments = msm.metastable_assignments


#get PDB structures of the metastable macrostates
pcca_samples = msm.sample_by_distributions(msm.metastable_distributions, 100)

pdb = md.load("")
pdb_top = pdb.topology

#read in trajectory files and concatenate into single file, reset indices
traj1 = md.load("", top=pdb)
traj2 = md.load("", top=pdb)
traj3 = md.load("", top=pdb)

merged = md.join([traj1,traj2,traj3])

indices = []
for arr in pcca_samples:
    z = arr[:,1]
    indices.append(z)
indices = [list(arr) for arr in indices]
indices_df = pd.DataFrame(indices).T  
    
for counter, lst in enumerate(indices):
    frames = merged[lst]
    frames.save_pdb("pcca_samples_{}.pdb".format(counter + 1))



## kinetic analysis of the MSM and get mean-first passage times
print("state\tπ\t\tG/kT")
for i, s in enumerate(msm.metastable_sets):
    p = msm.pi[s].sum()
    print("{}\t{:f}\t{:f}".format(i + 1, p, -np.log(p)))



#extract mean first passage times (MFPT) between states and coarse-grain flux
mfpt = np.zeros((nstates, nstates))
for i in range(nstates):
    for j in range(nstates):
        mfpt[i, j] = msm.mfpt(
            msm.metastable_sets[i],
            msm.metastable_sets[j])


print("MFPT / ns:")
mfpt_df = pd.DataFrame(np.round(mfpt, decimals=2), index=range(1, nstates + 1), 
                       columns=range(1, nstates + 1))

inverse_mfpt = np.zeros_like(mfpt)
nz = mfpt.nonzero()
inverse_mfpt[nz] = 1.0 / mfpt[nz]

imfpt_df = pd.DataFrame(inverse_mfpt,index=range(1, nstates + 1), columns=range(1, nstates + 1))

pyemma.plots.plot_network(
    inverse_mfpt,
    arrow_label_format="%.1f ns",
    arrow_labels=mfpt,
    arrow_scale=3.0,
    state_labels=range(1, nstates + 1),
    size=12,
    state_sizes=highest_membership)
    


#create networkX object of the MSM
transition_matrix = msm.transition_matrix
prob_matrix = msm.observation_probabilities
assignments = msm.metastable_assignments

G = nx.Graph()
num_nodes = len(transition_matrix)
G.add_nodes_from(range(1, num_nodes + 1))


for i in range(num_nodes):
    for j in range(i + 1, num_nodes):  # Only consider upper triangle to avoid duplicate edges
        weight = transition_matrix[i][j]
        if weight > 0:
            G.add_edge(i + 1, j + 1, weight=weight) 
     
            
assignment_list = assignments.tolist()
for i, attribute in enumerate(assignment_list):
    G.nodes[i + 1]['attribute'] = attribute
    
nx.write_graphml(G, "hERa_MSM.graphml")       
            
            

# save everything (example filenames below)
msm.save("hERa_bayesian.pyemma", model_name="hERa_bayesian_msm", overwrite=True)
tica.save("hERa_bayesian.pyemma", model_name="tICA", overwrite=True)
cluster.save("hERa_bayesian.pyemma", model_name="tICA_clusters", overwrite=True)

np.savetxt("hERa_microstate_assignments.csv", assignments.reshape(-1,1), delimiter=",")
np.savetxt("hERa_MSM_transition_matrix.csv", transition_matrix, delimiter=",")
np.savetxt("hERa_MSM_probability_matrix.csv",prob_matrix, delimiter=",")
np.savetxt("hERa_tICA_coords.csv", tica_concatenated, delimiter=",")
np.savetxt("hERa_microstate_assignments.csv", dtrajs_concatenated.reshape(-1,1), delimiter=",")
np.savetxt("hERa_eigvectors.csv", eigvec, delimiter=",")
np.savetxt("hERa_state_size.csv", highest_membership, delimiter=",")
mfpt_df.to_csv("hERa_mfpt_matrix.csv")
np.savetxt("hERa_inverse_mfpt.csv", inverse_mfpt, delimiter=",")
indices_df.to_csv("hERa_pcca_frames.csv")




    