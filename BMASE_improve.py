
"""
@author: Hosein Toosi
"""
from tree_tools import *
from tree_io import *
from clustering import *
import numpy as np
import copy as cp
import pandas as pd
import itertools
import re
from sklearn.cluster import KMeans
import pickle
import argparse


def unique_Kmeans_clustering(As,Bs,num_clusters,n_init,pf,err):
    res = set()
    result = []
    data = (As+0.0)/(As+Bs)
    for count in range(n_init):
        assign = tuple(reorder_assignmens(KMeans(n_clusters=num_clusters,n_init=1).fit(data).labels_))
        if assign not in res:
            res.add(assign)
            result += [clustering(As,Bs,assign,pf,err)]
    return result

parser = argparse.ArgumentParser(description='Generates likely tumor evolution histories given somatic readcounts.')
parser.add_argument('infile', metavar='inputfile', type=str, nargs='+', help='an integer for the accumulator')
parser.add_argument('-e', metavar='error', default=0.001, type=float,help='Sequencing error probability (Default:0.001)')
parser.add_argument('-nc', metavar ='max_clusts', default=12, type=int, help='Maximum possible number of subclones')
parser.add_argument('-s', metavar ='sparsity', default=-1, type=float,help='this controls sparsity of solutions, between 0. and .1, considered 0 otherwise')
parser.add_argument('-n', metavar ='n_trees', default=3, type=int,help='Number of clusterings candidates analysed per number of clusters')
parser.add_argument('-t', metavar ='top_trees', default=1, type=int,help='Number of top solutions to return')
args = parser.parse_args()

inputfile = args.infile[0]
err = args.e
sparsity = args.s
n_trees = args.n
max_clusts= args.nc
top_trees = args.t

# inputfile = 'bamse.dat'
# err = 0.001
# sparsity = -1.0
# n_trees = 3
# max_clusts= 10
# top_trees = 1

print max_clusts
table_data = pd.read_table(inputfile)
max_clusts = min(len(table_data),max_clusts)
pf = part_func(len(table_data),max_clusts)
print len(table_data),pf


cluster_colors = ["plum", "lightyellow", "palegreen", "pink", "paleturquoise", "tan", "moccasin", "lightcyan3", "oldlace", "palevioletred", "olivedrab1", "lightgray", "steelblue1"]

sample_colors = ["red", "firebrick4", "brown4", "darkgoldenrod4" ,"deeppink1","darkgreen", "cyan4", "darkorchid", "blue", "dimgray"]   


ref_cols=[re.match('(.*)?.ref',x) for x in table_data.columns.values]
var_cols=[re.match('(.*)?.var',x) for x in table_data.columns.values]
ref_names =[x.string.replace('.ref','') for x in filter(None,ref_cols)]
var_names =[x.string.replace('.var','') for x in filter(None,var_cols)]
sample_names = set.intersection(set(ref_names),set(var_names))
sample_names = list(sample_names)
sample_names.sort()
print sample_names
X={'As':table_data[[x+'.var' for x in sample_names]].as_matrix(),'Bs':table_data[[x+'.ref' for x in sample_names]].as_matrix()}
print X['As'].shape
clusterings = []
for K in range(1,max_clusts+1):
    Y = (X['As']+0.0)/(X['Bs']+X['As'])
    print Y.shape
    score = -1*float("inf")
    assign = KMeans(n_clusters=K,n_init=100).fit(Y).labels_
    clusterings += [clustering(X['As'],X['Bs'],assign,pf,err)]
    raw_scores= np.array([x.cluster_prior + x.maxlikelihood for x in clusterings])
    print raw_scores
if np.any(np.diff(raw_scores)<0):
    candidate_numbers = raw_scores.argsort()[-3:]
else:
    candidate_numbers = [max_clusts]
print 'possible number of clusters:' , candidate_numbers
clusts = []
for x in candidate_numbers:
   clusts += list(unique_Kmeans_clustering(X['As'],X['Bs'],x,n_trees,pf,err))
for x,clust in enumerate(clusts):
   print 'analysis clustering: ' +str(x+1) +' of '+str(len(clusts))
#   clust.get_t_normals(err)
#    clust.get_trees()
   clust.get_candidate_tree()
   clust.get_trees2(clust.trees[0])
   print len(clust.trees)
    
#   if sparsity <0 :
   clust.analyse_trees_regular2()
trees = []
for clust in clusts:
   trees += clust.trees
ordered = np.argsort([-1*x['totalscore'] for x in  trees])
trees = [trees[x] for x in ordered]
for x in trees:
   x['sample_names']=list(sample_names)
sparse_trees =[]
for t in range(min(top_trees,len(trees))):
    this_tree = trees[t]
    if (sparsity>0 and sparsity <1):
        temp_clust = clustering(X['As'],X['Bs'],trees[t]['assign'],pf,err)
        temp_clust.trees = [trees[t]]
        temp_clust.analyse_trees_sparse2(sparsity,err)
        this_tree = temp_clust.trees[0]
        sparse_trees +=[cp.deepcopy(this_tree)]
        this_tree = remove_zeros(this_tree)
    this_tree = remove_zeros(this_tree)
    this_tree['vaf'] = this_tree['Bmatrix'].dot(this_tree['clone_proportions'])
    
    write_dot_files(this_tree,sample_colors,cluster_colors,'tree_number_'+str(t)+'_subclones.dot','tree_number_'+str(t)+'_samples.dot')
    write_text_output([this_tree],'tree_number_'+str(t)+'.txt')

if (sparsity > 0. and sparsity < 1.):
    pickle.dump(sparse_trees,open('bamse.pickle','w'))
    write_text_output(sparse_trees,'bamse_output.txt')
else:
    pickle.dump(trees[:min(top_trees,len(trees))],open('bamse.pickle','w'))
    write_text_output(trees[:min(top_trees,len(trees))],'bamse_output.txt')