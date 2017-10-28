# BAMSE
BAyesian Model SElection for Inferring the subclonal history of tumor samples

Usage:
=====

usage: bamse.py [-h] [-e error] [-nc max_clusts] [-s sparsity] [-n n_trees]
                [-t top_trees]
                inputfile [inputfile ...]
                
inputfile: a tab delimited file including with somatic mutations as rows and columns named sample_name.ref for reference reads and sample_name.var for each sample

sequencing_error: sequencing error rate for ref -> var and vice versa

sparsity: a number between zero and one that represent the prior probability that any subclone is absent at any sample, negative values are interpreted as zero

num_trees: number of top KMeans clusterings to consider

max_clusters: maximum number of clusters to consider for KMeans

top_trees: number of top solutions to produce output for


Output:
========

dot files for visualization with graphviz

text files and pickle files to load with python, one per tree, describing the trees.  
