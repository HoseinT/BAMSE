from clustering import *
from tree_io import get_ascii_tree
from itertools import combinations

K=6
alltrees = 0
for m in range(1,K): 
    leaf_sets = [x for x in itertools.combinations(range(K),m)]
    for leaf_set in leaf_sets:
        sample_prufer = prufer_treeset(K,list(leaf_set),None)
        values = sample_prufer.values
        rem_choices = sample_prufer.rem_choices
        trees = [sample_prufer]
        while rem_choices>0:
            next_trees = []
            for tree in trees:
                next_trees += tree.spawn(10)
            trees = next_trees
            rem_choices = trees[0].rem_choices
        # for x in trees:
        #     print(repr(x))
        print(len(trees),list(leaf_set))
        alltrees += len(trees)
print(alltrees)
print(K**(K-1))