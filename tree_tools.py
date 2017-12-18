#!/usr/bin/env python2
""" This module contains general functions for dealing with evolutionary trees"""

import numpy as np
import copy as cp
import itertools

def indexTobinary(inds, length):
    """ input a set of indexes and output a list with those indices set to one"""
    a = [0 for _ in range(length)]
    for ind in inds:
        a[ind] = 1
    return a


def get_matrix(otree):
    """ Returns the Bmatrix of a tree in numpy and the set of ansectors for each node"""
    tree = list(otree)
    K = len(tree)
    desc = np.diag(np.ones(K))
    ans = [[k] for k in range(K)]
    node = tree.index(-1)
    frontier = [node]
    while len(frontier) > 0:
        node = frontier[-1]
        childs = get_childs(tree, node)
        for x in childs:
            ans[x] += ans[node]
        if len(childs) > 0:
            for x in itertools.product(ans[node], childs):
                desc[x] = 1
        del frontier[-1]
        frontier += childs
    return desc, ans


def reorder_tree(tree, old_labels, new_labels):
    """ Given a tree and current labeling and newlabeling, returns the new tree"""
    locs = [old_labels.index(x) for x in new_labels]
    newlocs = [new_labels.index(x) for x in old_labels]
    old_parents = [tree[x] for x in locs]
    new_tree = [newlocs[x] if not x == -1 else -1 for x in old_parents]
    return new_tree


def canonizeF(tree, node=-1):
    """ Returns the canonical unlabeled string representation and score for each tree"""
    if node == -1:
        node = list(tree).index(-1)
    children = get_childs(tree, node)
    if len(children) == 0:
        return {'canstr':'[]', 'scre':1}
    else:
        a = [canonizeF(tree, x) for x in children]
        scores = [x['scre'] for x in a]
        #print scores
        canonstrs = [x['canstr'] for x in a]
        #print canonstrs
        score = np.prod([len(list(group)) for key, group in itertools.groupby(canonstrs)])* \
        np.prod(scores)
        canon_str = '['+''.join(x for x in sorted(canonstrs))+']'
        return {'canstr': canon_str, 'scre': score}

def get_tree(mat):
    """ Return the tree given Bmatrix"""
    rev = np.linalg.inv(mat)
    tree = -1*np.ones(len(mat))
    wheres = np.where(rev == -1)
    tree[wheres[1]] = wheres[0]
    return tree

def sim_data_readcounts(K, M, altrees, num_mut, coverage=200, min_in_clust=2, error=0.003):
    """ Generates simulated data:
    K = number of subclones
    M = number of samples
    num_mut = number of mutations
    coverage is fixed
    min_in_clust = minnimum number of mutations in each cluster
    error = sequencing error"""

    ind = np.random.randint(len(alltrees[K]))
    true_tree = alltrees[K][ind]
    true_sm = np.array([np.random.dirichlet(np.ones(K+1))[:K] for m in range(M)])
    B = get_matrix(true_tree)[0]
    true_cm = np.array([B.dot(x) for x in true_sm])
    true_cm = true_cm.transpose()
    ps = np.random.dirichlet(np.ones(K))
    assign = np.random.choice(range(K), num_mut-min_in_clust*K, p=ps)
    assign = np.concatenate((np.repeat(np.arange(K), min_in_clust), assign))
    np.random.shuffle(assign)
    covs = coverage*np.ones((num_mut, M)).astype(int)
    As = np.random.binomial(covs, true_cm[assign]*(0.5-error)+error)
    Bs = covs-As
    return {'As':As, 'Bs':Bs, 'vaf':true_cm, 'assign':assign, 'tree':true_tree, \
    's':np.array(true_sm).transpose(), 'Bmatrix': get_matrix(true_tree)[0], 'Amatrix':\
    np.linalg.inv(get_matrix(true_tree)[0])}

def reorder_assignmens(assign):
    """reorders assignments so that the the fisrt element is assgned to first cluster
    and so on"""
    index = np.unique(assign, return_index=True)[1]
    return np.argsort(np.argsort(index))[assign]

def get_childs(tree, node):
    """ retruns the childs of a node in tree"""
    return [x for x in range(len(tree)) if tree[x] == node]

def get_desc(tree,node):
    """Returns the Descendents of a node in tree"""
    if len(get_childs(tree, node)) == 0:
        return []
    else:
        return get_childs(tree, node)+reduce(lambda x, y: x+y, [get_desc(tree, x) \
        for x in get_childs(tree, node)])


def collapse_node(t,n):
    """ Removes a node "n" with one child from tree "t" and output the resulting tree"""
    if not len(get_childs(t, n)) == 1:
        raise 'can not collapse node with other than one child'
    child = int([x for x in range(len(t)) if t[x] == n][0])
    par = t[n]
    res = cp.deepcopy(t)
    res[child] = par
    x = range(n)+range(n+1, len(t))
    res = res[x]
    for x in range(len(res)):
        if res[x] > n:
            res[x] -= 1
    return res

def collapse_assign(c, n, ch):
    """ given assignment vector "c", this reassignes everything assigned to "n" to "ch"
    and returns new assignment"""
    res = cp.deepcopy(c)
    for x in range(len(res)):
        if res[x]==n:
            res[x] = ch
        if res[x] > n:
            res[x] -= 1
    return res

def remove_zeros(tree):
    """ remove nodes with zero cell fraction and only one child"""
    sums = [sum(y) for y in tree['clone_proportions']]
    ns = [n for n in range(len(sums)) if sums[n] < 0.001]
    nchilds = [len(get_childs(tree['tree'], n)) for n in ns]
    while 1 in nchilds:
        n = ns[nchilds.index(1)]
        child = int([x for x in range(len(tree['tree'])) if tree['tree'][x] == n][0])
        tree['tree'] = collapse_node(tree['tree'], n)
        tree['assign'] = collapse_assign(tree['assign'], n, child)
        tree['clone_proportions'] = np.delete(tree['clone_proportions'], n, 0)
        sums = [sum(y) for y in tree['clone_proportions']]
        ns = [n for n in range(len(sums)) if sums[n] < 0.001]
        nchilds = [len(get_childs(tree['tree'], n)) for n in ns]
    tree['Bmatrix'] = get_matrix(tree['tree'])[0]
    tree['Amatrix'] = np.linalg.inv(tree['Bmatrix'])
    return tree


# def get_child_nodes(prufer, node):
#     return [x[not x.index(node)] for x in prufer.tree_repr if node in x]
