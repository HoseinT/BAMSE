#!/usr/bin/env python2
"""The clustering class is defined here,  it defines a clustering, the prior
and the error model for the clustering"""
from tree_tools import *
from tree_io import get_ascii_tree
import numpy as np
import itertools
from scipy.special import gammaln
import copy as cp
from scipy.misc import comb, logsumexp
import cvxpy as cvx
from math import factorial
import functools

#tree_numbers is the sequence of number of unlabeled rooted trees 
tree_numbers = np.array([1, 1, 1, 2, 4, 9, 20, 47, 108, 252, 582, 1345,\
 3086, 7072, 16121, 36667, 83099, 187885, 423610, 953033, 2139158, 4792126,\
  10714105, 23911794, 53273599, 118497834, 263164833, 583582570])

def distrib(a,b,err,length = 512):
    x = np.linspace(0,1,length)
    cf = x*(0.5-err)+err
    y = gammaln(a+b+1)-gammaln(a+1)-gammaln(b+1) + np.log(cf)*a+np.log(1-cf)*b
    return y

def prior_fractions(K,Dir_alpha=1, length=512):
    x = np.linspace(0,1,length)
    if (Dir_alpha == 1):
        return 1.0/K * ( 0 - gammaln(K * Dir_alpha)) * np.ones(length)  
    return 1.0/K * (K * gammaln(Dir_alpha) - gammaln(K * Dir_alpha) + (Dir_alpha -1) * np.log(x))      

def log_conf_penalty(assign,pf):
    partition = np.unique(assign, return_counts = True)[1]
    partition_config = np.unique(partition, return_counts = True)[1]
    return np.sum(gammaln(partition+1.)) - gammaln(np.sum(partition)+1) + np.sum(gammaln(partition_config+1.)) - np.log(pf[len(partition)-1]) + np.log(tree_numbers[len(partition)]) - (len(partition)-1)*np.log(len(partition))



class clustering:

    def __init__(self, variant_reads, normal_reads, assignments, pf, err):
        # am and bmm are mutationsXsamples matrices for variant and normal reads
        self.am = np.array(variant_reads)
        self.bm = np.array(normal_reads)
        #0 based integer vactor of assignment of variants to clusters
        self.assign = np.array(assignments).astype(int)
        self.pf = pf
        self.err = err

        #num mutations
        N = len(self.am)
        #num clusters
        K = max(assignments) + 1

        #### aggregate reads for each cluster + determine subclone order for heuristic search
        self.As = np.array([np.sum(self.am[self.assign == x], 0) for x in range(K)])
        self.Bs = np.array([np.sum(self.bm[self.assign == x], 0) for x in range(K)])
        K, M = self.As.shape
        self.distribs = [[distrib(self.As[x, m], self.Bs[x, m], self.err) for m in range(M)] \
        for x in range(K)]
        self.priors = [[prior_fractions(K) for m in range(M)] \
        for x in range(K)]


        # log maximum likelihood of data for this clustering
        self.maxlikelihood = np.sum(gammaln(self.am + self.bm+2.) - \
        gammaln(self.am + 1.) - gammaln(self.bm + 1.)) - np.sum(gammaln(self.As +self.Bs +2) \
        -gammaln(self.As + 1) - gammaln(self.Bs + 1))
        # model prior for clustering
        self.cluster_prior = log_conf_penalty(self.assign, self.pf)

    def reorder(self):
        K, M = self.As.shape
        self.order = sorted(np.unique(self.assign), key=functools.cmp_to_key(self.compare_clusters))

        ### reorder subclones
        self.assign = np.array([self.order.index(x) for x in self.assign])
        self.As = np.array([np.sum(self.am[self.assign == x], 0) for x in range(K)])
        self.Bs = np.array([np.sum(self.bm[self.assign == x], 0) for x in range(K)])

        self.distribs = [[distrib(self.As[x, m], self.Bs[x, m], self.err) for m in range(M)] \
        for x in range(K)]
        self.priors = [[prior_fractions(K) for m in range(M)] \
        for x in range(K)]
        self.means = (self.As +0.)/(self.As +self.Bs)

    def get_tree_score(self,tree,nodes):
        #print nodes
        return np.sum(tree_integrate_multisample(tree, \
        np.array([self.distribs[n] for n in nodes]), np.array([self.priors[n] for n in nodes])))
    
    def compare_clusters(self, clustA, clustB):  ### add error
        """ With ISA constraints compares sublcones A and B to see which is statistically more
        probable to be the parent of the other"""
        return int(np.sign(self.get_tree_score(np.array([-1,0]), [clustB,clustA])  - \
        self.get_tree_score(np.array([-1,0]), [clustA,clustB])))

    def mean_average(self, clust):
        As = np.array([np.sum(self.am[self.assign == x], 0) for x in range(max(self.assign)+1)])
        Bs = np.array([np.sum(self.bm[self.assign == x], 0) for x in range(max(self.assign)+1)])
        return 1.0 - np.mean((As +0.)/(As + Bs))
        

    def get_candidate_tree(self):
        """ Get an initial Heuristic Tree"""
        tree = [-1]
        if len(self.order) == 1:
            self.trees = [{'tree':tree, 'root':0}]
            return [{'tree':tree, 'root':0}]
        for n in range(1, len(self.order)):
            scores = [self.get_tree_score(tree+[x], [y for y in range(n+1)]) for x in range(n)]
            tree += [np.argmax(scores)]
        self.trees = [{'tree':tree, 'root':0}]
        return [{'tree':tree, 'root':0}]

    def get_trees2(self, candidate_tree):
        """ Get all the trees that have a larger score than candidate_tree
        generating trees using the construction method in pitman double counting proof
        for Cayley's formula """
        K, M = self.As.shape
        score = self.get_tree_score(candidate_tree['tree'], [y for y in range(K)]) + \
        np.log(canonizeF(candidate_tree['tree'])['scre'])
        allts = []
        for rootnode in range(K):
            ts = [treeset(K)]
            #print rootnode
            nodes = [n for n in range(K) if not n == rootnode]
            #print nodes
            for n in nodes:
                new_ts = []
                for t in ts:
                    ind = t.node_assigns[n]
                    allowed_parents = [p for p in range(K) if p not in t.values[ind]['nodes']]
                    for par in allowed_parents:
                        candidate_t = t.join_root_to_node(n, par)
                        candidate_score = np.sum([self.get_tree_score(x['tree'], x['nodes']) for x in candidate_t.values])
                        if candidate_score + np.log(factorial(candidate_t.number_of_leaves())) + 3\
                         >= score:
                            print('scores=', candidate_score, score)
                            print('trees =', candidate_t.values, candidate_tree['tree'])
                            new_ts += [candidate_t]
                ts = new_ts
            allts += ts
        self.trees = [{'tree':reorder_tree(t.values[0]['tree'], t.values[0]['nodes'], [y for y in range(K)]),\
        'root':t.values[0]['nodes'][t.values[0]['tree'].index(-1)]} for t in allts \
        if self.get_tree_score(t.values[0]['tree'], t.values[0]['nodes']) + \
        np.log(canonizeF(t.values[0]['tree'])['scre']) >= score]
        return self.trees

    def get_trees3(self, candidate_tree):
        """ Get all the trees that have a larger score than candidate_tree
        generating trees using the construction method in prufer_treeset class """
        K, M = self.As.shape
        distribs = self.distribs
        priors = self.priors
        score = self.get_tree_score(candidate_tree['tree'], [y for y in range(K)]) + \
        np.log(canonizeF(candidate_tree['tree'])['scre'])
        results = []
        subtree_dict = dict()
        counter = 1
        for number_of_clones in range(1, K):
            leaf_sets = [x for x in itertools.combinations(range(K), number_of_clones)]
            for leaf_set in leaf_sets:
                counter += 1
                print(counter)
                sample_prufer = prufer_treeset(K, list(leaf_set), distribs, priors)
                results += sample_prufer.spawn_all_the_way(score,subtree_dict)
        return results

    def analyse_trees_regular2(self):
        """ get tree metrices, comute model posterior and ML values without the sparsity"""
        K, M = self.As.shape
        for tree in self.trees:
            B = get_matrix(tree['tree'])[0]
            A = np.linalg.inv(B)
            tree['Amatrix'] = A
            tree['Bmatrix'] = B
            tree['assign'] = self.assign
            score = self.get_tree_score(tree['tree'], [x for x in range(K)])
            get_max_values_r2(tree, self.As,self.Bs, self.err)
            tree['vaf'] = tree['Bmatrix'].dot(tree['clone_proportions'])
            tree['score'] = score +np.log(canonizeF(tree['tree'])['scre'])
            tree['totalscore']= tree['score']+self.cluster_prior +             self.maxlikelihood


def log_conf_penalty2(assign,alphaa):
    partition = np.unique(assign, return_counts = True)[1]
    return gammaln(alphaa)+np.sum(gammaln(partition+1.))+len(partition)*np.log(alphaa) - gammaln(np.sum(partition)+alphaa)


class treeset:
    ###  a class containing a set of trees for the branch and bound step
    def __init__(self,length):
        ### modify to include more init options
        self.values= []
        for x in range(length):
            self.values +=  [{'tree':[-1],'nodes':[x],'root':[x]}]
        self.node_assigns = [y for y in range(length)]

    def join_root_to_node(self,root,node):
        #### places root under node and returns a new treeset
        new_treeset = cp.deepcopy(self)
        ind1 = new_treeset.node_assigns[node]
        ind2 = new_treeset.node_assigns[root]
        tree1 = new_treeset.values[ind1]
        tree2 = new_treeset.values[ind2]
        if not tree2['tree'][tree2['nodes'].index(root)] == -1:
            print('Error: the root argumnet must be a root in the treeset')
            return
        if ind1 ==ind2 :
            print('Error: select two distict trees')
            return
        tree = [x+len(tree1['tree']) if not x==-1 else tree1['nodes'].index(node) for x in tree2['tree']]
        tree = tree1['tree']+tree
        nodes = tree1['nodes']+tree2['nodes']
        root = tree1['root']
        new_treeset.values = [i for j, i in enumerate(new_treeset.values) if j not in [ind1,ind2]]
        new_treeset.values += [{'tree':tree,'nodes':nodes,'root':root}]
        for i,x  in enumerate(new_treeset.node_assigns):
            if x in [ind1,ind2]:
                new_treeset.node_assigns[i] = len(new_treeset.values)-1 
            elif x>min(ind1,ind2):
                new_treeset.node_assigns[i] -= 1
                if x>max(ind1,ind2):
                    new_treeset.node_assigns[i] -= 1

        return new_treeset

    def number_of_leaves(self):
        leaves = 0
        for value in self.values:
            tree = value['tree']
            leaves += len([x for x in range(len(tree)) if x not in tree])
        return leaves    

class prufer_treeset():
    """ The class for prufer treesets, for generating all labeled trees using prufer sequence.
    this is treeset with following exceptions:
    - it has a defined set of leaves
    - in each step a root in the treeset is joined as the child to the root of another
    - this way we can have faster evaluations in exchange for memory
    - integrations inside the class
    - rc_distibs hold the convolution of distribs for all childs of a root at each time"""

    def __init__(self,length,leaves,distribs,priors):
        """ length is the number of subclones, leaves are the indices of leaf nodes and 
        distribs is the list of distributios for each subclone"""
        self.length = length
        self.values= []
        self.node_assigns = [y for y in range(length)]
        self.uleaves = set(leaves)      # unattached leaves
        self.remnodes = set([x for x in range(self.length) if x not in leaves]) # set of remaining nodes
        self.spent_nodes = set()
        self.rem_choices = self.length - 1  #number of remaining choices
        self.distribs = distribs
        self.priors = priors
        self.M = distribs.shape[1]
        self.prufer_seq = []
        for x in range(length):
            self.values +=  [{'tree':[-1],'nodes':[x],'root':x,'rc_distrib': None,\
            'r_distrib': self.distribs[x,:,:],'sum_distrib':sum([logsumexp(self.distribs[x,m,:]) for m in range(self.M)])}]
    def __repr__(self):
        """  represenation of the class"""
        return '\n'.join([get_ascii_tree(x['tree'],x['nodes']) for x in self.values])+'\n_________________\n'

    def join_root_to_node(self, rnode, node, subtree_lookup_dict):
        """ joins a treeset as a child of another
        recommendations: use better subtree_key"""
        new_treeset = cp.deepcopy(self)
        ind1 = new_treeset.node_assigns[node]
        ind2 = new_treeset.node_assigns[rnode]
        tree1 = new_treeset.values[ind1]
        tree2 = new_treeset.values[ind2]
        if not tree2['tree'][tree2['nodes'].index(rnode)] == -1:
            print('Error: the root argumnet must be a root in the treeset')
            return
        if not tree1['tree'][tree1['nodes'].index(node)] == -1:
            print('Error: the root argumnet must be a root in the treeset')
            return
        if ind1 ==ind2 :
            print('Error: select two distict trees')
            return
        tree = [x+len(tree1['tree']) if not x==-1 else tree1['nodes'].index(node) for x in tree2['tree']]
        tree = tree1['tree']+tree
        nodes = tree1['nodes']+tree2['nodes']
        root = tree1['root']
        new_treeset.values = [i for j, i in enumerate(new_treeset.values) if j not in [ind1,ind2]]
        new_treeset.prufer_seq +=[root]
        for i,x  in enumerate(new_treeset.node_assigns):
            if x in [ind1,ind2]:
                new_treeset.node_assigns[i] = len(new_treeset.values)
            elif x>min(ind1,ind2):
                new_treeset.node_assigns[i] -= 1
                if x>max(ind1,ind2):
                    new_treeset.node_assigns[i] -= 1
        new_treeset.uleaves.remove(rnode)
        new_treeset.spent_nodes.add(rnode)
        new_treeset.rem_choices -= 1
        tree_prufer = [x for x in new_treeset.prufer_seq if new_treeset.node_assigns[x]==len(new_treeset.values)]
        subtree_key = tuple(sorted(nodes)+[-100]+tree_prufer)

        if subtree_key in subtree_lookup_dict:
            new_treeset.values += [subtree_lookup_dict[subtree_key]]
            # print("tree found in dict")
            return new_treeset

        if tree1['rc_distrib'] is None:
            new_rc_distrib = np.array([trap_int(tree2['r_distrib'][m,:]) for m in range(self.M)])
        else:
            new_rc_distrib = np.array([my_convolve(tree2['r_distrib'][m,:],tree1['rc_distrib'][m,:])\
            for m in range(self.M)])
        
        new_distrib = new_rc_distrib + self.distribs[root,:,:]
        new_sum_distrib = sum([logsumexp(new_distrib[m,:]) for m in range(self.M)])
        new_value = {'tree':tree,'nodes':nodes,'root':root, \
        'r_distrib': new_distrib, 'rc_distrib': new_rc_distrib, 'sum_distrib':new_sum_distrib}
        new_treeset.values += [new_value]
        subtree_lookup_dict[subtree_key] = new_value
        return new_treeset

    def spawn(self, current_score, subtree_lookup_dict):
        """ performs the branch and bound algorithm using the current score
        output is a list of prufer_treesets"""
        child = min(self.uleaves)
        result = []
        for parent in self.remnodes:
            new_treeset = self.join_root_to_node(child, parent, subtree_lookup_dict)
            if sum([x['sum_distrib'] for x in new_treeset.values]) > current_score:
                print('score =', sum([x['sum_distrib'] for x in new_treeset.values]), 'current score=', current_score)
                # print(self.__repr__())
                if (not new_treeset.uleaves) or (new_treeset.rem_choices < len(new_treeset.remnodes)):
                    new_treeset.remnodes.remove(parent)
                    new_treeset.uleaves.add(parent)
                    result += [new_treeset]
                else:
                    new_treeset2 = cp.deepcopy(new_treeset)
                    new_treeset.remnodes.remove(parent)
                    new_treeset.uleaves.add(parent)
                    result += [new_treeset, new_treeset2]
        return result

    def spawn_all_the_way(self, current_score, subtree_lookup_dict):
        """ spawnes (connects two trees in treeset) untill the treeset has just one tree
        in each step checks whether the maximum reachable score is not less than current score"""
        rem_choices = self.rem_choices
        trees = [self]
        while rem_choices>0:
            next_trees = []
            for tree in trees:
                next_trees += tree.spawn(current_score, subtree_lookup_dict)
            trees = next_trees
            if trees:
                rem_choices = trees[0].rem_choices
            else:
                break
        return trees


def trap_int(xs):
    x = list(xs)
    for i in range(1,len(x)):
        x[i] = np.logaddexp(x[i-1],x[i])
    return np.array(x) - np.log(len(x)) 

def my_convolve(a,b):
    l=len(a)
    if not l == len(b):
        print("vector must be of same length for this function")
        return
    return np.array([logsumexp(b[:x+1]+a[x::-1]) for x in range(l)]) - np.log(l)

def eval_int(tree,node,distribs,priors):

    childs =  get_childs(tree,node)

    if len(childs) == 0:
        return my_convolve(distribs[node], priors[node])
    else:
        return distribs[node] + functools.reduce(lambda x,y:my_convolve(x,y),[priors[node]]+[eval_int(tree,x,distribs,priors) for x in childs])



def tree_integrate_single_sample(tree, distribs, priors):
    root = [i for i,x in enumerate(tree) if x == -1][0]
    return trap_int(eval_int(tree,root,distribs, priors))[-1]


def tree_integrate_multisample(tree, distribs, priors):
    K,M,_ = distribs.shape
    score = []
    for m in range(M):
        distrib = [distribs[x,m] for x in range(K)]
        prior = [priors[x,m] for x in range(K)]
        score += [tree_integrate_single_sample(tree,distrib,prior)]
    return np.array(score)

def part_func(n,kmax):
    T=[np.ones(min(k,kmax)) for k in range(1,n+1)]
    if n < 4:
        return T[n-1]
    m = 4
    while (m<n+1):
        for k in range(1,min(m,kmax)):
            T[m-1][k] = T[m-2][k-1]
            if m-1-k > k:
                 T[m-1][k] += T[m-2-k][k]
        m = m+1
    return T[n-1]

def solve4(B,am,bm,er,profile):
    #print profile
    n = len(B)
    M = am.shape[1]
    res = []
    value=0
    for m in range(M):
        Am = am[:,m].transpose()
        x = cvx.Variable(n)
        Bm = bm[:,m].transpose()
        obj2 = cvx.Maximize(cvx.sum_entries(Am*cvx.log(B*x/2.*(1-er)+(1-B*x/2.)*er)+Bm*cvx.log((1-B*x/2.)*(1-er)+B*x/2.*er)))
        constraints = [x>=np.zeros(n), B*x >= np.zeros(n), B*x <= np.ones(n), x<= profile[m]]
        prob = cvx.Problem(obj2, constraints)
        prob.solve(verbose=False)
        res +=[x.value]
        value += prob.value
    return {'s':np.array(res).transpose()[0],'value':value}

def get_max_values_r2(regular_tree, As, Bs, err):
    B = regular_tree['Bmatrix']
    K,M = As.shape
    result = solve4(B,As,Bs,err,[np.ones(K) for _ in range(M)])
    regular_tree['clone_proportions'] = result['s']
    return regular_tree['clone_proportions']