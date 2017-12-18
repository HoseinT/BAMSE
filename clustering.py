#!/usr/bin/env python2
"""The clustering class is defined here,  it defines a clustering, the prior
and the error model for the clustering"""
from tree_tools import *
import numpy as np
import itertools
from scipy.special import gammaln
import copy as cp
from scipy.misc import comb, logsumexp
import cvxpy as cvx
from math import factorial

tree_numbers = np.array([1, 1, 1, 2, 4, 9, 20, 47, 108, 252, 582, 1345, 3086, 7072, 16121, 36667, 83099, 187885, 423610, 953033, 2139158, 4792126, 10714105, 23911794, 53273599, 118497834, 263164833, 583582570])

class clustering:

    def __init__(self, variant_reads, normal_reads, assignments,pf,err):
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
        self.As = np.array([np.sum(self.am[self.assign == x], 0) for x in range(max(self.assign)+1)])
        self.Bs = np.array([np.sum(self.bm[self.assign == x], 0) for x in range(max(self.assign)+1)])
        self.order = sorted(np.unique(self.assign), self.compare_clusters)

        ### reorder subclones
        self.assign = np.array([self.order.index(x) for x in self.assign])
        self.As = np.array([np.sum(self.am[self.assign == x], 0) for x in range(max(self.assign)+1)])
        self.Bs = np.array([np.sum(self.bm[self.assign == x], 0) for x in range(max(self.assign)+1)])

        self.means = (self.As +0.)/(self.As +self.Bs)
        # log maximum likelihood of data for this clustering
        self.maxlikelihood = np.sum(gammaln(self.am + self.bm+2.) - \
        gammaln(self.am + 1.) - gammaln(self.bm + 1.)) - np.sum(gammaln(self.As +self.Bs +2) \
        -gammaln(self.As + 1) - gammaln(self.Bs + 1))
        # model prior for clustering



        self.cluster_prior = log_conf_penalty(self.assign, self.pf)



    def compare_clusters(self, clustA, clustB):  ### add error
        """ With ISA constraints compares sublcones A and B to see which is statistically more
        probable to be the parent of the other"""
        return int(np.sign(np.sum(tree_integrate_multisample(np.array([-1., 0.]), \
        np.ones(2), self.As[[clustB, clustA], :], self.Bs[[clustB, clustA], :], self.err, 0.0))\
        - np.sum(tree_integrate_multisample(np.array([-1., 0.]), \
        np.ones(2), self.As[[clustA, clustB], :], self.Bs[[clustA, clustB], :], self.err, 0.0))))

    def get_candidate_tree(self):
        """ Get an initial Heuristic Tree"""
        tree = [-1]
        if len(self.order) == 1:
            self.trees = [{'tree':tree, 'root':0}]
            return [{'tree':tree, 'root':0}]
        for n in range(1, len(self.order)):
            scores = [get_tree_score(tree+[x], range(n+1), self.As, self.Bs, self.err) for x in range(n)]
            tree += [np.argmax(scores)]
        self.trees = [{'tree':tree, 'root':0}]
        return [{'tree':tree, 'root':0}]

    def get_trees2(self, candidate_tree):
        """ Get all the trees that have a larger score than candidate_tree"""
        K, M = self.As.shape
        score = get_tree_score(candidate_tree['tree'], range(K), self.As, self.Bs, self.err) + \
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
                        candidate_score = np.sum([get_tree_score(x['tree'], x['nodes'], self.As,\
                        self.Bs, self.err) for x in candidate_t.values])
                        if candidate_score + np.log(factorial(candidate_t.number_of_leaves()))\
                         >= score:
                            print 'scores=', candidate_score, score
                            print 'trees =', candidate_t.values, candidate_tree['tree']
                            new_ts += [candidate_t]
                ts = new_ts
            allts += ts
        self.trees = [{'tree':reorder_tree(t.values[0]['tree'], t.values[0]['nodes'], range(K)),\
        'root':t.values[0]['nodes'][t.values[0]['tree'].index(-1)]} for t in allts \
        if get_tree_score(t.values[0]['tree'], t.values[0]['nodes'], self.As, self.Bs, self.err) + \
        np.log(canonizeF(t.values[0]['tree'])['scre']) >= score]
        return self.trees

    def analyse_trees_regular2(self):
        """ get tree metrices, comute model posterior and ML values without the sparsity"""
        K, M = self.As.shape
        for tree in self.trees:
            B = get_matrix(tree['tree'])[0]
            A = np.linalg.inv(B)
            tree['Amatrix'] = A
            tree['Bmatrix'] = B
            tree['assign'] = self.assign
            score = np.sum(tree_integrate_multisample(tree = tree['tree'], b = np.ones(K), As = self.As, Bs = self.Bs , err = self.err, p= 0. ))
            get_max_values_r2(tree, self.As,self.Bs, self.err)
            tree['score'] = score +np.log(canonizeF(tree['tree'])['scre'])
            tree['totalscore']= tree['score']+self.cluster_prior +             self.maxlikelihood

    def analyse_trees_sparse2(self, prob_zero_cluster):
        #get tree metrices, comute model posterior and ML values with sparsity
        p = prob_zero_cluster
        K,M = self.As.shape
        for tree in self.trees:
            B = get_matrix(tree['tree'])[0]
            A= np.linalg.inv(B)
            tree['Amatrix'] = A
            tree['Bmatrix'] = B
            tree['assign'] = self.assign
            profls = np.array(list(itertools.product([0, 1], repeat=K)))
            integs = [tree_integrate_multisample(tree = tree['tree'], b = x,As = self.As, Bs = self.Bs, err = self.err, p = p) for x in profls]
            tree['probability_profile'] = integs
            get_max_values2(tree, self.As, self.Bs, self.err)
            tree['score'] = np.prod( np.sum(integs,0)) * canonizeF(tree['tree'])['scre']
            tree['totalscore']= np.log(tree['score'])+self.cluster_prior +             self.maxlikelihood

def log_conf_penalty(assign,pf):
    partition = np.unique(assign, return_counts = True)[1]
    partition_config = np.unique(partition, return_counts = True)[1]
    return np.sum(gammaln(partition+1.)) - gammaln(np.sum(partition)+1) +     np.sum(gammaln(partition_config+1.)) - np.log(pf[len(partition)-1]) +     np.log(tree_numbers[len(partition)]) - (len(partition)-1)*np.log(len(partition))


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
        self.node_assigns = range(length)

    def join_root_to_node(treeset,root,node):
        #### places root under node and returns a new treeset
        new_treeset = cp.deepcopy(treeset)
        ind1 = new_treeset.node_assigns[node]
        ind2 = new_treeset.node_assigns[root]
        tree1 = new_treeset.values[ind1]
        tree2 = new_treeset.values[ind2]
        if not tree2['tree'][tree2['nodes'].index(root)] == -1:
            print 'Error: the root argumnet must be a root in the treeset'
            return
        if ind1 ==ind2 :
            print 'Error: select two distict trees'
            return
        tree = [x+len(tree1['tree']) if not x==-1 else                 tree1['nodes'].index(node) for x in tree2['tree']]
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
    
# def distrib(a,b,err,length = 2048):
#     x = np.linspace(0,1,length)
#     y = beta.pdf(x*(0.5-err)+err,a+1,a+b+2)
#     return y/np.sum(y)*length


def distrib(a,b,err,length = 512):
    x = np.linspace(0,1,length)
    cf = x*(0.5-err)+err
    y = gammaln(a+b+1)-gammaln(a+1)-gammaln(b+1) + np.log(cf)*a+np.log(1-cf)*b
    return y


# def trap_int(xs):
#     #print max(xs)
#     y = np.zeros(len(xs))
#     if len(xs) == 0:
#         return y
#     y[0] = - float('inf')
#     if len(xs) == 1:
#         return y
#     for i in range(1,len(xs)):
#         y[i] = logsumexp([y[i-1],xs[i]-np.log(2),xs[i-1]-np.log(2)])
#     return y - np.log(len(xs)-1)


# def trap_int(xs, ref = 5):
#     scale = ref - max(xs)
#     x = xs + scale
#     return np.log(np.cumsum(np.exp(x))) - scale - np.log(len(x))

def trap_int(xs):
    x = list(xs)
    for i in range(1,len(x)):
        x[i] = np.logaddexp(x[i-1],x[i])
    return np.array(x) - np.log(len(x)) 

    
# def trap_int(x):
#     return (np.cumsum(x) - (x[0]+x)/2.0)/(len(x)-1)

def my_convolve(a,b):
    l=len(a)
    if not l == len(b):
        print "vector must be of same length for this function"
        return
    return np.array([logsumexp(b[:x+1]+a[x::-1]) for x in range(l)]) - np.log(l)

# def my_convolve(x,y):
#     res = (x[0]*y+y[0]*x)/2.0
#     return (np.convolve(x,y)[:len(x)] - res) / (len(x)-1)


def my_convolve2(x,y):
    res = (x[0]*y+y[0]*x)/2.0
#    print len(res)
    return (fftconvolve(x,y)[:len(x)] - res) / (len(x)-1)


def eval_int(tree,node,distribs,starred):

    childs =  get_childs(tree,node)

    if len(childs) == 0:
        return distribs[node]
    elif len(childs) ==1:
        if starred[childs[0]] == '*':
            return distribs[node] + eval_int(tree,childs[0],distribs,starred)
        else:
            return distribs[node] + trap_int(eval_int(tree,childs[0],distribs,starred))
    elif len(childs)>1:
        if starred[childs[0]] == '*':
            ch0= eval_int(tree,childs[0],distribs,starred)
        else:
            ch0= trap_int(eval_int(tree,childs[0],distribs,starred))
        return distribs[node] + reduce(lambda x,y:my_convolve(x,y),[ch0]+[eval_int(tree,x,distribs,starred) for x         in childs[1:]])


def tree_integrate_single_sample(tree,b,distribs,p):
    s = 0
    if not p==0:
        s = np.log(p)*(len(b)-sum(b)) + np.log(1-p)*sum(b)
    starred = ['-' for _ in range(len(tree))]
    culled_tree = cp.deepcopy(tree)
    for node in np.where(b==0)[0]:
        if len(get_childs(tree,node))==0:
            s += distribs[node][0]
            culled_tree[node] = 10000
        else:
            starred[get_childs(tree,node)[0]] = '*'
    root = list(tree).index(-1)
    return s - gammaln(np.sum(b)+2) + trap_int(eval_int(culled_tree,root,distribs,starred))[-1]


def tree_integrate_multisample(tree,b,As,Bs,err,p):
    K,M = As.shape
    score = []
    for m in range(M):
        distribs = [distrib(As[x,m],Bs[x,m],err) for x in range(K)]
        score += [tree_integrate_single_sample(tree,b,distribs,p)]
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


def get_max_values2(sparse_tree,As,Bs,err):
    prob_profile, B = sparse_tree['probability_profile'],    sparse_tree['Bmatrix']
    K,M = As.shape
    max_inds = np.argmax(prob_profile,0)
    profls = np.array(list(itertools.product([0, 1], repeat=K)))
    max_profs = profls[max_inds]
    result = solve4(B,As,Bs,err,max_profs)
    sparse_tree['clone_proportions'] = result['s']
    return sparse_tree['clone_proportions']

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

def get_tree_score(tree,nodes,As,Bs,err):
    #print nodes
    return np.sum(tree_integrate_multisample(tree,np.ones(len(tree)),As[nodes,:], Bs[nodes,:],err,0.0))
