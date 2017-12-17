
"""
@author: Hosein Toosi
"""

import sys
import numpy as np
import copy as cp
import pandas as pd
import itertools
from scipy.stats import beta
from scipy.optimize import fsolve
from scipy.special import lambertw
from scipy.misc import comb, logsumexp
from scipy.special import gammaln, betainc
import re
import cvxpy as cvx
from sklearn.cluster import KMeans
import pickle
# import matplotlib.pyplot as plt
import  os
from math import factorial



#os.chdir(os.path.dirname(sys.argv[1]))
class clustering:

    def __init__(self,variant_reads, normal_reads, assignments):
        # am and bmm are mutationsXsamples matrices for variant and normal reads
        self.am = np.array(variant_reads)
        self.bm = np.array(normal_reads)
        #0 based integer vactor of assignment of variants to clusters
        self.assign = np.array(assignments).astype(int)
        
        #num mutations
        N = len(self.am)
        #num clusters
        K = max(assignments) + 1
        
        #### aggregate reads for each cluster + determine subclone order for heuristic search
        self.As =  np.array([np.sum(self.am[self.assign==x],0) for x in         range(max(self.assign)+1)])
        self.Bs =  np.array([np.sum(self.bm[self.assign==x],0) for x in         range(max(self.assign)+1)])
        self.order = sorted(np.unique(self.assign), self.compare_clusters)
        
        ### reorder subclones
        self.assign  = np.array([self.order.index(x) for x in self.assign])
        self.As =  np.array([np.sum(self.am[self.assign==x],0) for x in         range(max(self.assign)+1)])
        self.Bs =  np.array([np.sum(self.bm[self.assign==x],0) for x in         range(max(self.assign)+1)])
        
        self.means = (self.As +0.)/(self.As +self.Bs)
        # log maximum likelihood of data for this clustering
        self.maxlikelihood = np.sum(gammaln(self.am + self.bm+2.) -         gammaln(self.am + 1.)-gammaln(self.bm + 1.)) -         np.sum(gammaln(self.As +self.Bs +2)-gammaln(self.As + 1) -         gammaln(self.Bs + 1))
        # model prior for clustering



        self.cluster_prior =  log_conf_penalty(self.assign)

        #self.cluster_prior =log_conf_penalty2(self.assign,alphaa) - (K-1)* np.log(K)


        
    def compare_clusters(self, clustA, clustB):
        return int(np.sign(np.sum(tree_integrate_multisample(np.array([-1.,0.]),np.ones(2),self.As[[clustB,clustA],:],            self.Bs[[clustB,clustA],:],err,0.0)) - np.sum(tree_integrate_multisample(np.array([-1.,0.]),            np.ones(2),self.As[[clustA,clustB],:], self.Bs[[clustA,clustB],:],err,0.0))))
    
    def get_candidate_tree(self):
        tree = [-1]
        if len(self.order) ==1:
            self.trees =  [{'tree':tree, 'root':0}]
            return [{'tree':tree, 'root':0}]
        for n in range(1,len(self.order)):
            scores = [get_tree_score(tree+[x],range(n+1), self.As, self.Bs) for x in range(n)]
            tree += [np.argmax(scores)]
        self.trees =  [{'tree':tree, 'root':0}]
        return [{'tree':tree, 'root':0}]

    
    def get_trees2(self,candidate_tree):
        K,M = self.As.shape
        score = get_tree_score(candidate_tree['tree'],range(K), self.As, self.Bs)+ np.log(canonizeF(candidate_tree['tree'])['scre'])
        allts = []
        for rootnode in range(K):
            ts = [treeset(K)]
            #print rootnode
            nodes = [n for n in range(K) if not n ==rootnode]
            #print nodes
            for n in nodes:
                new_ts = []
                for t in ts:
                    ind = t.node_assigns[n]
                    allowed_parents = [p for p in range(K) if p not in t.values[ind]['nodes']]
                    for par in allowed_parents:
                        candidate_t = t.join_root_to_node(n,par)
                        candidate_score = np.sum([get_tree_score(x['tree'],x['nodes'], self.As, self.Bs)                                 for x in candidate_t.values])
                        if candidate_score +np.log(factorial(candidate_t.number_of_leaves())) >= score:
                            print 'scores=',candidate_score, score
                            print 'trees =', candidate_t.values, candidate_tree['tree']
                            new_ts += [candidate_t]
                ts = new_ts
            allts += ts
        self.trees = [{'tree':reorder_tree(t.values[0]['tree'],t.values[0]['nodes'],range(K)),                       'root':t.values[0]['nodes'][t.values[0]['tree'].index(-1)]} for t in allts                     if get_tree_score(t.values[0]['tree'],t.values[0]['nodes'], self.As, self.Bs) + np.log(                       canonizeF(t.values[0]['tree'])['scre']) >= score ] 
        return self.trees

    
    
    def analyse_trees_regular2(self,err):
        #get tree metrices, comute model posterior and ML values without the sparsity
        K,M = self.As.shape
        for tree in self.trees:
            B = get_matrix(tree['tree'])[0]
            A= np.linalg.inv(B)
            tree['Amatrix'] = A
            tree['Bmatrix'] = B
            tree['assign'] = self.assign
            score = np.sum(tree_integrate_multisample(tree = tree['tree'],                  b = np.ones(K), As = self.As, Bs = self.Bs , err = err, p= 0. ))
            get_max_values_r2(tree, self.As,self.Bs, err)
            tree['score'] = score +np.log(canonizeF(tree['tree'])['scre'])
            tree['totalscore']= tree['score']+self.cluster_prior +             self.maxlikelihood

    def analyse_trees_sparse2(self, prob_zero_cluster,err):
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
            integs = [tree_integrate_multisample(tree = tree['tree'],             b = x,As = self.As, Bs = self.Bs, err = err, p = p) for x in profls]
            tree['probability_profile'] = integs
            get_max_values2(tree, self.As, self.Bs, err)
            tree['score'] = np.prod( np.sum(integs,0)) * canonizeF(tree['tree'])['scre']
            tree['totalscore']= np.log(tree['score'])+self.cluster_prior +             self.maxlikelihood
            

#    def slice_sampler(self,score):
        
    


    


    
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


def log_conf_penalty(assign):
    partition = np.unique(assign, return_counts = True)[1]
    partition_config = np.unique(partition, return_counts = True)[1]
    return np.sum(gammaln(partition+1.)) - gammaln(np.sum(partition)+1) +     np.sum(gammaln(partition_config+1.)) - np.log(pf[len(partition)-1]) +     np.log(tree_numbers[len(partition)]) - (len(partition)-1)*np.log(len(partition))


def log_conf_penalty2(assign,alphaa):
    partition = np.unique(assign, return_counts = True)[1]
    return gammaln(alphaa)+np.sum(gammaln(partition+1.))+len(partition)*np.log(alphaa) - gammaln(np.sum(partition)+alphaa)


def random_assign(N,L):
    #randomly assign N objects to L groups
    diff = N-L
    weights = np.diff((np.round(diff*np.concatenate(([0.],np.sort(np.random.random(L-1)),[1.])),0)).astype(int))+1
    if np.sum(weights) < N:
        weights[-1] += 1
    assigns = np.concatenate([x*np.ones(weights[x]) for x in range(L)])
    np.random.shuffle(assigns)
    return assigns.astype('int')


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

def write_to_dot(destination_dot, tree, sample_names,colors=['darkseagreen1','deepskyblue','cornsilk','coral'
,'deeppink','gray','crimson','navy','salmon','yellow','violet','aquamarine']):
    props = tree['clone_proportions']
    with open(destination_dot,'w') as fi:
        fi.write('digraph G\n{\n    rankdir = UD;\n')
        for n in range(len(tree['tree'])):
            fi.write('node'+str(n)+'\n[\nshape = none\n ')
            fi.write('label = <<table border="0" cellspacing="0">\n<tr>')
            for s in range(props.shape[1]):
                if props[n,s]>0.001:
                    fi.write('<td  border="1" fixedsize="true" width="' +                     str(int(np.around(props[n,s]*300)))+'" height="20" bgcolor="'+colors[s]+'">')
                    if props[n,s]>0.05:
                        fi.write(str(int(np.around(props[n,s]*100)))+ '% </td>\n')
                    else:
                        fi.write('</td>\n')
            fi.write('</tr>\n</table>>\n]\n')
        fi.write('legend \n[\nshape = none\n ')
        fi.write('label = <<table border="0" cellspacing="0">\n')
        for s in range(props.shape[1]):
            fi.write('<tr><td  border="1" fixedsize="true" width="100" ' +             ' height="20" bgcolor="'+colors[s]+'">'+sample_names[s]+'</td></tr>\n')
        fi.write('\n</table>>\n]\n')

        for n in range(len(tree['tree'])):
            if tree['tree'][n] >= 0:
                fi.write('node'+str(int(tree['tree'][n]))+' -> node'+                str(n)+ '[label ="'+str(sum(tree['assign']==n))+'"]\n')

        fi.write('}')


def indexTobinary(inds,length):
    a = [0 for x in range(length)]
    for ind in inds:
        a[ind]=1
    return a


def get_child_nodes(prufer,node):
    return [x[not x.index(node)] for x in prufer.tree_repr if node in x]

def get_childs(tree,node):
    return [i for i, x in enumerate(tree) if x == node]

def get_matrix(otree):
    tree = list(otree)
    K = len(tree)
    desc = np.diag(np.ones(K))
    ans = [[k] for k in range(K)]
    node = tree.index(-1)
    frontier = [node]
    while len(frontier)>0:
        node = frontier[-1]
        childs = get_childs(tree,node)
        for x in childs:
            ans[x] +=ans[node]
        if len(childs)>0:
            for x in itertools.product(ans[node],childs):
                desc[x] = 1
        del frontier[-1]
        frontier += childs
    return desc,ans


def reorder_tree(tree, old_labels, new_labels):
    locs = [old_labels.index(x) for x in new_labels]
    newlocs = [new_labels.index(x) for x in old_labels]
    old_parents = [tree[x] for x in locs]
    new_tree = [newlocs[x] if not x==-1 else -1 for x in old_parents]
    return new_tree


def canonizeF(tree,node=-1):
    if node ==-1:
        node = list(tree).index(-1)
    children = get_childs(tree,node)
    if len(children)==0:
        return {'canstr':'[]','scre':1}
    else:
        a = [canonizeF(tree,x) for x in children]
        scores = [x['scre'] for x in a]
        #print scores
        canonstrs = [x['canstr'] for x in a]
        #print canonstrs
        score = np.prod([len(list(group)) for key, group in         itertools.groupby(canonstrs)])*np.prod(scores)
        canon_str = '['+''.join(x for x in sorted(canonstrs))+']'
        return {'canstr': canon_str,'scre': score}


def get_tree_score(tree,nodes,As,Bs):
    #print nodes
    return np.sum(tree_integrate_multisample(tree,np.ones(len(tree)),As[nodes,:],            Bs[nodes,:],err,0.0))


def get_tree(mat):
    rev = np.linalg.inv(mat)
    tree = -1*np.ones(len(mat))
    wheres = np.where(rev==-1)
    tree[wheres[1]] = wheres[0]
    return tree

def sim_data_readcounts(K,M,altrees,num_mut,coverage = 200, min_in_clust=2,error=0.003):
    ind = np.random.randint(len(alltrees[K]))
    true_tree = alltrees[K][ind]
    true_sm = np.array([np.random.dirichlet(np.ones(K+1))[:K] for m in range(M)])
    B=get_matrix(true_tree)[0]
    true_cm = np.array([B.dot(x) for x in true_sm])
    true_cm = true_cm.transpose()
    ps = np.random.dirichlet(np.ones(K))
    assign = np.random.choice(range(K),num_mut-min_in_clust*K,p=ps)
    assign = np.concatenate((np.repeat(np.arange(K),2),assign))
    covs = coverage *np.ones((num_mut,M)).astype(int)
    As = np.random.binomial(covs,true_cm[assign]*(0.5-error)+error)
    Bs = covs-As
    return {'As':As,'Bs':Bs , 'vaf':true_cm,'assign':assign,    'tree':true_tree,'s':np.array(true_sm).transpose(),    'Bmatrix': get_matrix(true_tree)[0],'Amatrix':    np.linalg.inv(get_matrix(true_tree)[0])}

def reorder_assignmens(assign):
    #reorders assignments so that the the fisrt element is assgned to first cluster
    #and so on
    index = np.unique(assign,return_index=True)[1]
    return np.argsort(np.argsort(index))[assign]

def unique_Kmeans_clustering(As,Bs,num_clusters,n_init):
    res = set()
    result = []
    data = (As+0.0)/(As+Bs)
    for count in range(n_init):
        assign = tuple(reorder_assignmens(KMeans(n_clusters=num_clusters,n_init=1).fit(data).labels_))
        if assign not in res:
            res.add(assign)
            result += [clustering(As,Bs,assign)]
    return result


def table_expression(remheight, remwidth, border, orient, tree, node, vafs, colors):
    bgcolors = ['white']+list(colors[1:])
    res =''
    if len(get_childs(tree,node))==0:
        res += r'<td FIXEDSIZE="TRUE" HEIGHT="'+str(remheight)+r'" WIDTH="'+str(remwidth)+r'" border="'+str(border)+r'" color="'+colors[node]+'" bgcolor="'+bgcolors[node]+'"></td>'
        res += '\n'
        return res
    else:
        res += r'<td><table  FIXEDSIZE="TRUE" HEIGHT="'+str(remheight)+r'" WIDTH="'+str(remwidth)+r'" bgcolor="'+bgcolors[node]+'" border="'+str(border)+r"""" cellborder='0' color=""" +'"'+colors[node]+r"""" cellspacing='0' cellpadding='0'>"""
        res += '\n'
        childs = get_childs(tree,node)
        proportion = vafs[node]- sum([vafs[child] for child in childs])
        proportions = np.array([0.0,proportion]+[vafs[child] for child in childs])
        proportions = proportions/sum(proportions)
        cumprops = np.cumsum(proportions)
        if orient == 'h':
            res += r'<tr>'
            effwidth = remwidth - border*(len(childs)+1)
            effheight = remheight - 2*border
            remwidths = np.diff(np.around(effwidth*cumprops,0)).astype(int)
            if remwidths[0]> 2*border:
                res +=r'<td FIXEDSIZE="TRUE" HEIGHT="'+str(effheight)+r'" WIDTH="'+str(remwidths[0])+r'" border="'+str(border)+r"""" color='white'></td>"""
                res +='\n'
            else:
                remwidths[1] += border
#             nodes_past =1

            for ind in range(len(childs)):
#                 print nodes_past
                res += table_expression(effheight, remwidths[ind+1], border, 'v', tree, childs[ind], vafs, colors)
#                 nodes_past = nodes_past+1+len(get_desc(tree, childs[ind]))
            res +='\n'+r'</tr>'
        if orient =='v':
            effheight = remheight - border*(len(childs)+1)
            effwidth = remwidth - 2*border
            remheights = np.diff(np.around(effheight*cumprops,0)).astype(int)
            if remheights[0]>2*border:
                res += r'<tr>'
                res +=r'<td FIXEDSIZE="TRUE" HEIGHT="'+str(remheights[0])+r'" WIDTH="'+str(effwidth)+r'" border="'+str(border)+r"""" color='white'></td>"""
                res +='\n</tr>'
            else:
                remheights[1] += border
#             nodes_past=1
            for ind in range(len(childs)):
                res += r'<tr>'
                res += table_expression(remheights[ind+1], effwidth, border, 'h', tree, childs[ind], vafs,colors)
#                 nodes_past = nodes_past+1+len(get_desc(tree, childs[ind]))
                res +=r'</tr>'+'\n'
            res +='\n'
        res += '</table></td>'
    return res
            

def put_sample_in_node(tree, vafs_tr, sample_names, sample_colors, cluster_colors, height=100, width=100, border=2, orient='v'):
        res=''
        for ind in range(len(sample_names)):
            otree =[-1]+list((np.array(tree)+1).astype(int))
            colors = tuple(["black"]+list(cluster_colors))
            vafs = [1.0]+list(vafs_tr[ind])
            existing_clones = [0]+[x for x in range(1,len(vafs)) if (vafs[x]>0.04)]
            absent_clones = [x for x in range(len(vafs)) if x not in existing_clones]
            for x in absent_clones:
                otree[x] = -5
            
            res +=   sample_names[ind]+' [\n    shape=plaintext\n    xlabel ="'+sample_names[ind]+            '"\n    label=<\n' + table_expression(height, width, border, 'v', otree, 0, vafs,colors)[4:-5]+            '\n  >];'
        return res
    
    
def get_childs(tree,node):
    return [x for x in range(len(tree)) if tree[x]==node]

def get_desc(tree,node):
    if len(get_childs(tree,node))==0:
        return []
    else:
        return get_childs(tree,node)+reduce(lambda x,y:x+y,[get_desc(tree,x) for x in get_childs(tree,node)])


def write_dot_files(dat,sample_colors, cluster_colors,treeoutfile,samplesoutfile):
    res =  put_sample_in_node(dat['tree'],dat['vaf'].transpose(),dat['sample_names'],sample_colors,cluster_colors,height=150,width=150)
    res = 'digraph {\n'+res+'}\n'
    with open(samplesoutfile,'w') as fi:
        fi.write(res)
    res=''
    clone_letters = [chr(x) for x in range(65,len(dat['tree'])+65)]
    for x in range(len(dat['tree'])):
        num_muts = len([y for y in dat['assign'] if y==x])
        res += chart_plot(clone_letters[x],cluster_colors[x],dat['vaf'].transpose()[:,x],dat['sample_names'], sample_colors, num_muts)
    res +='\n'
    for x in range(len(dat['tree'])):
        if not x==dat['root']:
            res += 'Subclone'+clone_letters[int(dat['tree'][x])]+' -> Subclone'+clone_letters[x]+'\n'
    res = 'digraph {\n'+res+'}\n'
    with open(treeoutfile,'w') as fi:
        fi.write(res)

def collapse_node(t,n):
    if not len(get_childs(t,n)) ==1:
        raise 'can not collapse node with other than one child'
        return
    child = int([x for x in range(len(t)) if t[x]==n][0])
    par = t[n]
    res = cp.deepcopy(t)
    res[child] = par
    x = range(n)+range(n+1,len(t))
    res = res[x]
    for x in range(len(res)):
        if res[x]>n:
            res[x] -=1
    return res

def collapse_assign(c,n,ch):
    res = cp.deepcopy(c)
    for x in range(len(res)):
        if res[x]==n:
            res[x] = ch
        if res[x] > n:
            res[x] -= 1
    return res

def remove_zeros(tree):
    sums = [sum(y) for y in tree['clone_proportions']]
    ns = [n for n in range(len(sums)) if sums[n]<0.001]
    nchilds = [len(get_childs(tree['tree'],n)) for n in ns]
    while 1 in nchilds:
        n = ns[nchilds.index(1)]
        child = int([x for x in range(len(tree['tree'])) if tree['tree'][x]==n][0])
        tree['tree'] = collapse_node(tree['tree'],n)
        tree['assign'] = collapse_assign(tree['assign'],n,child)
        tree['clone_proportions'] = np.delete(tree['clone_proportions'],n,0)
        sums = [sum(y) for y in tree['clone_proportions']]
        ns = [n for n in range(len(sums)) if sums[n]<0.001]
        nchilds = [len(get_childs(tree['tree'],n)) for n in ns]
    tree['Bmatrix'] = get_matrix(tree['tree'])[0]
    tree['Amatrix'] = np.linalg.inv(tree['Bmatrix'])
    return tree

def chart_plot(subclone_letter,color,percentages,sample_names, sample_colors, num_muts, bar_width=30,border=3):
    res = ''
    res += 'Subclone'+subclone_letter+' [\n      shape=plaintext\n      label=<<table FIXEDSIZE="TRUE" HEIGHT="170" WIDTH="'+str(bar_width*len(sample_names)+60)+    '" color="'+color+'" bgcolor="'+color+'" border="'+str(border)+'" cellborder="0" cellspacing="0">\n'
    res +='<tr><td rowspan ="3">'+subclone_letter+'-'+str(num_muts)+' </td>\n'
    for ind in range(len(sample_names)):
        percent = int(100*percentages[ind])
        if percent <100:
            res +='<td FIXEDSIZE="TRUE" HEIGHT="110" WIDTH="'+str(bar_width)+            '" cellpadding="0"><table cellspacing="0" cellpadding="0" cellborder="0" border="0"><tr><td '+            'FIXEDSIZE="TRUE" HEIGHT="'+str(100-int(100*percentages[ind]))+'" WIDTH="'+str(bar_width-2)+            '" > </td></tr>'
        if percent > 2:
            res +=' <tr><td FIXEDSIZE="TRUE" HEIGHT="'+str(int(100*percentages[ind]))+'" WIDTH="'+str(bar_width-2)+        '" bgcolor="'+sample_colors[ind]+'" > </td></tr>'
        res +='</table></td>'
    res +='</tr><tr>'
    for x in percentages:
        res += '<td>'+str(int(np.around(100*x,0)))+'</td>'
    res +='</tr><tr>'
    for x in sample_names:
        res += '<td>'+x+'</td>'
    res +='</tr></table> \n>]'

    return res

def write_text_output(data,destfile):
    with open(destfile,'w') as fi:
        for i,tree in enumerate(data):
            fi.write('Solution Number '+str(i+1)+'\n')
            fi.write('tree = ' + str(tree['tree']) + '\n')
            fi.write('logscore = '+ str(tree['totalscore']) + '\n')
            fi.write('ML Subclone Fractions = ' + str(tree['clone_proportions']) + '\n' )
            fi.write ('Mutations Subclone Membership = ' + str([chr(x+65) for x in tree['assign']]) + '\n')
            fi.write('----------------------------------------------------------------')
    return

import argparse

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
tree_numbers = np.array([1, 1, 1, 2, 4, 9, 20, 47, 108, 252, 582, 1345, 3086, 7072, 16121, 36667, 83099, 187885, 423610, 953033, 2139158, 4792126, 10714105, 23911794, 53273599, 118497834, 263164833, 583582570])

cluster_colors = ["plum", "lightyellow", "palegreen", "pink", "paleturquoise", "tan", "moccasin",                  "lightcyan3", "oldlace", "palevioletred", "olivedrab1", "lightgray", "steelblue1"]

sample_colors = ["red", "firebrick4", "brown4", "darkgoldenrod4" ,"deeppink1",                 "darkgreen", "cyan4", "darkorchid", "blue", "dimgray"]   


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
    clusterings += [clustering(X['As'],X['Bs'],assign)]
    raw_scores= np.array([x.cluster_prior + x.maxlikelihood for x in clusterings])
    print raw_scores
if np.any(np.diff(raw_scores)<0):
    candidate_numbers = raw_scores.argsort()[-3:]
else:
    candidate_numbers = [max_clusts]
print 'possible number of clusters:' , candidate_numbers
clusts = []
for x in candidate_numbers:
   clusts += list(unique_Kmeans_clustering(X['As'],X['Bs'],x,n_trees))
for x,clust in enumerate(clusts):
   print 'analysis clustering: ' +str(x+1) +' of '+str(len(clusts))
#   clust.get_t_normals(err)
#    clust.get_trees()
   clust.get_candidate_tree()
   clust.get_trees2(clust.trees[0])
   print len(clust.trees)
    
#   if sparsity <0 :
   clust.analyse_trees_regular2(err)
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
        temp_clust = clustering(X['As'],X['Bs'],trees[t]['assign'])
        temp_clust.trees = [trees[t]]
        temp_clust.analyse_trees_sparse2(sparsity,err)
        this_tree = temp_clust.trees[0]
        sparse_trees +=[cp.deepcopy(this_tree)]
        this_tree = remove_zeros(this_tree)
    this_tree = remove_zeros(this_tree)
    this_tree['vaf'] = this_tree['Bmatrix'].dot(this_tree['clone_proportions'])
    
    write_dot_files(this_tree,sample_colors,cluster_colors,'tree_number_'+str(t)+'_subclones.dot','tree_number_'+str(t)+'_samples.dot')
#    write_to_dot('tree_number_'+str(t)+'.dot',this_tree,list(sample_names))
    write_text_output([this_tree],'tree_number_'+str(t)+'.txt')

if (sparsity > 0. and sparsity < 1.):
    pickle.dump(sparse_trees,open('bamse.pickle','w'))
    write_text_output(sparse_trees,'bamse_output.txt')
else:
    pickle.dump(trees[:min(top_trees,len(trees))],open('bamse.pickle','w'))
    write_text_output(trees[:min(top_trees,len(trees))],'bamse_output.txt')

