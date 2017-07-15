# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 19:07:53 2017

@author: hosein
"""
alphaa = 1.
import sys
import numpy as np
#from itertools import groupby
from scipy.stats import mvn
#from sklearn import mixture
import copy as cp
import pandas as pd
import itertools
from scipy.stats import norm,truncnorm
from scipy.special import betainc
from scipy.optimize import fsolve
from scipy.special import lambertw
from scipy.misc import comb
from scipy.special import gammaln
from scipy.stats import multivariate_normal as mvnorm
import re
import cvxpy as cvx
from sklearn.cluster import KMeans
import pickle

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
        #aggregate reads for each cluster
        self.As =  np.array([np.sum(self.am[self.assign==x],0) for x in \
        range(max(self.assign)+1)])
        self.Bs =  np.array([np.sum(self.bm[self.assign==x],0) for x in \
        range(max(self.assign)+1)])
        self.means = (self.As +0.)/(self.As +self.Bs)
        # log maximum likelihood of data for this clustering
        self.maxlikelihood = np.sum(gammaln(self.am + self.bm+1.) - \
        gammaln(self.am + 1.)-gammaln(self.bm + 1.)) - \
        np.sum(gammaln(self.As +self.Bs +1)-gammaln(self.As + 1) - \
        gammaln(self.Bs + 1))
        # model prior for clustering

        

        self.cluster_prior = -np.log(stril_appx(N,K)) + \
        log_conf_penalty(self.assign) - (K-1)* np.log(K)
        
        #self.cluster_prior =log_conf_penalty2(self.assign,alphaa) - (K-1)* np.log(K)
        
    def get_t_normals(self,err):
        #err is one third of sequencing error rate
        (K,M) = self.As.shape
        self.mus = np.zeros((K,M))
        self.sigs = np.zeros((K,M))
        self.coeffs = np.zeros((K,M))
        for k in range(K):
            for m in range(M):
                self.mus[k,m] , self.sigs[k,m] = t_beta_to_t_normal(self.As[k,m]+1., \
                self.Bs[k,m]+1.,err)
                self.coeffs[k,m] = truncnorm.pdf(nearest(self.mus[k,m]),a=(0.-self.mus[k,m])/self.sigs[k,m],\
                b=(1.-self.mus[k,m])/self.sigs[k,m], loc = self.mus[k,m] , scale = self.sigs[k,m]) /\
                norm.pdf(nearest(self.mus[k,m]), loc = self.mus[k,m] , scale = self.sigs[k,m])
                
        
    def get_trees(self):

        margin=0
        treeset =get_all_trees(self.means,margin)
        while (len(treeset)<1):
            margin -=0.01
            treeset =get_all_trees(self.means,margin)
        self.trees = treeset
        
        
    def analyse_trees_regular(self):
        #without the sparsity   
        K,M = self.As.shape
        for tree in self.trees:
            B = get_matrix(tree['tree'])[0]        
            A= np.linalg.inv(B)
            tree['Amatrix'] = A
            tree['Bmatrix'] = B
            tree['mus'] = self.mus
            tree['sigs'] = self.sigs
            tree['assign'] = self.assign
            score = 1            
            for m in range(M):
                mu = np.dot(A,self.mus[:,m])
                covar = A.dot(np.diag(self.sigs[:,m]**2)).dot(A.transpose())
#                coeffs = np.prod([1./(norm.cdf((1-self.mus[x,m])/self.sigs[x,m]) - \
#                norm.cdf((0.-self.mus[x,m])/self.sigs[x,m]))  for x in range(K)])
           
                score *= mvn.mvnun(np.zeros(K),np.ones(K),mu,covar)[0]*np.prod(self.coeffs[:,m])        
            get_max_values_r2(tree, self.As,self.Bs, err)
            tree['score'] = score*canonizeF(tree['tree'])['scre']
            tree['totalscore']= np.log(tree['score'])+self.cluster_prior + \
            self.maxlikelihood



    def analyse_trees_sparse(self, prob_zero_cluster):
        p = prob_zero_cluster
        K,M = self.As.shape
        for tree in self.trees:
            B = get_matrix(tree['tree'])[0]        
            A= np.linalg.inv(B)
            tree['mus'] = self.mus
            tree['sigs'] = self.sigs
            tree['Amatrix'] = A
            tree['Bmatrix'] = B
            tree['assign'] = self.assign            
            integs = integ_all_profiles(self.mus,self.sigs,self.coeffs,A,p)
            tree['probability_profile'] = integs
            get_max_values2(tree, self.As, self.Bs, err)
            tree['score'] = np.prod( np.sum(integs)) * canonizeF(tree['tree'])['scre']
            tree['totalscore']= np.log(tree['score'])+self.cluster_prior + \
            self.maxlikelihood


def nearest(x):
    if x<= 0:
        return 0.
    if x<=1:
         return x
    if x>1:
        return 1.

def stril_appx(n,k):
    # Approximation to Stirling numbers of second kind
    v = (n+0.0)/k
    G = np.real(-1*lambertw(-v*np.exp(-v)))
    return np.sqrt(n-k)/(np.sqrt(n*(1-G)) \
    *G**k*(v-G)**(n-k))*((n-k)/np.exp(1))**(n-k)*comb(n,k)

def log_conf_penalty(assign):
    partition = np.unique(assign, return_counts = True)[1]
    return np.sum(gammaln(partition+1.)) - gammaln(np.sum(partition)+1)


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


def clusterpenalty(am,bm,assign):
    a= np.array([np.sum(am[assign==x],0) for x in  range(max(assign)+1)])
    b= np.array([np.sum(bm[assign==x],0) for x in  range(max(assign)+1)])
    return np.sum(gammaln(am+bm+1.)-gammaln(am+1.)-gammaln(bm+1.))-np.sum(gammaln(a+b+1)-gammaln(a+1)-gammaln(b+1))
    


def trunc_mean(mu,sig):
    #mean of a truncated normal
    return float(truncnorm.stats(a=(0.-mu)/sig, b=(1.-mu)/sig, loc = mu , scale = sig, moments='m'))

def trunc_var(mu,sig):
    #variance of a truncated normal
    return float(truncnorm.stats(a=(0.-mu)/sig, b=(1.-mu)/sig, loc = mu , scale = sig, moments='v'))

def equations(p,M,V):
    mu, sig = p
    return (trunc_mean(mu,sig)-M, trunc_var(mu,sig)-V)


def t_beta_to_t_normal(a,b,err):
    #approximate the PDF with a truncated normal distribution
    A=0.5-err
    M  =  1/A * (betainc(a+1,b,0.5)-betainc(a+1,b,err))/ (betainc(a,b,0.5)-betainc(a,b,err)) * (a+0.0)/(a+b) - err/A
    #V = sum((x-M)**2*y1)/len(x)
    V = 1/A**2 * (betainc(a+2,b,0.5)-betainc(a+2,b,err))/ (betainc(a,b,0.5)-betainc(a,b,err)) \
    * (a+0.0)* (a+1.)/(a+b)/(a+b+1.) - 2*err/A**2 *  (betainc(a+1,b,0.5)-betainc(a+1,b,err))/ (\
    betainc(a,b,0.5)-betainc(a,b,err)) * (a+0.0)/(a+b) + err**2/A**2 - M**2
    mu = ((a-1.)/(a+b-2.)-err)/(0.5-err)
    mu, sig =  fsolve(equations, (((a-1.0)/(a+b-2.0))/A-err, np.sqrt(V)), args = (M,V))
    return mu , sig





def integ_all_profiles(mus,sigs,coeffs,A,p):
    K = len(A)
    profls = np.array(list(itertools.product([0, 1], repeat=K)))
    return [integ_mvn(mus,sigs,coeffs,A,upper,p) for upper in profls]



def integ_mvn(mus,sigs,coeffs,A,upper,p):
    knowns = np.where(upper==0)[0]
    unknowns = np.where(upper==1)[0]
    kn = len(knowns)
    uk = len(unknowns)
#    K,M =mus.shape
#    coeffs = np.zeros((K,M))
#    for k in range(K):
#        for m in range(M):
#                coeffs[k,m] = truncnorm.pdf(0,a=(0.-mus[k,m])/sigs[k,m],\
#                b=(1.-mus[k,m])/sigs[k,m], loc =mus[k,m] , scale = sigs[k,m]) /\
#                norm.pdf(0, loc = mus[k,m] , scale = sigs[k,m])
#            
#        
##    coeffs = 1./(norm.cdf((1.-mus)/sigs) - \
##   norm.cdf((0.-mus)/sigs))
#    print coeffs
    bin_coeff = (1-p)**(sum(upper))* p ** (len(upper)-sum(upper))
    mu = A.dot(mus)
    if kn==0:
        return [mvn.mvnun(np.zeros(uk),np.ones(uk), mu[:,m], \
        A.dot(np.diag(sigs[:,m]**2)).dot(A.transpose()))[0] \
            *np.prod(coeffs[:,m])* bin_coeff for m in range(mu.shape[1])]
    if uk==0:
        return [mvnorm.pdf(np.zeros(kn), mu[:,m], \
        A.dot(np.diag(sigs[:,m]**2)).dot(A.transpose())) \
            *np.prod(coeffs[:,m])* bin_coeff for m in range(mu.shape[1])]
    else:
        score = []
        for m in range(mu.shape[1]):
            covar = A.dot(np.diag(sigs[:,m]**2)).dot(A.transpose())
            sig11 =  covar[[[x] for x in unknowns],unknowns]
            sig22 =  covar[[[x] for x in knowns],knowns]
            sig12 = covar[[[x] for x in unknowns],knowns]
            sig21 = covar[[[x] for x in knowns],unknowns]
            mu1 = mu[unknowns,m]
            mu2 = mu[knowns,m]
            mu_c = mu1 - sig12.dot(np.linalg.inv(sig22)).dot(np.zeros(kn)-mu2)
            sig_c = sig11 - sig12.dot(np.linalg.inv(sig22)).dot(sig21)
            score += [mvn.mvnun(np.zeros(uk),np.ones(uk),mu_c,sig_c)[0] \
            * mvnorm.pdf(np.zeros(kn),mu2,sig22) *np.prod(coeffs[:,m])* bin_coeff]
        return score 




def get_max_values(sparse_tree):
    prob_profile, A , mus , sigs = sparse_tree['probability_profile'],\
    sparse_tree['Amatrix'],sparse_tree['mus'],\
    sparse_tree['sigs']
    K = len(A)
    max_inds = np.argmax(prob_profile,0)
    profls = np.array(list(itertools.product([0, 1], repeat=K)))
    max_profs = profls[max_inds]
    mu = A.dot(mus)
    for m in range(mu.shape[1]):
        upper = max_profs[m]     
        knowns = np.where(upper==0)[0]
        unknowns = np.where(upper==1)[0]
        kn = len(knowns)
        uk = len(unknowns)    
        if kn==0:
            continue
        if uk==0:
            mu[:,m] = np.zeros(K)
        else:
            covar = A.dot(np.diag(sigs[:,m]**2)).dot(A.transpose())
            sig22 =  covar[[[x] for x in knowns],knowns]
            sig12 = covar[[[x] for x in unknowns],knowns]
            mu1 = mu[unknowns,m]
            mu2 = mu[knowns,m]
            mu[unknowns,m] = mu1 - sig12.dot(np.linalg.inv(sig22)).dot(np.zeros(kn)-mu2)
            mu[knowns,m] = np.zeros(kn)
    sparse_tree['clone_proportions'] = mu
    return mu




def get_max_values2(sparse_tree,As,Bs,err):
    prob_profile, B = sparse_tree['probability_profile'],\
    sparse_tree['Bmatrix']
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
#        print Am
        x = cvx.Variable(n)
        Bm = bm[:,m].transpose()
        obj2 = cvx.Maximize(cvx.sum_entries(Am*cvx.log(B*x/2.*(1-er)+(1-B*x/2.)*er)+Bm*cvx.log((1-B*x/2.)*(1-er)+B*x/2.*er)))    
        constraints = [x>=np.zeros(n), B*x >= np.zeros(n), B*x <= np.ones(n), x<= profile[m]]
        prob = cvx.Problem(obj2, constraints)
        prob.solve(verbose=False)
        res +=[x.value]
        value += prob.value
    return {'s':np.array(res).transpose()[0],'value':value}



def get_max_values_r(regular_tree):
    A , mus , sigs = regular_tree['Amatrix'],regular_tree['mus'],\
    regular_tree['sigs']
    K = len(A)
    mu = A.dot(mus)
    for m in range(mu.shape[1]):
        upper = np.where(mu[:,m] <= 0, np.zeros(K),np.ones(K))     
        knowns = np.where(upper==0)[0]
        unknowns = np.where(upper==1)[0]
        kn = len(knowns)
        uk = len(unknowns)    
        if kn==0:
            continue
        if uk==0:
            mu[:,m] = np.zeros(K)
        else:
            covar = A.dot(np.diag(sigs[:,m]**2)).dot(A.transpose())
            sig22 =  covar[[[x] for x in knowns],knowns]
            sig12 = covar[[[x] for x in unknowns],knowns]
            mu1 = mu[unknowns,m]
            mu2 = mu[knowns,m]
            mu[unknowns,m] = mu1 - sig12.dot(np.linalg.inv(sig22)).dot(np.zeros(kn)-mu2)
            mu[knowns,m] = np.zeros(kn)
    regular_tree['clone_proportions'] = mu
    return mu



def get_max_values_r2(regular_tree, As, Bs, err):
    B = regular_tree['Bmatrix']
    K,M = As.shape
    result = solve4(B,As,Bs,err,[np.ones(K) for _ in range(M)])
    regular_tree['clone_proportions'] = result['s']
    return regular_tree['clone_proportions']



    
#label= "c'+str(n)+'"\n
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
                    fi.write('<td  border="1" fixedsize="true" width="' + \
                    str(int(np.around(props[n,s]*300)))+'" height="20" bgcolor="'+colors[s]+'">')
                    if props[n,s]>0.05:
                        fi.write(str(int(np.around(props[n,s]*100)))+ '% </td>\n')
                    else:
                        fi.write('</td>\n')
            fi.write('</tr>\n</table>>\n]\n')
        fi.write('legend \n[\nshape = none\n ')
        fi.write('label = <<table border="0" cellspacing="0">\n')
        for s in range(props.shape[1]):
            if props[n,s]>0:
                fi.write('<tr><td  border="1" fixedsize="true" width="100" ' + \
                ' height="20" bgcolor="'+colors[s]+'">'+sample_names[s]+'</td></tr>\n')
        fi.write('\n</table>>\n]\n')

        for n in range(len(tree['tree'])):
            if tree['tree'][n] >= 0:
                fi.write('node'+str(int(tree['tree'][n]))+' -> node'+\
                str(n)+ '[label ="'+str(sum(tree['assign']==n))+'"]\n')
        
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
        score = np.prod([len(list(group)) for key, group in \
        itertools.groupby(canonstrs)])*np.prod(scores)
        canon_str = '['+''.join(x for x in sorted(canonstrs))+']'
        return {'canstr': canon_str,'scre': score}


def sum_min_dif(x,y):
    return np.sum(np.minimum(np.zeros(len(x)),x-y))

def get_all_trees(values,margin):
    K=len(values)
    M= len(values[0])
    result=[]
    for x in range(K):
        sticks = [-100*np.ones(M) for k in range(K)]
        sticks[x] = values[x]
        tree = -50*np.ones(K)
        tree[x]= -1
        result += [{'sticks':sticks, 'tree':tree, 'root':x, 'margin':margin}]
#    print result
    for count in range(1,K):
        newresult = []
        for res in result:
            node = (res['root']+count)%K
            for par in range(K):
                if sum_min_dif(res['sticks'][par],values[node])>res['margin']:
                    newres = cp.deepcopy(res)
                    newres['margin'] -= min(0,sum_min_dif(res['sticks'][par],values[node]))
                    newres['sticks'][par] -= values[node]
                    newres['sticks'][node] = values[node]
                    newres['tree'][node] = par
                    newresult +=[newres]
            for par in range(K):
                if sum_min_dif(values[par],values[node])>res['margin']:
                    childs =[x for x in range(K) if res['tree'][x]==par and \
                    sum_min_dif(values[node],values[x])>res['margin']]
                    childs_all = [x for x in range(K) if res['tree'][x]==par]
                    for ind in range(1,len(childs)+1):
                        for subset in itertools.combinations(childs, ind):
                            complement = [x for x in childs_all if x not in subset]
                            if sum_min_dif(values[node],sum([values[x] for x \
                            in subset]))>res['margin'] and sum_min_dif(values[par],sum([values[x] for x \
                            in complement])+values[node])>res['margin']- \
                            min(0,sum_min_dif(values[node],sum([values[x] for x \
                            in subset]))):
                                newres = cp.deepcopy(res)
                                for x in subset:
                                    newres['tree'][x]=node
                                newres['tree'][node]=par
                                newres['sticks'][node] = values[node]-sum([values[x] for x \
                            in subset])
                                newres['sticks'][par] = values[par]-sum([values[x] for x \
                            in complement])-values[node] 
                                newres['margin'] = res['margin']-min(0,sum_min_dif(values[par],sum([values[x] for x \
                            in complement])+values[node])+sum_min_dif(values[node],sum([values[x] for x \
                            in subset])))                                
                                newresult +=[newres]
        result = newresult            
    return result


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
    return {'As':As,'Bs':Bs , 'vaf':true_cm,'assign':assign,\
    'tree':true_tree,'s':np.array(true_sm).transpose(),\
    'Bmatrix': get_matrix(true_tree)[0],'Amatrix':\
    np.linalg.inv(get_matrix(true_tree)[0])}

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


#import pickle
#with open('AllTrees.txt') as f:
#    [alltrees, treemakeup] = pickle.load(f)
#
#
#K=8
#X = sim_data_readcounts(K,3,alltrees,100,coverage=200)
#print K


inputfile = sys.argv[1]
err = float(sys.argv[2])
sparsity = float(sys.argv[3])
n_trees = int(sys.argv[4])
max_clusts= int(sys.argv[5])
top_trees = int(sys.argv[6])
print max_clusts
table_data = pd.read_table(inputfile)
ref_cols=[re.match('(.*)?.ref',x) for x in table_data.columns.values]
var_cols=[re.match('(.*)?.var',x) for x in table_data.columns.values]
ref_names =[x.string.replace('.ref','') for x in filter(None,ref_cols)]
var_names =[x.string.replace('.var','') for x in filter(None,var_cols)]
sample_names = set.intersection(set(ref_names),set(var_names))
sample_names = list(sample_names)
sample_names.sort()
X={'As':table_data[[x+'.var' for x in sample_names]].as_matrix(),\
'Bs':table_data[[x+'.ref' for x in sample_names]].as_matrix()}
clusterings = []
for K in range(1,max_clusts):
    Y = (X['As']+0.0)/(X['Bs']+X['As'])
    score = -1*float("inf")
    assign = KMeans(n_clusters=K,n_init=n_trees).fit(Y).labels_
    clusterings += [clustering(X['As'],X['Bs'],assign)]
    raw_scores= np.array([x.cluster_prior + x.maxlikelihood for x in clusterings])
    print raw_scores
    candidate_numbers = raw_scores.argsort()[-2:]
print 'possible number of clusters:' , candidate_numbers
clusts = []
for x in candidate_numbers:
    clusts += list(unique_Kmeans_clustering(X['As'],X['Bs'],x,n_trees))
for x,clust in enumerate(clusts):
    print 'analysis clustering: ' +str(x+1) +' of '+str(len(clusts))
    clust.get_t_normals(err)
    clust.get_trees()
    print clust.trees[0]['tree']
    if sparsity <0 :
        clust.analyse_trees_regular()
    else:
        clust.analyse_trees_sparse(sparsity)
trees = []
for clust in clusts:
    trees += clust.trees
ordered = np.argsort([-1*x['totalscore'] for x in  trees])
trees = [trees[x] for x in ordered]
for x in trees:
    x['sample_names']=list(sample_names)
for t in range(min(top_trees,len(trees))):
    write_to_dot('tree_number_'+str(t)+'.dot',trees[t],list(sample_names))
    with open('tree_number_'+str(t)+'.txt','w') as fi:
        fi.write(str(trees[t]))
pickle.dump(trees,open('bamse.pickle','w'))      