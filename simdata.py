
from subprocess import check_call,check_output
import numpy as np
import pandas as pd
import re
import itertools
import os
import scipy
import timeit
import sys
import pickle
import bisect
from scipy.stats import binom
import matplotlib.pyplot as plt
import seaborn as sns


def sim_data_readcounts(K,M,alltrees,num_mut,coverage = 200, min_in_clust=2,error=0.003):
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



def sim_data_sparse_readcounts(K,M,alltrees,num_mut,sparsity = 0.3,coverage = 200, min_in_clust=2,error=0.003):
    ind = np.random.randint(len(alltrees[K]))
    true_tree = alltrees[K][ind]
    true_sm = np.zeros((K,M))
    while(np.min(np.sum(true_sm,0))==0 or np.min(np.sum(true_sm,1))==0):
        z= np.random.binomial(K,sparsity)
        true_sm = np.array([np.random.permutation(np.concatenate((np.zeros(z),np.random.dirichlet(np.ones(K-z+1))[:K-z]),0)) for m in range(M)])
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








def get_childs(tree,node):
    return [i for i, x in enumerate(tree) if x == node]

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

(2>3) and (3>4)

def compare_trees(trueT, testT, locations,phylo=False):
    trees_match=False
    relation_match = 0
    if canonizeF(trueT['tree'])['canstr'] == canonizeF(testT['tree'])['canstr']:
        trees_match = True
    if (phylo and min(testT['assign'])>0) :
        otree = list(np.array(testT['tree'])[1:]-1)
        if canonizeF(trueT['tree'])['canstr'] == canonizeF(otree)['canstr']:
            trees_match = True

    squared_error = np.mean((trueT['vaf'][trueT['assign'][locations]]     - testT['vaf'][testT['assign'][locations]])**2)
    allpairs = 0
    for (i,j) in itertools.combinations(locations,2):
        truei = trueT['assign'][i]
        truej = trueT['assign'][j]
        testi = testT['assign'][i]
        testj = testT['assign'][j]
        truef = (truei,truej)
        trueb = (truej,truei)
        testf = (testi,testj)
        testb = (testj,testi)
        if trueT['Amatrix'][truef] == testT['Amatrix'][testf]         and trueT['Amatrix'][trueb] == testT['Amatrix'][testb]        and trueT['Bmatrix'][truef] == testT['Bmatrix'][testf]        and trueT['Bmatrix'][trueb] == testT['Bmatrix'][testb]:
            relation_match +=1
        allpairs +=1
    return {'Tree Structure Matched?':trees_match,'Correctly Inferred Relationships %':     float(relation_match)/allpairs,'Subclone Fraction Squared Error':squared_error}

def parseLicheeWithRef(lichefile,mut_file):
    with open(lichefile,'r') as fi:
        contents = fi.read()
    temp = pd.read_table(mut_file)
    N=len(temp)
    nodes = re.findall('Nodes:([^\*]*)\n\n',contents,re.MULTILINE)[0]
    nodelines = nodes.split('\n')
    orig=[]
    for index, row in temp.iterrows():
        orig += [(re.findall('[0-9XY]+',row['#chrom'])[0],row['pos'])]
    orig = [(chr2num(x[0]),x[1]) for x in orig]
    SNVtable = re.findall('(snv[0-9]+): ([0-9]+) ([0-9]+)', contents)
    origstr = [str(x[0])+'-'+str(x[1]) for x in orig]
    SNVstr = [(x[0],str(x[1])+'-'+str(x[2])) for x in SNVtable]
    SNVmap = dict((x[0],origstr.index(x[1])) for x in SNVstr)
    sols = re.findall('\*\*\*\*\n([^\*]*)(\n\*\*\*\*|SNV info)',contents,re.MULTILINE)
    res=[]
    for sol in sols:
        score  = float(re.findall('score: (.*)',sol[0],re.MULTILINE)[0])
        edges = re.findall('([0-9]+) -> ([0-9]+)',sol[0],re.MULTILINE)
        edges  =[(int(x[0]),int(x[1])) for  x in edges]
        tree = -1*np.ones(len(edges))
        for x in edges:
            if x[0] > 0:
                tree[x[1]-1] = x[0]-1
        order = [int(x.split('\t')[0]) for x in nodelines[1:]]
        profile = np.array([[int(y) for y in x.split('\t')[1]] for x in nodelines[1:]])
        weights = [[float(z) for z in re.findall('[0-9\.]+',x.split('\t')[2])] for x in nodelines[1:]]
        w = np.zeros(profile.shape)
        for p  in range(len(profile)):
            w[p,np.where(profile[p]==1)[0]] = np.array(weights[p])
        rind = [order.index(x) for x in range(1,len(order)+1)]
        assigns = np.array([[SNVmap[y] for y in x.split('\t')[3:]] for x in nodelines[1:]])
        assign = np.array([max([k if n in assigns[rind][k] else -1 for k in range(len(tree))]) for n in range(N)])
        #print assigns, assign , rind
        B= get_matrix(tree)[0]
        res += [{'score':score, 'tree':tree, 'Bmatrix':B, 'Amatrix':np.linalg.inv(B), 'order':order, 'profile':profile[rind], 'weights':scipy.delete(w[rind],0,1), 'assign':assign}]
    return res

def trap_multiple(uniqs,filename_prefix,varianc):
    check_call('rm -r -f combined/', shell= True)
    for ind in range(len(uniqs)):
        sample = uniqs[ind]
        with open(filename_prefix+str(ind),"w") as f:
            f.write('DATATYPE GAUSSIAN '+str(varianc)+'\n')
            for i in range(len(sample)):
                f.write(r'SIGNAL A<sub>'+str(i)+r'</sub> '+str(sample[i])+'\n')
        f.close()
    check_call(r'java -jar /home/hosein/Projects/Paper/trap/'+    'TrApWithDependencies.jar --multisample '+filename_prefix+r'{0..'+    str(len(uniqs)-1)+'}',shell=True)
    for ind in range(len(uniqs)):
        check_call('rm '+filename_prefix+str(ind), shell= True)
    results = check_output(r'ls combined/imgs | grep .csv', shell=True).split('\n')[:-1]
#    results = ['combined/imgs/'+x for x in results[:-1]]
    M = len(uniqs)
    K = len(uniqs[0])
    num_res = len(results)/M
    results = []
    for r in range(num_res):
        s=[]
        for m in range(M):
            temp = pd.read_table('combined/imgs/clones_'+str(r+1)+'_'+            str(m)+'.csv',sep=",")
            s +=[temp[' Abundance']]
        s = np.array(s).transpose()
        tree = np.zeros(K)
        mat=temp[['A'+str(i) for i in range(K)]].as_matrix()
        root = [i for i in range(len(mat)) if sum(mat[i])==1]
        tree[root[0]] = -1
        leaves = root
        while (len(leaves)>0):
            new_leaves = []
            for leaf in leaves:
                childs = [i for i in range(len(mat)) if not                 any(mat[i]-mat[leaf]-indexTobinary([i],K))]
                tree[childs]= leaf
                new_leaves += childs
            leaves = new_leaves
        mat = mat.transpose()
        results += [{'tree':tree,'Bmatrix':mat, 's':s, 'vaf': mat.dot(s),         'Amatrix':np.linalg.inv(mat)}]
    return(results)

def parse_ancestree_output(outfile):
    with open(outfile,'r') as f:
        contents = f.read()
    empty_line_parsed = contents.split('\n\n')
    num_sol  = int(re.search('^[^#]*',empty_line_parsed[0]).group(0))
    result = []
    for count in range(num_sol):
        usage_and_mat = empty_line_parsed[(count+1)*3].split('\n')
        M = int(usage_and_mat[0])
        K = int(usage_and_mat[1])
        N = int(empty_line_parsed[1].split('\n')[1])
        usage = np.array([[float(x) for x in usage_and_mat[y+2].split(' ')[:-1]] for         y in range(M)]).transpose()
        mat = np.array([[float(x) for x in usage_and_mat[y+M+4].split(' ')[:-1]] for         y in range(K)]).transpose()
#        assign = [int(x) for x  in (empty_line_parsed[(count+1)*3+2].split('\n')[1].split(' ')[:-1])]
#        print assign
        assign1=[[int(x) for x in y.split(';')] for y in (empty_line_parsed[(count+1)*3+2].split('\n')[1].split(' ')[:-1])]
        assign = [max([k if n in assign1[k] else -1 for k in range(K)]) for n in range(N)]
        vaf = mat.dot(usage)/2
        result +=[{'assign':np.array(assign),'K':K,'M':M,'s':usage,'Bmatrix':mat,'Amatrix':np.linalg.inv(mat), 'tree':get_tree(mat), 'vaf':vaf}]
    return result



def chr2num(chrom):
    try:
        return int(chrom)
    except:
        if chrom == 'X':
            return 23
        if chrom == 'Y':
            return 24
        else:
            return -1

def indexTobinary(inds,length):
    a = [0 for x in range(length)]
    for ind in inds:
        a[ind]=1
    return a

def get_tree(mat):
    rev = np.linalg.inv(mat)
    tree = -1*np.ones(len(mat))
    wheres = np.where(rev==-1)
    tree[wheres[1]] = wheres[0]
    return tree



def read_Pastri_F(F_file):
    # parse F files output of pastri for vafs
    with open(F_file,'r') as fi:
        fi.readline()
        dims = fi.readline()
        temp = re.findall(pattern = '([0-9]+), ([0-9]+)',string = dims)
        K = int(temp[0][0])
        M = int(temp[0][1])
        F = np.zeros((K,M))
        for k in range(K):
            l = fi.readline()
            F[k,:] = [float(x) for x in re.findall(pattern = '([0-9,.]+)\s+', string=l)]
    return {'vaf':F}


def read_Pastri_C(C_file):
    #parse C_file output of Pastri for assignments
    with open(C_file,'r') as fi:
        mutations = []
        clusters = []
        for line in fi:
            nums = [int(x) for x in line.split('\t')]
            mutations += nums[1:]
            clusters += [nums[0] for _ in range(len(nums)-1)]
        num_mut = len(mutations)
        assign = np.zeros(len(mutations))
        for i,x in enumerate(mutations):
            assign[x] = int(clusters[i])
    return {'assign':np.ndarray.astype(assign,int)}

def read_Pastri_tree(tree_file):
    #parse the labeled tree file output of PASTRI
    with open(tree_file, 'r') as fi:
        fi.readline()
        l = fi.readline()
        edges = {}
        while l[0] in [str(x) for x in range(10)]:
            nodes = [int(x)  for x in l.split('\t')]
            edges.update({nodes[1]: nodes[0]})
            l = fi.readline()
        K = len(edges)+1
        Amatrix = np.diag(np.ones(K))
        tree = [-1 for _ in range(K)]
        for nodekey in edges:
            Amatrix[edges[nodekey],nodekey] = -1
            tree[nodekey] = edges[nodekey]
        Bmatrix = np.linalg.inv(Amatrix)
    return {'Amatrix':Amatrix, 'Bmatrix': Bmatrix, 'tree': tree}

def parse_Pastri (tree_file,F_file,C_file):
    res = {}
    res.update(read_Pastri_tree(tree_file))
    res.update(read_Pastri_C(C_file))
    res.update(read_Pastri_F(F_file))
    res['clone_proportions'] = res['Amatrix'].dot(res['vaf'])
    return res

