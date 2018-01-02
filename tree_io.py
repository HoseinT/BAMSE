""" This file contatins functions for writing output files"""
from tree_tools import *
from asciitree import LeftAligned
import numpy as np
import pickle

def table_expression(remheight, remwidth, border, orient, tree, node, vafs, colors):
    """ generates the table html node expression for the sample tiling output"""
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

def get_asciitree_dict(tree,labels,node):
    return {labels[x]:get_asciitree_dict(tree,labels,x) for x in get_childs(tree,node)}

def get_ascii_tree(tree,labels):
    root = list(tree).index(-1)
    tree_dict =  {labels[root]:get_asciitree_dict(tree,labels,root)}
    tr = LeftAligned()
    return tr(tree_dict)

def write_text_output(data,destfile):
    with open(destfile,'w') as fi:
        for i,tree in enumerate(data):
            fi.write('Solution Number '+str(i+1)+'\n')
            fi.write('tree = ' + str(tree['tree']) + '\n')
            depth = get_tree_depth(tree['tree'])
            print_depth = max(depth) - np.array(depth) + 1
            labels = [chr(x+65)+' '*4*print_depth[x] + \
            str(np.around(tree['VAF'][x,:],2)) for x in range(len(tree['tree']))]
            fi.write(get_ascii_tree(tree['tree'],labels)+'\n')
            fi.write('logscore = '+ str(tree['totalscore']) + '\n')
            fi.write('ML Subclone Fractions = \n' + str(tree['clone_proportions']) + '\n' )
            fi.write ('Mutations Subclone Membership = ' + str([chr(x+65) for x in tree['assign']]) + '\n')
            fi.write('----------------------------------------------------------------')
    return
