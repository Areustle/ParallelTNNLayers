"""Faster identification of optimal contraction sequences for tensor networks
Reference:
    R. N. C. Pfeifer, et al.: Phys. Rev. E 90, 033315 (2014)
    10.1103/PhysRevE.90.033315
    
Tensorial Neural Networks: Generalization of Neural Networks and Application to Model Compression
    Jiahao Su, Jingling Li, Bobby Bhattacharjee, Furong Huang
    arXiv:1805.10352
"""

"""Tensorflow adaptation of https://github.com/smorita/Tensordot/blob/master/netcon.py 
This allows for generalized tensor operations as described in """
#original__author__ = "Satoshi MORITA <morita@issp.u-tokyo.ac.jp>"
#__date__ = "24 March 2016"

#new__author__ = "Tahseen Rabbani<trabbani@math.umd.edu"

import tensorflow as tf
import numpy as np
import math
import sys
import logging
import time
#import config

DIMS = {} #Dictionary containing dimension of bonds. (dimensions of indices)

class Tensor:
    """Strips down a tensor as a list of key properties for the purposes of netcon.
    Attributes:
        rpn: contraction sequence with reverse polish notation.
        decomp: a list 
        bonds: list of indices in Eisenstein notation
        is_new: a flag.
    """

    def __init__(self,rpn=[],decomp=[],bonds=[],cost=0.0,is_new=True):
        self.rpn = rpn[:]
        self.decomp = decomp  #Translates from reverse polish notation to indicate tensors involved. 
        self.bonds = bonds
        self.cost = cost
        self.is_new = is_new

    def __str__(self):
        return "{0} : bond={1} cost={2:.6e} decomp={3}  new={4}".format(
            self.rpn, list(self.bonds), self.cost, self.decomp, self.is_new)


def gnetcon(tn,bonds,costmodel=0):
    """Find optimal contraction sequence.
    Args:
        tn: Tensor network/list of tensors.
        bonds: a list of bonds/legs for each tensor, in modified** (see below) Eisenstein notation. Where 
        two tensors are to be contracted, the same subscript will be used. An example:
        
        Bond legs should be in the same order as the dimensions listed by get_shape()
        
        This can handle generalized tensor operations. 
        
        **FORMATTING**
        The bond/leg lists require sensitive formatting. Please read below. 
        
        -Leg names must be of type string. Use *lower-case* letter-number combinations, 
        EXCEPT in the case of convolution and partial outer product (see below):
        
        -Bond legs should be in the same order as the dimensions listed by tf.get_shape()
        
        -For outer product, just use different leg names.
        Example: [i0,j,k] & [k,l,m1] have no common indices, so an outer product is performed.
        
        -For convolution, append * to the end of leg name, use same leg name, but with different casing.
        We assume here that convolution between i & j pads the new leg to dimension length of max(i,j).
        USE LOWER CASE FOR THE LEG WITH HIGHER DIMENSION.
        
        Example: [a,b*,c] & [d,B*,e] will convolve between the b*/B* indices. 
        Please note that this assumes the convolution for a particular leg is only between two tensors.
        I.e., [a,b*,c] & [d,B*,e] & [f, b*, h] is NOT valid.
       
       -For partial outer product, append + to end of leg names, with upper and lower casing.
        Example: [n,p+,q] & [r,P+,s] will perform a partial outer product between p+/P+ indices.
       
       -For contraction, just leave the leg names the same.
        Example" [i0,j3,k] & [i0,l2,m] will contract between the i indices. 
        
        Example of a tensor network with multiple tensor operations.
        
        Here, we are contracting on bonds j between A & B, convolving on bonds j&J between A & C
        and taking a partial outer product on bond k between C & D.
        We would express the bonds as follows:
        A_bonds = [j,q*]
        B_bonds = [j,l,m]
        C_bonds = [Q, k+]
        D_bonds = [n, k+, p]
           
            q   Q
          A---^---C
          |       | 
        j |       = k
          |       |
          B       D
         / \     / \ 
        l   m   n   p
        
    costmodel=0 # of floating point operations
                 =1 cost of storing intermediate arrays (better for GPUs)
 
    Return:
        rpn: Optimal operation sequence with reverse polish notation.
        E.g.,  [B,A,C,-1,-1] means perform all shared operations between A & C and then perform
                remaining shared operations with B.
        cost: Total contraction cost.
    """
    tensor_set = _init(tn,bonds)

    n = len(tensor_set)
    min_key=min(DIMS.keys(), key=(lambda k: DIMS[k]))
    xi_min = DIMS[min_key]
    mu_cap = 1.0
    mu_old = 0.0

    while len(tensor_set[-1])<1:
        logging.info("netcon: searching with mu_cap={0:.6e}".format(mu_cap))
        mu_next = sys.float_info.max
        for c in range(1,n):
            for d1 in range(math.floor((c+1)/2)):
                d2 = c-d1-1
                n1 = len(tensor_set[d1])
                n2 = len(tensor_set[d2])
                for i1 in range(n1):
                    i2_start = i1+1 if d1==d2 else 0
                    for i2 in range(i2_start, n2):
                        t1 = tensor_set[d1][i1]
                        t2 = tensor_set[d2][i2]

#                        if _is_disjoint(t1,t2): continue
                        if _is_overlap(t1,t2): continue

                        mu = _get_cost(t1,t2)
                        mu_0 = 0.0 if (t1.is_new or t2.is_new) else mu_old

                        if (mu > mu_cap) and (mu < mu_next): mu_next = mu
                        if (mu > mu_0) and (mu <= mu_cap):
                            t_new = _operate(t1,t2,costmodel)
                            is_find = False
                            for i,t_old in enumerate(tensor_set[c]):
                                if set(t_new.decomp) == set(t_old.decomp):
                                    if t_new.cost < t_old.cost:
                                        tensor_set[c][i] = t_new
                                    is_find = True
                                    break
                            if not is_find: tensor_set[c].append(t_new)
        mu_old = mu_cap
        mu_cap = max(mu_next, mu_cap*xi_min)
        for s in tensor_set:
            for t in s: t.is_new = False

        logging.debug("netcon: tensor_num=" +  str([ len(s) for s in tensor_set]))

    t_final = tensor_set[-1][0]
    return t_final.rpn, t_final.cost


def _init(tn, bonds):
    """Initialize a set of tensors from a tensor network."""
    tensor_set=[[] for t in tn]
    global DIMS
    i=0
    for t in tn:
        bondlist=[] 
        rpn = [t.name]
        decomp = [x for x in rpn if x != -1]
        shape=t.get_shape().as_list()
        j=0 #Index of individual legs. 
        for legname in bonds[i]: #Running list of index dimensions keyed by name.   
            DIMS[legname]=shape[j] 
            bondlist.append(legname)   
            j+=1
        cost = 0.0
        tensor_set[0].append(Tensor(rpn,decomp,bondlist,cost))
        i+=1
    return tensor_set


def _get_cost(t1,t2,costmodel=0):
    """Get the cost of conducting multiple generalized operations on two tensors."""
    cost = 1.0
    t1bonds=t1.bonds
    t2bonds=t2.bonds
    lowerbonds1= [k.lower() for k in t1bonds]
    lowerbonds2= [k.lower() for k in t2bonds] #lower case everything to identify convolving/p.o.p. pairs
    commonbonds=list(set(lowerbonds1).intersection(lowerbonds2)) #list out any common bonds.
    
    fullbonds=t1.bonds+t2.bonds #Full list of bonds.
    bonds_with_conv = set(_conv_pop(fullbonds,False,True)) #Determines bonds after performing p.o.p's
    
    #contractions/pop's, but not convolution
    
    if len(commonbonds)==0: #No common bonds, so perform outer product
        for b in fullbonds:
            cost *=DIMS[b] #Here, cost model is irrelevant.
        cost = cost + t1.cost + t2.cost 
        return cost
    if costmodel==0:
        for b in bonds_with_conv:
            cost *= DIMS[b] 
            #Multiplies bond dimensions together. For convolutions, both leg dimensions are multiplied in.
    elif costmodel==1:
        nocontracts = set(t1bonds).symmetric_difference(set(t2bonds)) #Bonds with no contractions
        bonds_with_no_pairs = set(_conv_pop(nocontracts,True,True,True)) #Bonds with no pairs at all. 
        for b in bonds_with_no_pairs:
            cost *= DIMS[b]
#                cost *= DIMS[b] #Multiplies *unshared* bond dimensions together.
#                #Since legs targeted for convolution/p.o.p. technically have different names, 
#                #symmetric_difference will not remove them, so we just check for the appended operator.
    cost = cost + t1.cost + t2.cost
    return cost


def _operate(t1,t2,costmodel=0):
    """Returns a modified tensor after all convolutions, outer products and contractions."""
    #assert (not _is_disjoint(t1,t2))
    rpn = t1.rpn + t2.rpn + [-1]
    
    
    decomp = set(t1.decomp).symmetric_difference(set(t2.decomp)) # XOR
    decomp = list(decomp) #Indicate which initial tensors are involved.
    
    t1bonds = set(t1.bonds)
    t2bonds = set(t2.bonds)
    
    fullbonds = (t1bonds).symmetric_difference(t2bonds) #First remove contracted pairs. 
 
    newbonds = list(_conv_pop(fullbonds))
    
    cost = _get_cost(t1,t2,costmodel)
    return Tensor(rpn,decomp,newbonds,cost)

#DEPRECATED
#If two tensors in a network have no leg in common, we'll perform  an outer product. 
#def _is_disjoint(t1,t2):
#    """Check if two tensors are disjoint."""
#    return set(t1.bonds).isdisjoint(set(t2.bonds))


def _is_overlap(t1,t2):
    """Check if two tensors have the same basic tensor."""
    return not set(t1.decomp).isdisjoint(set(t2.decomp))


def _print_tset(tensor_set):
    """Print tensor_set. (for debug)"""
    for level in range(len(tensor_set)):
        for i,t in enumerate(tensor_set[level]):
            print(level,i,t)
            
#Performs partial outer product and convolution on a list of bonds.
def _conv_pop(fullbonds,do_Conv=True,do_Pop=True,remove_all_pairs=False):
    newbonds=[]
    tempBonds = [x for x in fullbonds] #Cache the fullbonds list
    #Now we handle convolutions and partial outer products
    for legs in tempBonds:
        
        if not legs in fullbonds: continue #If we've removed this leg from fullbonds, we don't care.
        
        if legs[-1]=="*" and do_Conv:
            if DIMS[legs]==DIMS[legs.lower()] and (legs.upper() in fullbonds): #Check for a convolved pair.
                fullbonds.remove(legs.upper()) #Remove the convolving leg with lower dimension.
                if remove_all_pairs:
                    fullbonds.remove(legs) #Remove the entire convolution pair
            elif DIMS[legs]==DIMS[legs.upper()] and legs.lower() in fullbonds:
                fullbonds.remove(legs.upper()) #Else remove itself if it has the lower dimension
                if remove_all_pairs:
                    fullbonds.remove(legs.lower())
        elif legs[-1]=="+" and do_Pop:
            if legs==legs.upper() and (legs.lower() in fullbonds): #Check for a partial o.p. pair
                fullbonds.remove(legs.upper())
                if remove_all_pairs:
                    fullbonds.remove(legs) #Remove the entire partial outer product pair.
            elif legs.upper() in fullbonds:
                fullbonds.remove(legs.upper()) #Remove one of the o.p. pair 
                if remove_all_pairs:
                    fullbonds.remove(legs)
    newbonds = list(fullbonds)
    return newbonds
