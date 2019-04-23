"""Faster identification of optimal contraction sequences for tensor networks

Reference:
    R. N. C. Pfeifer, et al.: Phys. Rev. E 90, 033315 (2014)
    10.1103/PhysRevE.90.033315
"""
#Tensorflow adaptation of https://github.com/smorita/Tensordot/blob/master/netcon.py
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
        self.rpn = rpn
        self.decomp = decomp  #Translates from reverse polish notation to indicate tensors involved. 
        self.bonds = bonds
        self.cost = cost
        self.is_new = is_new

    def __str__(self):
        return "{0} : bond={1} cost={2:.6e} decomp={3}  new={4}".format(
            self.rpn, list(self.bonds), self.cost, self.decomp, self.is_new)


def netcon(tn,bonds):
    """Find optimal contraction sequence.

    Args:
        tn: Tensor network/list of tensors.
        bonds: a list of bonds for each tensor, in Eisenstein notation. Where 
        two tensors are to be contracted, the same subscript will be used. An example:
        
             i
       j---A---B---m
           |   |
           k   l
        If tensor order is [A, B]
        bond_dims[0]=['ijk']
        bond_dims[1]=['ilm']
        Bond legs should be in the same order as the dimensions listed by get_shape()
 
    Return:
        rpn: Optimal contraction sequence with reverse polish notation.
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

                        if _is_disjoint(t1,t2): continue
                        if _is_overlap(t1,t2): continue

                        mu = _get_cost(t1,t2)
                        mu_0 = 0.0 if (t1.is_new or t2.is_new) else mu_old

                        if (mu > mu_cap) and (mu < mu_next): mu_next = mu
                        if (mu > mu_0) and (mu <= mu_cap):
                            t_new = _contract(t1,t2)
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
        j=0
        for char in bonds[i]:
            DIMS[char]=shape[j] #Running list of index dimensions keyed by bond name
            bondlist.append(char)
            j+=1
        cost = 0.0
        tensor_set[0].append(Tensor(rpn,decomp,bondlist,cost))
        i+=1
    return tensor_set


def _get_cost(t1,t2):
    """Get the cost of contraction of two tensors."""
    cost = 1.0
    fullbonds=t1.bonds+t2.bonds
    for b in set(fullbonds): #Remove duplicate bonds to express an eisen contract.
        cost *= DIMS[b] #Multiplies bond dimensions together.
    cost = cost + t1.cost + t2.cost
    return cost


def _contract(t1,t2):
    """Return a contracted tensor"""
    assert (not _is_disjoint(t1,t2))
    rpn = t1.rpn + t2.rpn + [-1]
    decomp = set(t1.decomp).symmetric_difference(set(t2.decomp)) # XOR
    decomp = list(decomp)
    bonds = set(t1.bonds).symmetric_difference(set(t2.bonds))
    bonds = list(bonds)
    cost = _get_cost(t1,t2)
    return Tensor(rpn,decomp,bonds,cost)


def _is_disjoint(t1,t2):
    """Check if two tensors are disjoint."""
    return set(t1.bonds).isdisjoint(set(t2.bonds))


def _is_overlap(t1,t2):
    """Check if two tensors have the same basic tensor."""
    return not set(t1.decomp).isdisjoint(set(t2.decomp))


def _print_tset(tensor_set):
    """Print tensor_set. (for debug)"""
    for level in range(len(tensor_set)):
        for i,t in enumerate(tensor_set[level]):
            print(level,i,t)
