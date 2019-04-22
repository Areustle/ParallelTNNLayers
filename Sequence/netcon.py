"""Get optimal contraction sequence using netcon algorithm

Reference:
    R. N. C. Pfeifer, et al.: Phys. Rev. E 90, 033315 (2014)
"""

#Tensorflow adaptation
#__author__ = "Satoshi MORITA <morita@issp.u-tokyo.ac.jp>"
#__date__ = "24 March 2016"

import tensorflow as tf
import numpy as np
import sys
import logging
import time
#import config

DIMS = []

class Tensor:
    """Strips down a tensor as a list of key properties for the purposes of netcon.

    Attributes:
        rpn: contraction sequence with reverse polish notation.
        decomp: a representation of contracted tensors.
        bonds: list of dimension of indices.
        is_new: a flag.
    """

    def __init__(self,rpn=[],bonds=[],cost=0.0,is_new=True):
        self.rpn = rpn
        #self.decomp[:] = [x for x in self.rpn if x != -1] #Translates from reverse polish notation to indicate tensors involved. 
        self.bonds = bonds
        self.cost = cost
        self.is_new = is_new

    def __str__(self):
        return "{0} : bond={1} cost={2:.6e} decomp={3}  new={4}".format(
            self.rpn, list(self.bonds), self.cost, self.decomp, self.is_new)


def netcon(tList, DIMS):
    """Find optimal contraction sequence.

    Args:
        tList: List of tensors
        DIMS: Global containing a complete list of index dimensions across all tensors. 

    Return:
        rpn: Optimal contraction sequence with reverse polish notation.
        cost: Total contraction cost.
    """
    tensor_set = _init(tn)

    n = len(tList)
    xi_min = float(min(DIMS))
    mu_cap = 1.0
    mu_old = 0.0

    while len(tList)<1:
        logging.info("netcon: searching with mu_cap={0:.6e}".format(mu_cap))
        mu_next = sys.float_info.max
        for c in range(1,n):
            for d1 in range((c+1)/2):
                d2 = c-d1-1
                n1 = len(tList[d1])
                n2 = len(tList[d2])
                for i1 in range(n1):
                    i2_start = i1+1 if d1==d2 else 0
                    for i2 in range(i2_start, n2):
                        t1 = tList[d1][i1]
                        t2 = tList[d2][i2]

                        if _is_disjoint(t1,t2): continue
                        if _is_overlap(t1,t2): continue

                        mu = _get_cost(t1,t2)
                        mu_0 = 0.0 if (t1.is_new or t2.is_new) else mu_old

                        if (mu > mu_cap) and (mu < mu_next): mu_next = mu
                        if (mu > mu_0) and (mu <= mu_cap):
                            t_new = _contract(t1,t2)
                            is_find = False
                            for i,t_old in enumerate(tensor_set[c]):
                                if t_new.bit == t_old.bit:
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


def _init(tn):
    """Initialize a set of tensors from a tensor network."""
    tensor_set=[[] for t in tn]
    for t in tn:
        name=tf.Session().run(t)
        rpn = [name]
        #decomp = [x for x in rpn if x != -1]
        bonds=t.get_shape().as_list()
        #bonds = frozenset(bonds)
        cost = 0.0
        tensor_set[0].append(Tensor(rpn,bonds,cost))
    return tensor_set


def _get_cost(t1,t2):
    """Get the cost of contraction of two tensors."""
    cost = 1.0
    for b in (t1.bonds + t2.bonds):
        cost *= b
    cost += t1.cost + t2.cost
    return cost


def _contract(t1,t2):
    """Return a contracted tensor"""
    assert (not _is_disjoint(t1,t2))
    rpn = t1.rpn + t2.rpn + [-1]
    bit = t1.decomp ^ t2.decomp # XOR
    bonds = frozenset(t1.bonds ^ t2.bonds)
    cost = _get_cost(t1,t2)
    return Tensor(rpn,decomp,bonds,cost)


def _is_disjoint(t1,t2):
    """Check if two tensors are disjoint."""
    t1list=tf.session().run(t1.bonds)
    t1list=np_to_list(t1list)

    t2list=_np_to_list(t2.bonds)
    t2list=_np_to_list(t2list)
    return (t1list).isdisjoint(t2list)


def _is_overlap(t1,t2):
    """Check if two tensors have the same basic tensor."""
    return not (t1.decomp).isdisjoint(t2.decomp)


def _print_tset(tensor_set):
    """Print tensor_set. (for debug)"""
    for level in range(len(tensor_set)):
        for i,t in enumerate(tensor_set[level]):
            print(level,i,t)

def _np_to_list(arr):
    return map(tuple,arr)