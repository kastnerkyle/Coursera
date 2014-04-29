#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
def greedy(weights, values, bound):
    w = 0.
    v = 0.
    t = np.zeros(len(weights))
    for i in range(len(weights)):
        if (weights[i] + w) <= bound:
            t[i] = 1.
            v += values[i]
            w += weights[i]
    return w, v, t

def branch(weights, values, limit, m_weights=None, m_values=None, t=[]):
    if m_weights is None:
        m_weights = weights.copy()
    if m_values is None:
        m_values = values.copy()
    if len(m_weights):
        tw, tv, tt = branch(weights, values, limit - weights[0],
                            m_weights[1:], m_values[1:], t + [1])
        nw, nv, nt = branch(weights, values, limit,
                            m_weights[1:], m_values[1:], t + [0])
        print tw, tv, tt
        print nw, nv, nt
        if tv > nv:
            w, v, t = (tw, tv, tt)
        else:
            w, v, t = (nw, nv, nt)
    else:
        w = np.dot(weights, np.array(t))
        v = np.dot(values, np.array(t))
    return w, v, t


def branch_and_bound(weights, values, bound):
    #Arrange by maximum ratio
    d = weights / values
    si = np.argsort(d)[::-1]
    w_s = weights[si]
    v_s = values[si]

    #Weights to invert the output of greedy solution
    #because input is fed in sorted by weight:value ratio
    ti = np.argsort(si)

    #Get back weight and value for greedy
    #Result needs to be resorted due to getting sorted input
    gw, gv, t_u = greedy(w_s, v_s, bound)
    gt = t_u[ti]
    w, v, t = branch(weights, values, bound)
    return w, v, t

def solve_it(input_data):
    # parse the input
    lines = input_data.split('\n')

    first_line = lines[0].split()
    item_count = int(first_line[0])
    capacity = int(first_line[1])

    #values, weight
    items = np.zeros((item_count, 2))
    for i in range(1, item_count + 1):
        line = lines[i]
        parts = line.split()
        items[i - 1, :] = np.array([int(parts[0]), int(parts[1])])

    values = v = items[:, 0]
    weights = w = items[:, 1]
    bound = b = capacity

    #Trivial solution
    #w, v, t = greedy(w, v, b)

    #Better solution
    w, v, t = branch_and_bound(w, v, b)

    # taken needs to be a list
    # value needs to be a list as well
    taken = t.tolist()
    value = v

    # prepare the solution in the specified output format
    output_data = str(int(value)) + ' ' + ' ' + str(0) + '\n'
    output_data += ' '.join(map(lambda x: str(int(x)), taken))
    return output_data


import sys
if __name__ == '__main__':
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        input_data_file = open(file_location, 'r')
        input_data = ''.join(input_data_file.readlines())
        input_data_file.close()
        print solve_it(input_data)
    else:
        print 'This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)'
