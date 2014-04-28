#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
def simplex(c, A, b, t=None):
    #http://www.ms.uky.edu/~rwalker/Class%20Work%20Solutions/class%20work%208%20solutions.pdf
    #https://docs.google.com/presentation/d/1g2Ua2mkvZFTaLDV9lYQC7zzf7WU68vRNEtoTDieLq1Y/embed?hl=en&size=s#slide=id.g2eeb321b_0_59
    #From Discrete Optimization course
    while True:
        if t is None:
            #Tableau with cost on bottom
            #Slack variables after main variables
            #Contraints on the right
            # + 1 to handle cost row, + 2 to handle bounds + "P" column
            tableau = t = np.zeros((A.shape[0] + 1, A.shape[1] + 2))
            t[-1, :-2] = -c
            t[-1, -2] = 1.
            t[:-1, :-2] = A
            t[:-1, -1] = b

        val = t[-1, :].min()
        if val > -1E-6:
            print 'Optimality reached'
            break

        col = t[-1, :-1].argmin()
        row = np.argmin(t[:-1, -1] / t[:-1, col])
        t[row, :] /= t[row, col]
        for ri in range(t.shape[0]):
            if ri != row:
                t[ri, :] = t[ri, :] - t[ri, col] * t[row, :]

    ret = np.zeros(A.shape)
    ret[0, col] = t[0, -1]
    return col, ret


def solve_it(input_data):
    # parse the input
    lines = input_data.split('\n')

    first_line = lines[0].split()
    item_count = int(first_line[0])
    capacity = int(first_line[1])

    #Value, weight
    items = np.zeros((item_count, 2))
    for i in range(item_count):
        line = lines[i]
        parts = line.split()
        items[i, :] = np.array([int(parts[0]), int(parts[1])])

    value = items[:, 0]
    weight = items[:, 1]
    # min(-c.Tx)
    # Ax <= b, x >= 0
    # x = {1, 0} binary indicator
    # c = value
    # b = capacity
    # A = weight

    b = np.array([capacity])
    # Add slack variables
    c = np.hstack((value, [0]))
    A = np.hstack((weight, [1]))[:, None].T
    i, x = simplex(c, A, b)
    #Remove slack variables
    x = x.ravel()[:-1]
    print np.dot(c[:-1], x)
    # taken needs to be a list
    # value needs to be a list as well
    taken = x.tolist()
    value = value.tolist()

    # prepare the solution in the specified output format
    output_data = ' '.join(map(lambda x: str(int(x)), value)) + \
            ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, taken))
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
