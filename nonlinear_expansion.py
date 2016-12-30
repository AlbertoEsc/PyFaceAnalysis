#Functions for generating several types of non-linear expansions
#By Alberto Escalante. Alberto.Escalante@neuroinformatik.ruhr-uni-bochum.de First Version 10 Dec 2009
#Ruhr-University-Bochum, Institute of Neurocomputation, Group of Prof. Dr. Wiskott

import numpy
import scipy
import scipy.optimize
import sfa_libs

#functions suitable for any n-dimensional arrays
#this function is so important that we give it a name although it does almost nothing
def identity(x):
    return x

def abs_dif(x1, x2):
    return numpy.abs(x1 - x2)

def abs_sum(x1, x2):
    return numpy.abs(x1 + x2)

def multiply(x1, x2):
    return x1 * x2

def signed_sqrt_multiply(x1, x2):
    z = x1*x2
    return signed_sqrt(z)

def unsigned_sqrt_multiply(x1, x2):
    z = x1*x2
    return unsigned_sqrt(z)

def sqrt_abs_sum(x1, x2):
    return numpy.sqrt(numpy.abs(x1+x2))

def sqrt_abs_dif(x1, x2):
    return numpy.sqrt(numpy.abs(x1-x2))

def neg_expo(x, expo):
    signs = numpy.sign(x)
    y = numpy.abs(x)**expo * signs
    return y

#This should be done faster.... iterating over rows is too slow!!!!!
#Expansion with terms: f(x1,x1), f(x1,x2), ... f(x1,xn), f(x2,x2), ... f(xn,xn)
#If reflexive=True include terms f(xj, xj)
##This (twice commented) version is slower, so now it was replaced by the function bellow
##see experiments_general_expansion for a benchmark
##def pairwise_expansion(x, func, reflexive=True):
##    """Computes func(xi, xj) over all possible indices i and j.
##    if reflexive==False, only pairs with i!=j are considered
##    """
##    x_height, x_width = x.shape
##    if reflexive==True:
##        k=0
##        out = numpy.zeros((x_height, x_width*(x_width+1)/2))
##    else:
##        k=1
##        out = numpy.zeros((x_height, x_width*(x_width-1)/2))    
##    mask = numpy.triu(numpy.ones((x_width,x_width)), k) > 0.5
##    for i in range(0, x_height):
##        y1 = x[i].reshape(x_width, 1)
##        y2 = x[i].reshape(1, x_width)
##        yexp = func(y1, y2)
###        print "yexp=", yexp
##        out[i] = yexp[mask]
##    return out    

def pairwise_expansion(x, func, reflexive=True):
    """Computes func(xi, xj) over all possible indices i and j.
    if reflexive==False, only pairs with i!=j are considered
    """
    x_height, x_width = x.shape
    if reflexive==True:
        k=0
        out = numpy.zeros((x_height, x_width*(x_width+1)/2))
    else:
        k=1
        out = numpy.zeros((x_height, x_width*(x_width-1)/2))    
    mask = numpy.triu(numpy.ones((x_width,x_width)), k) > 0.5
#    mask = mask.reshape((1,x_width,x_width))
    y1 = x.reshape(x_height, x_width, 1)
    y2 = x.reshape(x_height, 1, x_width)
    yexp = func(y1, y2)
    
#    print "yexp.shape=", yexp.shape
#    print "mask.shape=", mask.shape
    out = yexp[:, mask]  
#    print "out.shape=", out.shape
    #yexp.reshape((x_height, N*N))
    return out 


#Expansion with terms: f(x_i,x_i), f(x_i,x_i), ... f(x_i,x_i+k), f(x_i+1,x_i+1), ... f(x_n-k,x_n)
#If reflexive=True include terms f(x_j, x_j)
##This (twice commented) version is slower, so now it was replaced by the function bellow
##see experiments_general_expansion for a benchmark
##def pairwise_adjacent_expansion(x, adj, func, reflexive=True):
##    """Computes func(xi, xj) over a subset of all possible indices i and j
##    in which abs(j-i) <= adj
##    if reflexive==False, only pairs with i!=j are considered
##    """
##    x_height, x_width = x.shape
##    if reflexive is True:
##        k=0
##    else:
##        k=1
###   number of variables, to which the first variable is paired/connected
##    mix = adj-k
##    out = numpy.zeros((x_height, (x_width-adj+1)*mix))
###
##    vars = numpy.array(range(x_width))
##    v1 = numpy.zeros(mix * (x_width-adj+1), dtype='int')
##    for i in range(x_width-adj+1):
##        v1[i*mix:(i+1)*mix] = i
###
##    v2 = numpy.zeros(mix * (x_width-adj+1), dtype='int')
##    for i in range(x_width-adj+1):
##        v2[i*mix:(i+1)*mix] = range(i+k,i+adj)
###    
##    for i in range(x_height):
##        out[i] = map(func, x[i][v1], x[i][v2])
###        print "yexp=", yexp
###        out[i] = yexp[mask]
##    return out    

def pairwise_adjacent_expansion(x, adj, func, reflexive=True):
    """Computes func(xi, xj) over a subset of all possible indices i and j
    in which abs(j-i) <= mix, mix=adj-k
    if reflexive==False, only pairs with i!=j are considered
    """
    x_height, x_width = x.shape
    if reflexive is True:
        k=0
    else:
        k=1
#   number of variables, to which the first variable is paired/connected
    mix = adj-k
    out = numpy.zeros((x_height, (x_width-adj+1)*mix))
#
    vars = numpy.array(range(x_width))
    v1 = numpy.zeros(mix * (x_width-adj+1), dtype='int')
    for i in range(x_width-adj+1):
        v1[i*mix:(i+1)*mix] = i
#
    v2 = numpy.zeros(mix * (x_width-adj+1), dtype='int')
    for i in range(x_width-adj+1):
        v2[i*mix:(i+1)*mix] = range(i+k,i+adj)
#   
#    print "v1=", v1 
#    print "v2=", v2 
#    print "x[:,v1].shape=", x[:,v1].shape
#    print "x[:,v2].shape=", x[:,v2].shape

    out = func(x[:,v1], x[:,v2])
#        print "yexp=", yexp
#        out[i] = yexp[mask]
    return out  

#Two-Halbs mixed product expansion
def halbs_product_expansion(x, func):
    """Computes func(xi, xj) over a subset of all possible indices i and j
    in which 0<=i<N and N<=j<2N
    where 2N is the dimension size
    """
    x_height, x_width = x.shape
    if x_width%2 != 0:
        ex = "input dimension must be of even!!!"
        raise ex
    N = x_width/2

    y1 = x[:,:N].reshape(x_height, N, 1)
    y2 = x[:,N:].reshape(x_height, 1, N)
    print "y1.shape=", y1.shape
    print "y2.shape=", y2.shape
    yexp = func(y1, y2)
    print "yexp.shape=", yexp.shape
    
    return yexp.reshape((x_height, N*N))

def halbs_multiply_ex(x):
    return halbs_product_expansion(x, multiply)
#
#xx = numpy.arange(20).reshape((5,4))
#print "xx=", xx
#yy = halbs_multiply_ex(xx)
#print "yy=", yy

def unsigned_11expo(x):
    return numpy.abs(x) ** 1.1

def signed_11expo(x):
    return neg_expo(x, 1.1)


def unsigned_15expo(x):
    return numpy.abs(x) ** 1.5

def signed_15expo(x):
    return neg_expo(x, 1.5)

def tanh_025_signed_15expo(x):
    return numpy.tanh(0.25 * neg_expo(x, 1.5)) / 0.25

def tanh_05_signed_15expo(x):
    return numpy.tanh(0.50 * neg_expo(x, 1.5)) / 0.5

def tanh_0125_signed_15expo(x):
    return numpy.tanh(0.125 * neg_expo(x, 1.5)) / 0.125

def unsigned_08expo(x):
    return numpy.abs(x) ** 0.8

def unsigned_08expo_m1(x):
    return numpy.abs(x-1) ** 0.8

def unsigned_08expo_p1(x):
    return numpy.abs(x+1) ** 0.8

def signed_06expo(x):
    return neg_expo(x, 0.6)

def unsigned_06expo(x):
    return numpy.abs(x) ** 0.6

def signed_08expo(x):
    return neg_expo(x, 0.8)

def signed_sqrt(x):
    return neg_expo(x, 0.5)

def unsigned_sqrt(x):
    return numpy.abs(x) ** 0.5

def signed_sqr(x):
    return neg_expo(x, 2.0)

def e_neg_sqr(x):
    return numpy.exp(-x **2)

#Weird sigmoid
def weird_sig(x):
    x1 = numpy.exp(-x **2)
    x1[x<0] = 2 -x1 [x<0]
    return x1

def weird_sig2(x):
    x1 = numpy.exp(- (x/2) **2 )
    x1[x<0] = 2 -x1 [x<0]
    return x1

def weird_sig_prod(x1, x2):
    z = x1*x2
    k1 = numpy.exp(- numpy.abs(z) **2)
    k1[z<0] = 2 - k1[z<0]
    return k1

def pair_abs_dif_ex(x):
    return pairwise_expansion(x, abs_dif, reflexive=False)

def pair_abs_sum_ex(x):
    return pairwise_expansion(x, abs_sum)

def pair_prod_ex(x):
    return pairwise_expansion(x, multiply)

def pair_sqrt_abs_sum_ex(x):
    return pairwise_expansion(x, sqrt_abs_sum)

def pair_sqrt_abs_dif_ex(x):
    return pairwise_expansion(x, sqrt_abs_dif, reflexive=False)

#Only product of strictly consecutive variables
def pair_prod_mix1_ex(x):
    return pairwise_adjacent_expansion(x, adj=2, func=multiply, reflexive=False)

def pair_prod_mix2_ex(x):
    return pairwise_adjacent_expansion(x, adj=3, func=multiply, reflexive=False)

def pair_prod_mix3_ex(x):
    return pairwise_adjacent_expansion(x, adj=4, func=multiply, reflexive=False)

#Only sqares of input variables
def pair_prod_adj1_ex(x):
    """returns x_i ^ 2 """
    return pairwise_adjacent_expansion(x, adj=1, func=multiply, reflexive=True)

#Squares and product of adjacent variables
def pair_prod_adj2_ex(x):
    """returns x_i ^ 2 and x_i * x_i+1"""
    return pairwise_adjacent_expansion(x, adj=2, func=multiply, reflexive=True)

def pair_prod_adj3_ex(x):
    return pairwise_adjacent_expansion(x, adj=3, func=multiply, reflexive=True)

def pair_prod_adj4_ex(x):
    return pairwise_adjacent_expansion(x, adj=4, func=multiply, reflexive=True)

def pair_prod_adj5_ex(x):
    return pairwise_adjacent_expansion(x, adj=5, func=multiply, reflexive=True)

def pair_prod_adj6_ex(x):
    return pairwise_adjacent_expansion(x, adj=6, func=multiply, reflexive=True)

def pair_sqrt_abs_dif_adj2_ex(x):
    """returns sqrt(abs(x_i - x_i+1)) """
    return pairwise_adjacent_expansion(x, adj=2, func=sqrt_abs_dif, reflexive=False)

def pair_sqrt_abs_dif_adj3_ex(x):
    return pairwise_adjacent_expansion(x, adj=3, func=sqrt_abs_dif, reflexive=False)

def pair_sqrt_abs_dif_adj4_ex(x):
    return pairwise_adjacent_expansion(x, adj=4, func=sqrt_abs_dif, reflexive=False)

def pair_sqrt_abs_dif_adj5_ex(x):
    return pairwise_adjacent_expansion(x, adj=5, func=sqrt_abs_dif, reflexive=False)

def pair_sqrt_abs_dif_adj6_ex(x):
    return pairwise_adjacent_expansion(x, adj=6, func=sqrt_abs_dif, reflexive=False)


#Normalized product of adjacent variables (or squares)
def signed_sqrt_pair_prod_adj2_ex(x):
    """returns f(x_i ^ 2) and f(x_i * x_i+1)"""
    return pairwise_adjacent_expansion(x, adj=2, func=signed_sqrt_multiply, reflexive=True)

def signed_sqrt_pair_prod_adj3_ex(x):
    """returns f(x_i ^ 2), f(x_i * x_i+1) and f(x_i * x_i+2)"""
    return pairwise_adjacent_expansion(x, adj=3, func=signed_sqrt_multiply, reflexive=True)

def signed_sqrt_pair_prod_mix1_ex(x):
    """returns f(x_i * x_i+1)"""
    return pairwise_adjacent_expansion(x, adj=2, func=signed_sqrt_multiply, reflexive=False)

def signed_sqrt_pair_prod_mix2_ex(x):
    """returns f(x_i * x_i+1)"""
    return pairwise_adjacent_expansion(x, adj=3, func=signed_sqrt_multiply, reflexive=False)

def signed_sqrt_pair_prod_mix3_ex(x):
    """returns f(x_i * x_i+1)"""
    return pairwise_adjacent_expansion(x, adj=4, func=signed_sqrt_multiply, reflexive=False)

def unsigned_sqrt_pair_prod_mix1_ex(x):
    """returns f(x_i * x_i+1)"""
    return pairwise_adjacent_expansion(x, adj=2, func=unsigned_sqrt_multiply, reflexive=False)

def unsigned_sqrt_pair_prod_mix2_ex(x):
    """returns f(x_i * x_i+1)"""
    return pairwise_adjacent_expansion(x, adj=3, func=unsigned_sqrt_multiply, reflexive=False)

