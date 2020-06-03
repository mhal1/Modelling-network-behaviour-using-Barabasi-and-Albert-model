#BA model
import networkx as nx
import random
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import os
import logbin as lb
import collections
import interpolation as ii
###############################################################################
# Will only print out information if N is below this number, use for testing
Nprint=20
# set seed to a number if you want to fix the random number seed for testing
# Otherwise set seed = None if you want true random numbers
seed=0
#seed = None  
# Random numbers, not used here but in general of use
# see http://docs.python.org/2/library/random.html
# I often use random.randrange, random.sample, and random.random
if seed is None:
    random.seed()
    print(' Random number seed not fixed for proper run')
else:
    random.seed(seed)
    print (' Random number seed fixed for testing')
###############################################################################
#functions
# algin the eqautions in latex
def initG(N,edges,m_o,k):
    # initialise graph 
    a=nx.Graph()
    # add nodes
    for i in range(N):
        a.add_node(i)
    # Now add edges    
    # avoid self-edges
    for s in range(m_o+1):
        #print(s)
        for t in range(s,N):
                a.add_edge(s,t)
                #shw(a)
                edges.append([s,t])
                k[s] += 1
                edges.append([t,s])
                k[t] += 1
    return a
#plot fn
def shw(G):
    plt.figure()
    nx.draw_circular(G)
    plt.show()
#adding edge with preferential attatchment
def addedge(Graph, edges,nod,k):
    # want to add one end of edge to new vertex
    # then add other edge to an existing vertex preferentially
    #now insert edges
    ch = random.choice(edges)
    #print(ch)
    v = random.choice(ch)
    #print(v,nod)
    # aviod self edges
   # shw(Graph)
    if v != nod:
    #then pick a vertex of this edge at random to join to the new vertex
        Graph.add_edge(nod,v)
        #print('edge added ',nod,v)
        edges.append([nod,v])
        edges.append([v,nod])
        k[nod] += 1
        k[v] += 1
    elif v == nod:
        #print(v,'=',nod)
        ch = random.choice(edges)
        #print('retry,',ch)
        v = random.choice(ch)
        #print(v,nod)
        Graph.add_edge(nod,v)
        k[nod] += 1
        k[v] += 1
        #print('edge added ',nod,v)
        edges.append([nod,v])
        edges.append([v,nod])
# need parameters
#N total number of inital vertices
N = 3
#m_o initial number of edges
m_o = 1
#m number of edges to add to each new vertex, m<m_o
m = 1
#initialise graph with two connected nodes
#plt.figure()
#nx.draw_circular(G)
#plt.show()
#edge that has been added shouldnt be added again
#n_add = 100000 #add this many nodes
#add nodes to initial graph
#degree of each nodes
#nodes = [x for x in range(N+n_add+1)]
#k = len(nodes)*[0]
def addnewnodes(N,n_add,G,m,edges,k):
    for i in range(N,n_add+N+1):
        G.add_node(i)
        # Now add edges    
        # avoid self-edges
        for h in range(m):
            addedge(G,edges,i,k)
def theoryval(m,n):
    a = m*(m+1)
    return (-1 + np.sqrt(1 + 4*n*a))/2

#list of m for same N
# need to do repeats and append before log binning!
repeats = 15
er = []
km = []
kmt = []
#################################################################################################
#plot a straight line through k1 vs N

for gg in [2,4,8]:
    nn =[10,100,1000,10000,100000]#,1000000]
    km = []
    kmt = []
    for n in nn:
        kmax = []
        for i in range(repeats):
            edges = []
            nodes = [x for x in range(N+n+1)]
            k = len(nodes)*[0] # initial list of nodes with zero degree
            G = initG(N,edges,2,k) # create initial graph
            addnewnodes(N,n,G,gg,edges,k) # add all nodes
            kmax.append(max(k))
        er.append(np.std(kmax))
        km.append(np.mean(kmax))
        print('done')
    plt.rcParams.update({'font.size': 32})
    plt.scatter(np.log(nn), np.log(km),s = 150, marker ='x') 
    xc = 0
    for i in nn:
        plt.scatter(np.log([i]), np.log(km[xc] + er[xc]) , marker ='_', s= 100 , c ='r')
        plt.scatter(np.log([i]), np.log(km[xc] - er[xc]), marker ='_', s =100, c = 'r')
        xc += 1
    ###fit linear to log(l)vs log(sk)
    ##
    x2 = np.array(np.log(nn))

    y2 = np.array(np.log(km))

    A = np.vstack([x2, np.ones(len(x2))]).T

    mm, c = np.linalg.lstsq(A, y2, rcond=None)[0]

    # Polynomial Regression

    def polyfit(x, y, degree):
        
        results = {}

        coeffs = np.polyfit(x, y, degree)

        # Polynomial Coefficients
         
        results['polynomial'] = coeffs.tolist()

        # r-squared
        
        pp = np.poly1d(coeffs)
        
        # fit values, and mean
        
        yhat = pp(x) 
        
        ybar = np.sum(y)/len(y)
        
        ssreg = np.sum((yhat-ybar)**2)
        
        sstot = np.sum((y - ybar)**2)
        
        results['determination'] = ssreg / sstot

        return results

    r_value = polyfit(x2,y2,1)['determination']

    print(r_value, mm)
    ##
    ##    dkt.append(m)

    a = np.arange(min(x2),max(x2),0.1)

    yy = []

    for x in a:

        yy.append(mm*x + c)

    # error on gradient and intercept

    pp, V = np.polyfit(x2, y2, 1, cov=True)

    print("D(1+k-t_s): {} +/- {}".format(pp[0], np.sqrt(V[0][0])))

    ##dkt.append(pp[0]) # value of D(1+k-t_s)
    ##
    ##dkte.append(np.sqrt(V[0][0])) #error on D(1+k-t_s)


    vb = ii.linearinter(a,yy)

    x3 = vb[0]

    y3 = vb[1]

    plt.rcParams.update({'font.size': 32})

    plt.scatter(x3, y3, s = 2, marker = "x", label = 'Linear fit m = '+str(gg))

    plt.rcParams.update({'font.size': 32})

    #plt.scatter(a,yy,s=0.1, label = 'linear fit')
    #plt.legend()
    ##plt.yscale('log')
    ##plt.xscale('log')
    plt.xlabel('ln(N)')
    plt.ylabel('ln($k_1$)')
    plt.rcParams.update({'font.size': 32})
    nnn = np.arange(10,10**6,100)
    for i in nn:
        kmt.append(theoryval(gg,i))
    plt.scatter(np.log(nn), np.log(kmt),s = 50)
    #plt.plot(nnn, kmt,color = 'b', label = 'Theoretical')
    #plt.scatter(nn, km,s = 150, marker = 'x', label = 'Numerical')
##    xc = 0
##    for i in nn:
##        plt.scatter([i], km[xc] + er[xc] , marker ='_', s= 400 , c ='r')
##        plt.scatter([i], km[xc] - er[xc], marker ='_', s =400, c = 'r')
##        xc += 1
##    plt.rcParams.update({'font.size': 32})
##    plt.xlabel('N')
##    plt.ylabel('$k_1$')
plt.legend()
plt.show()
