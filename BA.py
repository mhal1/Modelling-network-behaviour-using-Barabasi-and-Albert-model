#BA model
import networkx as nx
import random
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import os
import logbin as lb
###############################################################################
# Will only print out information if N is below this number, use for testing
Nprint=20
# set seed to a number if you want to fix the random number seed for testing
# Otherwise set seed = None if you want true random numbers
#seed=0
seed = None  
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
n_add = 100000 #add this many nodes
#add nodes to initial graph
#degree of each nodes
nodes = [x for x in range(N+n_add+1)]
k = len(nodes)*[0]
def addnewnodes(N,n_add,G,m,edges,k):
    for i in range(N,n_add+N+1):
        G.add_node(i)
        # Now add edges    
        # avoid self-edges
        for h in range(m):
            addedge(G,edges,i,k)
def logbinn(k,scale,N,n_add,m):
    #need to do checks for this 
    #need to log bin degrees 
    #theory
    pt = []
    kt = np.arange(1,2*(n_add+N+1),1)
    pt = theoryval(m,kt)
    #plt.figure()
    plt.rcParams.update({'font.size': 32})
    plt.plot(np.log(kt),np.log(pt), color = 'r', label = 'Theory, ' + str(m))
    #numerical
    #for i in nx.degree(G):
    #    k.append(i[1])
    #using edge list
    #form a list of nodes - this is the number of nodes plus the number you add
    nodes = [x for x in range(N+n_add+1)]
    #print(nodes)
    #k = len(nodes)*[0]
    #print('kinit ',k)
    #print(len(nodes),(N+n_add)) #these two should be equal, they are so good
    #print('k after',k)  
    #subtract mink
    knew = np.array(k) - np.array(len(k)*[min(k)])
    #print(min(k),m)
    #print(edges)
    p = lb.logbin(knew,scale=1)
    p2 = lb.logbin(knew,scale=scale)
    #add back kmin to k values
    kf1 = np.array(p[0]) + np.array(len(p[0])*[min(k)])
    kf2 = np.array(p2[0]) + np.array(len(p2[0])*[min(k)]) 
    return p,p2,kf1,kf2 #return the k distribution
def theoryval(m,k):
    pt = []
    a = 2*m*(m+1)
    for i in k:
        b = i*(i+1)*(i+2)
        pt.append(a/b)
    return pt # returns theory prob for each k
def chisqttest(m,p2): # mention why you did or didnt use the tests
    #for the statistical comparison
    # compare numerical data to theory
    # numerical only gives a fixed number of data points for certian k
    # plug this value of k into the theory and then use the kai squared test to compare answers.
    t = theoryval(m,p2[0])
    n = p2[1]
    statistic, pval = stats.chisquare( n, t)
    print('stat = ',statistic)
    print('p-value = ',pval)
    critval = stats.chi2.ppf(q=0.95,df=1) # list of data set is df, check rwsidue qq plot- tell you if it as gaussian or not buuilt in func.
    print("crit val = ",critval)
    #plt.figure()
    #shift back logbin data
    plt.scatter(np.log(p2[0]),np.log(p2[1]), label = 'Numerical ' + str(m))
    plt.scatter(np.log(p2[0]),np.log(t), label = 'Theory ' + str(m))
    plt.legend()
    plt.show()
#list of m for same N
# need to do repeats and append before log binning!
repeats = 30
for x in [2,4,8,16,32,64,128]:
    kreap = []
    for i in range(repeats):
        edges = []
        k = len(nodes)*[0] # initial list of nodes with zero degree
        G = initG(N,edges,2,k) # create initial graph
        p,p2,kf1,kf2 = [],[],[],[]
        addnewnodes(N,n_add,G,x,edges,k) # add all nodes
        kreap.extend(k)
        
    p,p2,kf1,kf2 = logbinn(kreap,1.35,N,n_add,x) # outputs log binned data with scaling.
    plt.rcParams.update({'font.size': 32})
    #plt.scatter(np.log(kf1),np.log(p[1]),s = 10, label = 'No scaling logbin, m ='+str(x))
    plt.scatter(np.log(kf2),np.log(p2[1]),s=150, label = 'Log binned, m = '+str(x) )
    #plt.legend()
    print('done ' + str(x)) 
    #plt.show()
    #chisqttest(x,p2)
plt.rcParams.update({'font.size': 32})
plt.xlabel('ln($k$)')
plt.ylabel('ln($p_{\infty}(k)$)')
plt.show()
# do repeats for each m value and append each repeat then log bin.
# should help to get finite sized stuff
