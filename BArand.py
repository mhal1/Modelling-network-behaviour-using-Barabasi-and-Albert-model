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
def initG(N,edges,m_o,k,nodes):
    # initialise graph 
    a=nx.Graph()
    # add nodes
    for i in range(N):
        a.add_node(i)
        nodes.append(i)
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
def addedge(Graph, edges,nod,k,nodes):
    # want to add one end of edge to new vertex
    # then add other edge to an existing vertex preferentially
    #now insert edges
    ch = random.choice(nodes) # need to make this a list of current nodes at this time, not all total nodes
    #print(ch)
    # aviod self edges
    if ch != nod:
    #then pick a vertex of this edge at random to join to the new vertex
        Graph.add_edge(nod,ch)
        #shw(G)
        #print('edge added ',nod,v)
        edges.append([nod,ch])
        edges.append([ch,nod])
        k[nod] += 1
        k[ch] += 1
    elif ch == nod:
        while ch == nod:
            ch = random.choice(nodes)
        Graph.add_edge(nod,ch)
        #shw(G)
        k[nod] += 1
        k[ch] += 1
        #print('edge added ',nod,v)
        edges.append([nod,ch])
        edges.append([ch,nod])
# need parameters
#N total number of inital vertices
N = 3
#initialise graph with two connected nodes
#plt.figure()
#nx.draw_circular(G)
#plt.show()
#edge that has been added shouldnt be added again
n_add = 10000 #add this many nodes
#add nodes to initial graph
#degree of each nodes
nodestotal = [x for x in range(N+n_add+1)]
def addnewnodes(N,n_add,G,m,edges,k,nodes):
    for i in range(N,n_add+N+1):
        G.add_node(i)
        nodes.append(i)
        # Now add edges    
        # avoid self-edges
        for h in range(m):
            addedge(G,edges,i,k,nodes)
            #shw(G)
def logbinn(k,scale,N,n_add,m):
    #need to do checks for this 
    #need to log bin degrees 
    #theory
    pt = []
    kt = np.arange(m,(n_add+N+1),1)
    pt = theoryval(m,kt)
    #plt.figure()
    #plt.rcParams.update({'font.size': 32})
    #plt.scatter(np.log(kt),np.log(pt), color = 'r', label = 'Theory, ' + str(m))
    #numerical
    #form a list of nodes - this is the number of nodes plus the number you add
    nodes = [x for x in range(N+n_add+1)] 
    #subtract mink
    knew = np.array(k) - np.array(len(k)*[min(k)])
    #print(min(k),m)
    #print(edges)
    p = lb.logbin(knew,scale=scale)
    #p2 = lb.logbin(knew,scale=scale)
    #add back kmin to k values
    kf1 = np.array(p[0]) + np.array(len(p[0])*[min(k)])
    #kf2 = np.array(p2[0]) + np.array(len(p2[0])*[min(k)]) 
    return p,kf1 #return the k distribution
def theoryval(m,kt):
    pt = []
    for i in kt:
        a = m**(i-m)
        b = (m+1)**(i-m+1)
        pt.append(a/b)
    return pt # returns theory prob for each k
def chisqttest(m,p2,kf): # mention why you did or didnt use the tests
    #for the statistical comparison
    # compare numerical data to theory
    # numerical only gives a fixed number of data points for certian k
    # plug this value of k into the theory and then use the kai squared test to compare answers.
    t = theoryval(m,kf)[:15]
    n = p2[1][:15]
    statistic, pval = stats.chisquare( n, t)
    print('stat = ',statistic,", for m = ",m)
    print('p-value = ',pval)
    critval = stats.chi2.ppf(q=0.95,df=len(kf)-1) # list of data set is df, check rwsidue qq plot- tell you if it as gaussian or not buuilt in func.
    print("crit val = ",critval)
    #plt.figure()
    #shift back logbin data
    #plt.scatter(np.log(kf),np.log(p2[1]), label = 'Numerical ' + str(m))
    #plt.scatter(np.log(kf[:15]),np.log(t), label = 'Theory ' + str(m))
    #plt.legend()
#list of m for same N
# need to do repeats and append before log binning!
repeats = 10
for x in [2,4,8,16,32,64]:
    kreap = []
    for i in range(repeats):
        edges = []
        nodes = []
        k = len(nodestotal)*[0] # initial list of nodes with zero degree
        G = initG(N,edges,2,k,nodes) # create initial graph
        addnewnodes(N,n_add,G,x,edges,k,nodes) # add all nodes
        kreap.extend(k)
    pt = []
    kt = np.arange(x,(n_add+N+1))
    pt = theoryval(x,kreap)
    plt.rcParams.update({'font.size': 32})
    #plt.scatter(np.log(kreap),np.log(pt),marker = '.', color = 'black')
    #p,p2,kf1,kf2 = logbinn(kreap,1.35,N,n_add,x) # outputs log binned data with scaling.
    #plt.scatter(np.log(kf1),np.log(p[1]),s = 10, label = 'No scaling logbin, m ='+str(x))
    #plt.scatter(np.log(kf2),np.log(p2[1]),s=150, label = 'Log binned, m = '+str(x) )
    p,kf = logbinn(kreap,1.35,N,n_add,x) # outputs log binned data with scaling.
    #p = lb.logbin(k,scale=1.35)
    print(len(p[0]),len(p[1]))
    #plt.scatter(np.log(kf),np.log(p[1]),s = 350,marker = '.', label = 'm ='+str(x))
    plt.legend()
    print('done ' + str(x)) 
    #plt.show()
    chisqttest(x,p,kf)
plt.rcParams.update({'font.size': 32})
plt.xlabel('ln($k$)')
plt.ylabel('ln($p_{\infty}(k)$)')
plt.show()
# do repeats for each m value and append each repeat then log bin.
# should help to get finite sized stuff
