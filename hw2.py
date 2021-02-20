#!/usr/bin/env python
# coding: utf-8

# In[552]:


# candidate splits (j,c) for numeric features should use a threshold c in feature dimension j in the form of x_j>=c

# c should be on values of that dimension present in the training data; i.e. the threshold is on training points,
# not in between training points. you may enumerate all features, and for each feature, use all possible values for
# that dimension

# you may skip those candidate splits with zero split information (i.e. the entropy of the split), and continue the
# enumeration

# the left branch of such a split is the "then" branch, and the right branch is "else"

# splits should be chosen using information gain ratio. if there is a tie you may break it arbitrarily

# the stopping criteria (for making a node into a leaf) are that:
# the node is empty, or
# all splits have zero gain ratio (if the entropy of the split is non-zero), or
# the entropy of any candidate split is zero

# to simplify, whenever there is no majority class in a leaf, let it predict y=1

import numpy as np
import math
import matplotlib.pyplot as plt
import random


# In[558]:


# read input datasets from the provided files
# data values are of form: (x1, x2, y)

D1data = np.loadtxt("D1.txt")
D2data = np.loadtxt("D2.txt")
D3data = np.loadtxt("D3leaves.txt")
Dbigdata = np.loadtxt("Dbig.txt")
Drunsdata = np.loadtxt("Druns.txt")
testdata = np.loadtxt("test.txt")


# In[518]:


# entropy: H(Y) = -sum_y[P(y)log_2(P(y))]
def entropy(Y):
    unique, count = np.unique(Y, return_counts=True, axis=0)
    prob = count/len(Y)
    ent = np.sum(-prob*np.log2(prob))
    return ent

# joint entropy: H(Y;X) = -sum_x[sum_y[P(x,y)log_2(P(x,y))]]
def joint_entropy(Y,X):
    YX = np.c_[Y,X]
    return entropy(YX)

# conditional entropy: H(Y|X) = sum_x[P(X=x)H(Y|X=x)]
def cond_entropy(Y,X):
    return joint_entropy(Y,X) - entropy(X)

# information gain: I(Y;X) = H(Y) - H(Y|X)
def inf_gain(Y,X):
    return entropy(Y) - cond_entropy(Y,X)

# information gain ratio: I(Y;X) / H(X)
def inf_gain_ratio(Y,X):
    return inf_gain(Y,X) / entropy(X)


# In[519]:


# a node in the tree with 2 child nodes
class node:
    left = None
    right = None
    index = -1
    threshold = -1
    label = -1
    lchild = None
    rchild = None


# In[520]:


# main method called recursively to build decision tree
def make_subtree(data,parent_if_left=None,parent_if_right=None):
    split_index, max_gain_ratio, split_threshold = predict_split(data)
    new_node = node()
    if parent_if_left != None:
        parent_if_left.lchild = new_node
    if parent_if_right != None:
        parent_if_right.rchild = new_node
    new_node.index = split_index
    new_node.threshold = split_threshold
    if (split_index == -1 or max_gain_ratio == 0):
        new_node.label = get_label(data)
        return new_node
    left_data, right_data = get_subtrees(data, split_index, split_threshold)
    new_node.left = make_subtree(left_data,new_node,None)
    new_node.right = make_subtree(right_data,None,new_node)
    return new_node


# In[521]:


# given the current data set, find the index of the optimal split,
# the optimal split value, and the optimal information gain ratio
def predict_split(data):
    
    # calculate avg inf gain
    avgGainNum = 0
    avgGainDenom = 0
    for col in (0,1):
        for xval in np.unique(data[:,col]):
            S = np.array([])
            for x in data[:,col]:
                if x >= xval:
                    S = np.append(S,1)
                else:
                    S = np.append(S,0)
            if entropy(S) == 0:
                continue
            gain = inf_gain(data[:,2], S)
            avgGainNum += gain
            avgGainDenom += 1
    if avgGainDenom != 0:
        avgGain = avgGainNum / avgGainDenom
    else:
        avgGain = -1

    #return best split
    maxgain = -math.inf
    maxval = -1
    maxcol = -1
    for col in (0,1):
        for xval in np.unique(data[:,col]):
            S = np.array([])
            for x in data[:,col]:
                if x >= xval:
                    S = np.append(S,1)
                else:
                    S = np.append(S,0)
            if entropy(S) == 0:
                continue
            if inf_gain(data[:,2],S) < avgGain:
                continue
            gain = inf_gain_ratio(data[:,2], S)
            if gain > maxgain:
                maxgain = gain
                maxval = xval
                maxcol = col

    return maxcol, maxgain, maxval


# In[522]:


# given a dataset and a split return the datasets for both
# the left and right subtrees
def get_subtrees(data, index, threshold):
    left = np.empty((0,3))
    right = np.empty((0,3))
    i = 0
    for x in data[:,index]:
        if x >= threshold:
            left = np.append(left, [[data[i,0], data[i,1], data[i,2]]], axis=0)
        else:
            right = np.append(right, [[data[i,0], data[i,1], data[i,2]]], axis=0)
        i += 1
    return left, right


# In[523]:


# given a dataset return the predicted label
def get_label(data):
    count0 = 0
    count1 = 0
    for y in data[:,2]:
        if y == 1:
            count1 += 1
        else:
            count0 += 1
    if count1 >= count0:
        return 1
    else:
        return 0


# In[524]:


# print a node for analysis of the tree by humans
def print_node(node,tabs=0):
    for i in range(tabs):
        print("  ",end='')
    if node.label != -1:
        print("predict",node.label)
    else:
        print("if x",node.index," >= ",node.threshold,sep='')


# In[525]:


# print the whole tree for human analysis
def print_tree(root_node,tabs=0):
    print_node(root_node,tabs)
    tabs +=1
    if root_node.lchild != None:
        print_tree(root_node.lchild,tabs)
    if root_node.rchild != None:
        print_tree(root_node.rchild,tabs)


# In[526]:


root = make_subtree(testdata)
print_tree(root)


# In[541]:


def plot_data(data):
    i = 0
    onesx = np.empty((0,2))
    onesy = np.empty((0,2))
    zerosx = np.empty((0,2))
    zerosy = np.empty((0,2))
    for y in data[:,2]:
        if y == 1:
            onesx = np.append(onesx, data[i,0])
            onesy = np.append(onesy, data[i,1])
        else:
            zerosx = np.append(zerosx, data[i,0])
            zerosy = np.append(zerosy, data[i,1])
        i += 1
    plt.plot(onesx, onesy, 'rd', label="y=1")
    plt.plot(zerosx, zerosy, 'bs', label="y=0")
    plt.legend()
    #plt.xlim(0,3)
    #plt.ylim(0,3)


# In[528]:


plot_data(testdata)


# In[535]:


# mostly copied from predict_split
# print all possible splits and the inf gain ratio for each
def list_candidates(data):
    for col in (0,1):
        for xval in np.unique(data[:,col]):
            S = np.array([])
            for x in data[:,col]:
                if x >= xval:
                    S = np.append(S,1)
                else:
                    S = np.append(S,0)
            if entropy(S) == 0:
                print("xval,col",xval,col)
                print("inf gain (0 entropy): ", inf_gain(data[:,2],S))
            else:
                print("xval,col",xval,col)
                print("inf gain ratio: ", inf_gain_ratio(data[:,2], S))


# In[536]:


list_candidates(Drunsdata)


# In[538]:


root = make_subtree(D1data)
print_tree(root)


# In[539]:


root = make_subtree(D2data)
print_tree(root)


# In[542]:


plot_data(D1data)


# In[543]:


plot_data(D2data)


# In[550]:


# input is a series of (x0,x1,y_act) values
# output includes predicted y val: (x0,x1,y_est)
def predict(data):
    root = make_subtree(data)
    for i in range(len(data)):
        curnode = root
        while curnode.lchild != None:
            if data[i,curnode.index] >= curnode.threshold:
                curnode = curnode.lchild
            else:
                curnode = curnode.rchild
        data[i,2] = curnode.label
    return data


# In[551]:


data = predict(D1data)
plot_data(data)


# In[548]:


data = predict(D2data)
plot_data(data)


# In[663]:


# generate the training sets data_# and the test set data_test

data = Dbigdata
train_indices = random.sample(range(10000),8192)
data_test = np.empty((0,3))
data_8192 = np.empty((0,3))
data_2048 = np.empty((0,3))
data_512 = np.empty((0,3))
data_128 = np.empty((0,3))
data_32 = np.empty((0,3))

for i in train_indices:
    data_8192 = np.append(data_8192, [[data[i,0], data[i,1], data[i,2]]], axis=0)
    
for i in range(10000):
    if i not in train_indices:
        data_test = np.append(data_test, [[data[i,0], data[i,1], data[i,2]]], axis=0)

for i in range(2048):
    data_2048 = np.append(data_2048, [[data_8192[i,0], data_8192[i,1], data_8192[i,2]]], axis=0)
    if i < 512:
        data_512 = np.append(data_512, [[data_8192[i,0], data_8192[i,1], data_8192[i,2]]], axis=0)
    if i < 128:
        data_128 = np.append(data_128, [[data_8192[i,0], data_8192[i,1], data_8192[i,2]]], axis=0)
    if i < 32:
        data_32 = np.append(data_32, [[data_8192[i,0], data_8192[i,1], data_8192[i,2]]], axis=0)


# In[666]:


# input is a series of (x0,x1,y_act) values
# output includes predicted y val: (x0,x1,y_est)
# this version allows you to input the decision tree and prints err%
def predict(data,root):
    count_total = 0
    count_wrong = 0
    for i in range(len(data)):
        curnode = root
        while curnode.lchild != None:
            if data[i,curnode.index] >= curnode.threshold:
                curnode = curnode.lchild
            else:
                curnode = curnode.rchild
        if data[i,2] != curnode.label:
            count_wrong +=1
        count_total +=1
    return count_wrong/count_total


# In[667]:


# print out the answers
#print("data_8192: ",predict(data_test,root_8192))
#print("data_2048: ",predict(data_test,root_2048))
#print("data_512: ",predict(data_test,root_512))
#print("data_128: ",predict(data_test,root_128))
#print("data_32: ",predict(data_test,root_32))


# In[668]:


# generate decision trees for each training set
#root_8192 = make_subtree(data_8192)
#root_2048 = make_subtree(data_2048)
#root_512 = make_subtree(data_512)
#root_128 = make_subtree(data_128)
root_32 = make_subtree(data_32)


# In[669]:


root_32 = make_subtree(data_32)
print("data_32: ",predict(data_test,root_32))


# In[670]:


root_128 = make_subtree(data_128)
print("data_128: ",predict(data_test,root_128))


# In[671]:


root_512 = make_subtree(data_512)
print("data_512: ",predict(data_test,root_512))


# In[672]:


root_2048 = make_subtree(data_2048)
print("data_2048: ",predict(data_test,root_2048))


# In[684]:


root_8192 = make_subtree(data_8192)
print("data_8192: ",predict(data_test,root_8192))


# In[680]:


def count_nodes(root,count=0):
    count += 1
    if root.lchild != None:
        count = count_nodes(root.lchild,count)
    if root.rchild != None:
        count = count_nodes(root.rchild,count)
    return count


# In[686]:


print("root_32",count_nodes(root_32))
print("root_128",count_nodes(root_128))
print("root_512",count_nodes(root_512))
print("root_2048",count_nodes(root_2048))
print("root_8192",count_nodes(root_8192))


# In[689]:


n = [7, 17, 55, 107, 189]
err = [0.07577433628318585, 0.053650442477876106, 0.03484513274336283, 0.020464601769911505, 0.00331858407079646]
plt.xlabel("number of tree nodes")
plt.ylabel("error")
plt.plot(n, err, 'rd')


# In[691]:


# input is a series of (x0,x1,y_act) values
# output includes predicted y val: (x0,x1,y_est)
# but use provided root
def predict2(data,root):
    for i in range(len(data)):
        curnode = root
        while curnode.lchild != None:
            if data[i,curnode.index] >= curnode.threshold:
                curnode = curnode.lchild
            else:
                curnode = curnode.rchild
        data[i,2] = curnode.label
    return data


# In[692]:


data = predict2(data_test,root_32)
plot_data(data)


# In[693]:


data = predict2(data_test,root_128)
plot_data(data)


# In[694]:


data = predict2(data_test,root_512)
plot_data(data)


# In[695]:


data = predict2(data_test,root_2048)
plot_data(data)


# In[696]:


data = predict2(data_test,root_8192)
plot_data(data)


# In[ ]:




