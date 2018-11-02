from util import entropy, information_gain, partition_classes
import numpy as np 
import scipy as sp
import ast

class DecisionTree(object):
    def __init__(self):
        # Initializing the tree as an empty dictionary or list, as preferred
        self.tree = []
        #self.tree = {}
        pass

    def learn(self, X, y):
        # TODO: Train the decision tree (self.tree) using the the sample X and labels y
        # You will have to make use of the functions in utils.py to train the tree
        
        # One possible way of implementing the tree:
        #    Each node in self.tree could be in the form of a dictionary:
        #       https://docs.python.org/2/library/stdtypes.html#mapping-types-dict
        #    For example, a non-leaf node with two children can have a 'left' key and  a 
        #    'right' key. You can add more keys which might help in classification
        #    (eg. split attribute and split value)
               
        root = {
                    'attribute_index' : None,
                    'value' : None,
                    'left': None,
                    'right': None
                }
        
        if entropy(y) < .2:                     ##Reasonable bound/cut off based on entropy to prevent overfitting
            return sp.stats.mode(y)[0][0]
        
        else:
            info_gain = 0
            split_attribute_index = 0
            split_value = X[0][split_attribute_index]
            
            for item in X: 
                for i in range(len(item)):
                    if information_gain(y, [partition_classes(X, y, i, item[i])[2], partition_classes(X, y, i, item[i])[3]]) > info_gain:
                        info_gain = information_gain(y, [partition_classes(X, y, i, item[i])[2], partition_classes(X, y, i, item[i])[3]]) ##Base attribute and split value on combination that maximizes info gain
                        split_attribute_index = i
                        split_value = item[i]  
            
            root['attribute_index'] = split_attribute_index
            root['value'] = split_value
            root['left'] = self.learn(partition_classes(X, y, split_attribute_index, split_value)[0], partition_classes(X, y, split_attribute_index, split_value)[2])
            root['right'] = self.learn(partition_classes(X, y, split_attribute_index, split_value)[1], partition_classes(X, y, split_attribute_index, split_value)[3])
            
            self.tree.insert(0, root)
        
            return root
    
    def classify(self, record):
        node = self.tree[0]
        keep_going = True
        label = 0
        
        while keep_going:
            if (type(record[node['attribute_index']]) is int and record[node['attribute_index']] <= node['value']):
                if node['left'] not in [0,1]:
                    node = node['left']
                else:
                    label = node['left']
                    keep_going = False
                    
            elif (type(record[node['attribute_index']]) is int and record[node['attribute_index']] > node['value']):
                if node['right'] not in [0,1]:
                    node = node['right']
                else:
                    label = node['right']
                    keep_going = False
            
            elif (type(record[node['attribute_index']]) is not int and record[node['attribute_index']] == node['value']):
                if node['left'] not in [0,1]:
                    node = node['left']
                else:
                    label = node['left']
                    keep_going = False
            
            elif (type(record[node['attribute_index']]) is not int and record[node['attribute_index']] != node['value']):
                if node['right'] not in [0,1]:
                    node = node['right']
                else:
                    label = node['right']
                    keep_going = False
        
        return label