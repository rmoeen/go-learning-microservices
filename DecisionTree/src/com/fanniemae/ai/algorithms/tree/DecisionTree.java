package com.fanniemae.ai.algorithms.tree;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Set;

public class DecisionTree {

	
	
	String[][] training_data = {
								{"Green", "3", "Apple"},
								{"Yellow", "3", "Apple"},
								{"Red", "1", "Grape"},
								{"Red", "1", "Grape"},
								{"Yellow", "3", "Lemon"}
							};
	
	//HashMap<String, FeatureCount> counts = new HashMap();
	//# Column labels.
	//# These are used only to print the tree.
	String[] header = {"color", "diameter", "label"};

//	#######
//	# Demo:
//	# class_counts(training_data)
//	#######
	
	public HashMap class_counts(String[][] data) {
	    //"""Counts the number of each type of example in a dataset."""
	    
		HashMap<String, FeatureCount> counts = new HashMap();
	    int rowCount = data.length;

	    System.out.println("Gini row count " + rowCount );
	    //if (rowCount <=0)	return null;
	    int columnCount = data[0].length;
	    String lbl = "";
	    FeatureCount f;
	    for (int i=0;i<rowCount;i++) {
	    	lbl = data[i][columnCount-1];
	    	if (!counts.containsKey(lbl)) {
	    		f = new FeatureCount();
	    		f.feature = lbl;
	    		counts.put(lbl, f);
	    	} else {
	    		f = counts.get(lbl); 
	    	}
	    	
	    	f.count++;
	    }

	    return counts;
	}

//	def unique_vals(rows, col):
//	    """Find the unique values for a column in a dataset."""
//	    return set([row[col] for row in rows])
//
//	#######
//	# Demo:
//	# unique_vals(training_data, 0)
//	# unique_vals(training_data, 1)
//	#######

	
	private HashSet uniqueValues(String [][] data, int col) {
		HashSet<String> uniqueVals = new HashSet();

		int rowCount = data.length;
	    //int columnCount = data[1].length;
	    for (int i=0;i<rowCount;i++) {
	    	uniqueVals.add(data[i][col]);
	    	
	    }
		
		return uniqueVals;
	}

	
	private Object build_tree(String[][] data) {
		
		System.out.println("---------------------------------   Splitting data");
		GainQuestion gq = find_best_split(data);
		
		if (gq.gain == 0) {
			System.out.println("+++++++++++++++++++++++++++++     Returning leaf");
			return new Leaf(this, data);
		}
		
		System.out.println("*********************************  Partitioning data");
		Partitioned p = partition(data,gq.q, gq.q.index);
		Object true_branch = null;
		if(p.truevalues != null)
			true_branch = build_tree(arryList2Arry(p.truevalues));
		Object false_branch = null;
		
		if(p.falsevalues != null)
			false_branch = build_tree(arryList2Arry(p.falsevalues));
		
		Decision_Node n = new Decision_Node(true_branch, false_branch, gq.q);
		
		System.out.println(" >>>>>>>>>>>>>>>>>>>>>>>>>>>>>   Returning decision node");
		return n;
	}
	
// STEP 1	
	
//	def build_tree(rows):
//	    """Builds the tree.
//
//	    Rules of recursion: 1) Believe that it works. 2) Start by checking
//	    for the base case (no further information gain). 3) Prepare for
//	    giant stack traces.
//	    """
//
//	    # Try partitioing the dataset on each of the unique attribute,
//	    # calculate the information gain,
//	    # and return the question that produces the highest gain.
//	    gain, question = find_best_split(rows)
//
//	    # Base case: no further info gain
//	    # Since we can ask no further questions,
//	    # we'll return a leaf.
//	    if gain == 0:
//	        return Leaf(rows)
//
//	    # If we reach here, we have found a useful feature / value
//	    # to partition on.
//	    true_rows, false_rows = partition(rows, question)
//
//	    # Recursively build the true branch.
//	    true_branch = build_tree(true_rows)
//
//	    # Recursively build the false branch.
//	    false_branch = build_tree(false_rows)
//
//	    # Return a Question node.
//	    # This records the best feature / value to ask at this point,
//	    # as well as the branches to follow
//	    # dependingo on the answer.
//	    return Decision_Node(question, true_branch, false_branch)
	

/*
 * 	Step 3, find the best question
 * 		3a- GINI
 * 		3b- Determine question
 * 		3c- Partition
 * 
 * 
 * */	
	
	private GainQuestion find_best_split(String[][] data) {
		GainQuestion gq = null;
		
		double best_gain = 0.0; //  # keep track of the best information gain
		Question best_question = null;

		double current_uncertainty = gini(data);
		System.out.println("Current Uncertainty " + current_uncertainty);
		
		int n_features = data[0].length - 1; //  # number of columns
		System.out.println("Features " + n_features);
		
		HashSet colValues;
		//Loop through each feature column
		for(int i=0;i<n_features; i++) {
			
			//Get unique columnn values
			colValues = uniqueValues(data, i);
			System.out.println(colValues);
			
			//Loop through each column value
			Iterator itr = colValues.iterator();
			
			while(itr.hasNext()) {
				//Create a question for each unique value in the column
				Question q = new Question(header[i],(String)itr.next(), i);
				
				//Partition rows based on the column value
				System.out.println("Question " + q.toString());
				Partitioned p = partition(data,q,i);
				System.out.println("True value count " + p.truevalues.size());
				//System.out.println("False value count " + p.falsevalues.size());				
				
				double gain = info_gain(p.truevalues, p.falsevalues, current_uncertainty);
				System.out.println("Gain " + gain);
				if(gain >= best_gain) {
					best_gain = gain;
					best_question = q;
					
				}
			}
			
		}
		
		gq = new GainQuestion();
		gq.gain = best_gain;
		gq.q = best_question;
		return gq;
	}


//	def find_best_split(rows):
//    """Find the best question to ask by iterating over every feature / value
//    and calculating the information gain."""
//    best_gain = 0  # keep track of the best information gain
//    best_question = None  # keep train of the feature / value that produced it
//    current_uncertainty = gini(rows)
//    n_features = len(rows[0]) - 1  # number of columns
//
//    for col in range(n_features):  # for each feature
//
//        values = set([row[col] for row in rows])  # unique values in the column
//
//        for val in values:  # for each value
//
//            question = Question(col, val)
//
//            # try splitting the dataset
//            true_rows, false_rows = partition(rows, question)
//
//            # Skip this split if it doesn't divide the
//            # dataset.
//            if len(true_rows) == 0 or len(false_rows) == 0:
//                continue
//
//            # Calculate the information gain from this split
//            gain = info_gain(true_rows, false_rows, current_uncertainty)
//
//            # You actually can use '>' instead of '>=' here
//            # but I wanted the tree to look a certain way for our
//            # toy dataset.
//            if gain >= best_gain:
//                best_gain, best_question = gain, question
//
//    return best_gain, best_question
//
//#######
//# Demo:
//# Find the best question to ask first for our toy dataset.
//# best_gain, best_question = find_best_split(training_data)
//# FYI: is color == Red is just as good. See the note in the code above
//# where I used '>='.
//#######
	
	
	
	private Partitioned partition(String[][] data, Question q, int colIndex) {
		// TODO Auto-generated method stub
		
		int rows = data.length;
		ArrayList<String[]> true_list = new ArrayList();
		ArrayList<String[]> false_list = new ArrayList();
		for(int i=0; i< rows; i++) {
			
			if (q.match(data[i], colIndex)) {
				true_list.add(data[i]);
			} else {
				false_list.add(data[i]);
			}
		}
		
		Partitioned p = new Partitioned();
		p.falsevalues = false_list;
		p.truevalues = true_list;
		return p;
	}

//	def partition(rows, question):
//	    """Partitions a dataset.
//
//	    For each row in the dataset, check if it matches the question. If
//	    so, add it to 'true rows', otherwise, add it to 'false rows'.
//	    """
//	    true_rows, false_rows = [], []
//	    for row in rows:
//	        if question.match(row):
//	            true_rows.append(row)
//	        else:
//	            false_rows.append(row)
//	    return true_rows, false_rows
//
//
//	#######
//	# Demo:
//	# Let's partition the training data based on whether rows are Red.
//	# true_rows, false_rows = partition(training_data, Question(0, 'Red'))
//	# This will contain all the 'Red' rows.
//	# true_rows
//	# This will contain everything else.
//	# false_rows
//	#######
//	
	
	private double gini(String[][] data) {
		// TODO Auto-generated method stub
		//HashMap<String, FeatureCount> counts = new HashMap();
		
		
		double prob = 0.0;
		double impurity  = 1.0;
		double rows = data.length;
		
		if(rows <= 0) {
			return 0.0;
		}
		
		HashMap<String, FeatureCount> counts = class_counts(data);
		Iterator itr = counts.values().iterator();
		
		while( itr.hasNext()){
			
			FeatureCount f = (FeatureCount)itr.next();
			prob = f.count / rows;
			//System.out.println("Feature " + f.feature + " count " + f.count + " rows " + rows + " prob " + prob);
			impurity -= Math.pow(prob,2);
		}
		
		return impurity;
	}

	
//	def gini(rows):
//	    """Calculate the Gini Impurity for a list of rows.
//
//	    There are a few different ways to do this, I thought this one was
//	    the most concise. See:
//	    https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity
//	    """
//	    counts = class_counts(rows)
//	    impurity = 1
//	    for lbl in counts:
//	        prob_of_lbl = counts[lbl] / float(len(rows))
//	        impurity -=  prob_of_lbl**2
//	    return impurity
//
//
//
//	#######
//	# Demo:
//	# Let's look at some example to understand how Gini Impurity works.
//	#
//	# First, we'll look at a dataset with no mixing.
//	# no_mixing = [['Apple'],
//	#              ['Apple']]
//	# this will return 0
//	# gini(no_mixing)
//	#
//	# Now, we'll look at dataset with a 50:50 apples:oranges ratio
//	# some_mixing = [['Apple'],
//	#               ['Orange']]
//	# this will return 0.5 - meaning, there's a 50% chance of misclassifying
//	# a random example we draw from the dataset.
//	# gini(some_mixing)
//	#
//	# Now, we'll look at a dataset with many different labels
//	# lots_of_mixing = [['Apple'],
//	#                  ['Orange'],
//	#                  ['Grape'],
//	#                  ['Grapefruit'],
//	#                  ['Blueberry']]
//	# This will return 0.8
//	# gini(lots_of_mixing)
//	#######
	

	private double info_gain(ArrayList truevalues, ArrayList falsevalues, double current_uncertainty) {
		// TODO Auto-generated method stub

		double gain = 0.0;
		gain = (truevalues.size() * 1.0) / ((truevalues.size()*1.0) + (falsevalues.size() *1.0));
		System.out.println("p = " + gain);
		double giniLeft  = gini(arryList2Arry(truevalues));
		System.out.println("gini left " + giniLeft);
		gain = current_uncertainty - gain*(giniLeft) - (1 - gain) * gini(arryList2Arry(falsevalues));
		
		return gain;
	}
	
	
	
	
private String[][] arryList2Arry(ArrayList valList) {
	
	int rowcount = valList.size();
	int colCount = header.length;
	
	String[][] data = new String[rowcount][colCount];
	
	Iterator itr = valList.iterator();
	int i=0;
	while(itr.hasNext()) {
		data[i++] = (String[])itr.next();
	}
	
	// TODO Auto-generated method stub
	return data;
}

//	def info_gain(left, right, current_uncertainty):
//	    """Information Gain.
//
//	    The uncertainty of the starting node, minus the weighted impurity of
//	    two child nodes.
//	    """
//	    p = float(len(left)) / (len(left) + len(right))
//	    return current_uncertainty - p * gini(left) - (1 - p) * gini(right)
	



//def print_tree(node, spacing=""):
//    """World's most elegant tree printing function."""
//
//    # Base case: we've reached a leaf
//    if isinstance(node, Leaf):
//        print (spacing + "Predict", node.predictions)
//        return
//
//    # Print the question at this node
//    print (spacing + str(node.question))
//
//    # Call this function recursively on the true branch
//    print (spacing + '--> True:')
//    print_tree(node.true_branch, spacing + "  ")
//
//    # Call this function recursively on the false branch
//    print (spacing + '--> False:')
//    print_tree(node.false_branch, spacing + "  ")
	
	private void print_tree(Object node, String spacing) {
		
		if(node instanceof Leaf) {
			Leaf lf = (Leaf)node;
			//System.out.println(
			System.out.println(spacing + extractFeatures(lf.predictions));
			return;
		}
		
		Decision_Node n = (Decision_Node)node;
		System.out.println(spacing + " " + n.q.toString());
		System.out.println(spacing + " " + "--> True");
		print_tree(n.true_branch, spacing + " ");
		
		System.out.println(spacing + " " + "--> False ");
		print_tree(n.false_branch, spacing + " ");
		
	}
	
	private String extractFeatures(HashMap mp) {
		Iterator itr = mp.values().iterator();
		String vals = "";
		String sep = "";
		while(itr.hasNext()) {
			FeatureCount f = (FeatureCount)itr.next();
			//System.out.println(spacing + " " + f.feature);
			vals = vals + sep + f.feature;
			sep = ",";
			
		}
		return  vals;
	}

	private String Array2String(String[] vals) {
		String output = "";
		
		for(int i=0; i<vals.length; i++) {
			output = output + ", " + vals [i];
		}
		
		return output;
	}
	
//	def classify(row, node):
//	    """See the 'rules of recursion' above."""
//
//	    # Base case: we've reached a leaf
//	    if isinstance(node, Leaf):
//	        return node.predictions
//
//	    # Decide whether to follow the true-branch or the false-branch.
//	    # Compare the feature / value stored in the node,
//	    # to the example we're considering.
//	    if node.question.match(row):
//	        return classify(row, node.true_branch)
//	    else:
//	        return classify(row, node.false_branch)
	
	private Object classify(String[] row, Object node) {
		Object retnode = null;
				
		if(node instanceof Leaf) {
			Leaf lf = (Leaf)node;
			return lf.predictions;
		}
				
		Decision_Node n = (Decision_Node)node;
		if(n.q.match(row, n.q.index)) {
			return classify(row, n.true_branch);
		} else {
			return classify(row, n.false_branch);
		}
		
	}
	
	
	
	
	/*
	 * 
	 * 
	 * """Code to accompany Machine Learning Recipes #8.
def is_numeric(value):
    """Test if a value is numeric."""
    return isinstance(value, int) or isinstance(value, float)

#######
# Demo:
# is_numeric(7)
# is_numeric("Red")
#######







#######
# Demo:
# Calculate the uncertainy of our training data.
# current_uncertainty = gini(training_data)
#
# How much information do we gain by partioning on 'Green'?
# true_rows, false_rows = partition(training_data, Question(0, 'Green'))
# info_gain(true_rows, false_rows, current_uncertainty)
#
# What about if we partioned on 'Red' instead?
# true_rows, false_rows = partition(training_data, Question(0,'Red'))
# info_gain(true_rows, false_rows, current_uncertainty)
#
# It looks like we learned more using 'Red' (0.37), than 'Green' (0.14).
# Why? Look at the different splits that result, and see which one
# looks more 'unmixed' to you.
# true_rows, false_rows = partition(training_data, Question(0,'Red'))
#
# Here, the true_rows contain only 'Grapes'.
# true_rows
#
# And the false rows contain two types of fruit. Not too bad.
# false_rows
#
# On the other hand, partitioning by Green doesn't help so much.
# true_rows, false_rows = partition(training_data, Question(0,'Green'))
#
# We've isolated one apple in the true rows.
# true_rows
#
# But, the false-rows are badly mixed up.
# false_rows
#######



class Leaf:
    """A Leaf node classifies data.

    This holds a dictionary of class (e.g., "Apple") -> number of times
    it appears in the rows from the training data that reach this leaf.
    """

    def __init__(self, rows):
        self.predictions = class_counts(rows)


class Decision_Node:
    """A Decision Node asks a question.

    This holds a reference to the question, and to the two child nodes.
    """

    def __init__(self,
                 question,
                 true_branch,
                 false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch










#######
# Demo:
# The tree predicts the 1st row of our
# training data is an apple with confidence 1.
# my_tree = build_tree(training_data)
# classify(training_data[0], my_tree)
#######

def print_leaf(counts):
    """A nicer way to print the predictions at a leaf."""
    total = sum(counts.values()) * 1.0
    probs = {}
    for lbl in counts.keys():
        probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"
    return probs


#######
# Demo:
# Printing that a bit nicer
# print_leaf(classify(training_data[0], my_tree))
#######

#######
# Demo:
# On the second example, the confidence is lower
# print_leaf(classify(training_data[1], my_tree))
#######

if __name__ == '__main__':

    my_tree = build_tree(training_data)

    print_tree(my_tree)

    # Evaluate
    testing_data = [
        ['Green', 3, 'Apple'],
        ['Yellow', 4, 'Apple'],
        ['Red', 2, 'Grape'],
        ['Red', 1, 'Grape'],
        ['Yellow', 3, 'Lemon'],
    ]

    for row in testing_data:
        print ("Actual: %s. Predicted: %s" %
               (row[-1], print_leaf(classify(row, my_tree))))

# Next steps
# - add support for missing (or unseen) attributes
# - prune the tree to prevent overfitting
# - add support for regression
	 * 
	 * */

	
	public void runClassification() {
		//class_counts(training_data);
//		Iterator itr = counts.values().iterator();
//		
//		while(itr.hasNext()) {
//			FeatureCount f = (FeatureCount)itr.next();
//			System.out.println("Feature " + f.feature + " count " + f.count);
//		}
		//System.out.println("Impurity " + gini(training_data));
		//find_best_split(training_data);
		
		Object myTree = build_tree(training_data);
		print_tree(myTree, "");
		System.out.println("Testing with " + Array2String(training_data[1]));
		HashMap predictions = (HashMap)classify(training_data[1], myTree);
		
		
		
		System.out.print("Expected " + training_data[1][2] + " Actual " + extractFeatures(predictions));
		
	}	
	
	
}

class GainQuestion{

	double gain = 0.0;
	Question q;
	
	
	
}

class FeatureCount {
	int count = 0;
	String feature = "";
}

class Partitioned{
	ArrayList truevalues;
	ArrayList falsevalues;
}

class Leaf{

	HashMap predictions = null;
	
	public Leaf(DecisionTree tree, String[][] data) {
		predictions = tree.class_counts(data);
	}
	
}

class Decision_Node{
	Object true_branch;		//Supposed to store another decision node
	Object false_branch;	//Supposed to store another decision node
	Question q;
	
	public Decision_Node(Object tBranch, Object fBranch, Question q) {
		this.q = q;
		true_branch = tBranch;
		false_branch = fBranch;
	}
	
	
}