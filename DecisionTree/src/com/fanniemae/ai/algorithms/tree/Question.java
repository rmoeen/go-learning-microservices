package com.fanniemae.ai.algorithms.tree;

public class Question {

	String column = "";	//Must be a column name
	String value = "";	//Value stored in that column
	int index = -1;
	
	public Question(String col, String val, int colIndex) {
		System.out.println("Creating a question " + val + " with " + colIndex);
		column = col;
		value = val;
		index = colIndex;
	}
	
	public boolean match(String[] compareVal, int colIndex) {
		System.out.println("Comparing " + value + " with " + compareVal[colIndex]);
		return value.equalsIgnoreCase(compareVal[colIndex]);
	}
	@Override
	public String toString() {
		return "Is " + column + "==" + value; 
	}
	
	
//	class Question:
//	    """A Question is used to partition a dataset.
//
//	    This class just records a 'column number' (e.g., 0 for Color) and a
//	    'column value' (e.g., Green). The 'match' method is used to compare
//	    the feature value in an example to the feature value stored in the
//	    question. See the demo below.
//	    """
//
//	    def __init__(self, column, value):
//	        self.column = column
//	        self.value = value
//
//	    def match(self, example):
//	        # Compare the feature value in an example to the
//	        # feature value in this question.
//	        val = example[self.column]
//	        if is_numeric(val):
//	            return val >= self.value
//	        else:
//	            return val == self.value
//
//	    def __repr__(self):
//	        # This is just a helper method to print
//	        # the question in a readable format.
//	        condition = "=="
//	        if is_numeric(self.value):
//	            condition = ">="
//	        return "Is %s %s %s?" % (
//	            header[self.column], condition, str(self.value))
//
//	#######
//	# Demo:
//	# Let's write a question for a numeric attribute
//	# Question(1, 3)
//	# How about one for a categorical attribute
//	# q = Question(0, 'Green')
//	# Let's pick an example from the training set...
//	# example = training_data[0]
//	# ... and see if it matches the question
//	# q.match(example)
//	#######
	
}
