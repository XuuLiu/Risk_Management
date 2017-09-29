Information_value.py is a class file to calculate iv and woe. To use the class, you need to change the file rote to yours in python file Data_random_splt.py.

Step 1
Data_random_splt.py：
1. Pick 840 bad men and 3360 good men randomly.(random seed=1000)
2. Calculate the woe and IV for all variables, remove 7 variables whose IV<0.03.


Step 2
woe_splt_vX.py is file that used to grouping variables in Use_data_raw.npy.
There is a class file: woe_splt_class.py, it can be used to grouping variables automatically.
Inside the class:
1. For discrete variable, using hierarchical clustering. 
    Max group number is 8.
    IV before grouping/ IV after grouping >0.8
2. For continuous variable, using clustering considered sequence order. 
    Group all values into 50 groups according to their quantiles;
    Combine all these groups while IV before grouping/ IV after grouping >0.8, untile group number is less than 8.
To use this class, I recommend that you should know types of variables. use str_splt for discrete variable, and flt_splt for continuous variable. 
Index of variables should be changed by your situation.


Step 3 
transform_woe.py use the woe grouping result to update the model data.


Step 4
logic.py this part contains two ways to deal with model data.
1. Turn grouping values into dump variable.
2. Use woe instead of grouping info.
After training model, use coefficient to calculate score for test data.

