Information_value.py is a class file to calculate iv and woe. To use the class, you need to change the file rote to yours in python file Data_random_splt.py.

Step 1
Data_random_splt.py：
1. Pick 840 bad men and 3360 good men randomly.(random seed=1000)
2. Calculate the woe and IV for all variances, remove 7 variance whose IV<0.03.

Step 2
woe_splt_vX.py is file that used to grouping variance in Use_data_raw.npy.

Step 3 
transform_woe.py use the woe grouping result to update the model data.

Step 4
logic.py this part contains two ways to deal with model data.
1. Turn grouping values into dump variance.
2. Use woe instead of grouping info.
After training model, use coefficient to calculate score for test data.

