raw0913 is raw data that drag from database，contains 98155 records and 35 variances.

smp0913 contains 4200 records and 35 variances, which are randomly picked from raw0913

val_index is the index of 35 variances in smp0913.
val_index_28new is the index of 35 variances in Use_data_raw.

Use_data_raw.npy is the python formed database, which will be used for modeling. It's generaged by Data_random_splt.py, containing 4200 records and 28 variance。

Information_value.py is a class file to calculate iv and woe. To use the class, you need to change the file rote to yours in python file Data_random_splt.py.

Data_random_splt.py：
1. Pick 840 bad men and 3360 good men randomly.(random seed=1000)
2. calculate the woe and iv for all variances, remove 7 variance whose iv<0.03.


woe_splt_vX.py is file that used to grouping variance in Use_data_raw.npy.
