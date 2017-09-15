raw0913为最初始的数据，98155条记录和35个变量。

smp0913为选择的的4200条记录和35个变量

val_index为smp0913对应的35个变量以及其对应的索引

Use_data_raw.npy为数据集，最终建模咱用的原始数据集，也就是Data_random_splt.py文件最终生成的，4200条记录和28个字段。

Information_value.py为外调的class，用于计算iv和woe，在Data_random_splt.py需要更改为你的保存路径才能调用。

Data_random_splt.py的功能如下：
1、	实现取840个坏人，3360个好人。（随机seed=1000）
2、	计算原始变量的woe和iv，并剔除iv<0.03的7个变量，生成最终表use_data_raw，该表字段对应序列及操作见文件【字段.xlsx】的sheet use字段。

字段.xlsx包括了初始raw字段的索引、剔除不显著后的索引、以及woe分组信息。

woe_splt.py为将Use_data_raw.npy根据woe相近的分组。
