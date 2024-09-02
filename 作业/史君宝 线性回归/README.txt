main.py 是文件(当前是最基本的版本，如果想改学习率和正则化参数里面已经有了，找一找将对应内容注释掉就可以)
Student_Performance.csv 是数据集

帮助文档.docx 是我写的部分大纲
帮助你写报告，最后一部分是对于拓展要求的回答

heatmap.pdf 是热力图
scatterplot_matrix.pdf 是散点矩阵图

train_loss.pdf 
是训练集在训练过程中随批次增加，loss函数值的变化情况

test_loss.pdf 
是测试集上每个数据的loss函数值

test_loss_percent.pdf 
是测试集上每个数据的loss函数值的密度分布情况，帮助我们看出loss函数值总体分布在哪个区间

test_loss_per.pdf 
是测试集上每个数据的偏差比  abs(y_hat-y)/y

test_loss_per_percent.pdf 
是测试集上每个数据的偏差比  abs(y_hat-y)/y 的密度分布情况，帮助我们看出偏差比总体分布在哪个区间