# 2020-DataFountain-Outbreak-Assistant
2020 DataFountain 疫情政务问答助手

![](images/rule_0.png)

![](images/rule_1.png)

![](images/rule_2.png)


# Trick


#### 一.
    
    发现: 原始数据集中, 同一个docid可能对应多个的question和answer.
    解决: 可以选取其中一条数据作为验证集, 其余数据作为训练集.
