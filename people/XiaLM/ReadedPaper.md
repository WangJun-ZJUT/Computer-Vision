#### Inter-Region Affinity Distillation for Road Marking Segmentation CVPR2020
[论文地址](https://arxiv.org/abs/2004.05304)

##### 总结：
&nbsp;&nbsp;    论文提出了Inter-Region affinity knowledge distillation(中心区域亲和力知识蒸馏)。
整个pipeline分为三部分：
1.提取图像AOI，把每一部分区域看作一个节点。
2.将AOI进行moment pooling，统计量化每个区域的特征分布。
3.通过以上构建中心区域亲和力图，并计算教师和学生网络的亲和力图之间的loss。



#### Ultra Fast Structure-aware Deep Lane Detection 
[论文地址](https://arxiv.org/abs/2004.11757)

##### 总结：
&nbsp;&nbsp; 论文提出的是基于anchor的网格区域车道线提取，
不同于其他基于像素分割的算法。提出了结构化损失函数，速度相比像素分割网络大大提升。

其方法是将一张车道线图片，分成h * (w+1)个网络，即h行，w+1列，其中最后一列表示没有车道线的背景，
网络输入tensor维度为h * (w+1) * C, C表示线的类别数。同样groudtruth也是h * (w+1)个网格(one-shot标签)，

