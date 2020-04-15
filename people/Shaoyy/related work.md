## 消息传递机制：

**相关滤波跟踪器：**
CF在图像的引入--MOSSE 《visual object tracking using adaptive correlation filters》
通过滤波模板实现目标图像和搜索图像之间的消息传递。通过找到一个滤波模板h，与输入图像f求相关性，得到相关图g。相关图g描述目标响应，越接近目标图像响应值越大。
$$g=f\bigotimes h$$
为了加快计算速度，引入傅里叶变化，函数互相关的傅里叶变化等于函数傅里叶变换的乘积：
$$F(g)=F(f\bigotimes h)=F(f)F(h)^*$$

**孪生网络跟踪器：**
用模板和搜索图像之间的互相关操作，把模板信息传递到搜索图像上。

- Cross Correlation: 在SiamFC中，模板特征在搜索区域上按照滑窗的方式获取不同位置的响应值，最终获得一个单通道相似度图。

- Up-Channel Cross Correlation: SiamRPN中将一个具有多个独立XCorr层的重卷积层级联而输出多通道相关特征。SiamRPN在更高层次的信息上进行互相关操作，如锚。通过添加一个大的卷积层(Up-Xcorr)来缩放模板的通道数目。之后通过cross correlation的方式，得到多通道相似图。控制升维的卷积进行信息传递，造成了参数分布的严重不平衡(如RPN模块包含了20M个参数，而特征提取模块只包含了4M个参数)，使SiamRPN的训练优化变得困难。

- Depthwise Cross Correlation：和UpChannel一样，在做correlation操作以前，模版和搜索分支会分别通过一个卷积层，但并不需要进行维度提升，这里只是为了提供一个非Siamese的特征（这里与SiamFC不同，比如回归分支，是非对称的，因为输出不是一个响应值；需要模版分支和搜索分支关注不同的内容）。因为同类别的对象在相同通道层拥有较高响应，逐通道计算correlation结果，能更有效的进行信息传递，且减小了计算负担和内存使用。最后，把correlation结果用1x1卷积层整合，就可以得到分类和回归的响应图。


## 模板更新
模板更新从为最简单的每隔T帧进行线性更新，发展到按每一帧模板的置信度决定是否更新，更新方式也从简单的线性更新变为非线性更新。同时，历史更新信息也可以帮助模板找到更新方向。

**线性更新策略：**
假设视频中所有帧之间以及不同视频之间的外观变化速率恒定。在实践中，对象模板的更新要求对于不同的跟踪情况有很大的不同，这取决于运动、模糊或反向等外部因素的复杂组合。因此，简单的线性更新往往不足以应付不断变化的更新需求，并推广到所有可能遇到的情况。 此外，这种更新在所有空间维度上也是恒定的，这不允许局部部分更新。 这在部分遮挡等情况下尤其具有破坏性，在这些情况下只需要更新模板的某一部分。 最后，过度依赖初始模板可能会遭受灾难性漂移并无法从跟踪失败中恢复。

- 使用简单的线性插值来更新每个帧中的模板：
《Visual object tracking using adaptive correlation filters》CVPR 2010  
《High-speed tracking with kernelized correlation filters》 TPAMI 2015  
《Learning spatially regularized correlation
filters for visual tracking》ICCV 2015  
《Learning background-aware correlation filters for visual tracking》ICCV 2017  
《Context-aware deep feature compression
for high-speed visual tracking》CVPR 2018  

用固定的更新计划会导致模板更加关注最近的帧，忘记对象的历史外观，把历史帧的子集作为训练样本会产生更好的效果，如《Beyond correlation filters: Learning continuous convolution operators for visual tracking》ECCV, 2016


**《Learning dynamic siamese network for visual object tracking》** ICCV 2017
Guo等人建议通过傅立叶域中的**正则化线性回归**来计算相对于初始模板的变换矩阵。由于在估计变换时只考虑初始模板，这种方法忽略了跟踪过程中观察到的历史对象变化， 这可能影响模板更新的平滑性。匹配公式：$$S^l_t=corr(V^l_{t-1}*f^l(O_1),W^l_{t-1}*f^l(Z_t))$$
其中，$V^l_{t-1}$是一个学习到的更新因子，作用于初始模板特征$f^l(O_1)$后，得到更新的目标模板。$W^l_{t-1}$也是一个更新因子，作用在当前搜索图片上$f^l(Z_t)$。$V^l_{t-1}$的求解只用了上一帧的跟踪结果和第一帧目标，鲁棒性不强。

**《Visual tracking by reinforced decision making》** 2017
Choi等人用**强化学习构造了模板选择策略**，根据策略网络从模板库中选择一个最佳模板用于下一帧的目标跟踪。模板库每隔K帧添加一个新的模板，抛弃一个旧模板。这种方法没有利用累计信息。

**《Context-aware deep feature compression for high-speed visual tracking》** CVPR 2018
线性更新滤波器的方法：以目标为中心的ROI经过调整大小后，送到VGG-Net之后得到特征映射$Z^{(t)}∈R^{S×S×N_c}$。$Z^{k,(t)}$是$Z^{(t)}$的第k个通道，它对应的相关滤波器的参数为$w^{k,(t)}$。
相关滤波器更新如下：
$$\overline w^{k,(t)}=(1-\gamma)\overline w^{k,(t-1)}+ \gamma w^{k,(t)}$$

**《Joint representation and truncated inference learning for correlation filter based tracking》** ECCV 2018
Yao等人建议使用SGD离线学习CF跟踪器的更新系数。

**《Learning dynamic memory networks for object tracking》** ECCV 2018
Yang等人提出使用一个长短期记忆(LSTM)来估计当前模板。具体方法是在在线跟踪过程中将以前的模板存储在外部存储器中，根据实际情况从外部存储器中读取模板，进行模板的更新。模板更新问题被转换成内存读取和写入问题。这种计算代价大，系统相当复杂。

**《Learning the Model Update for Siamese Trackers》** ICCV 2019
Yang等人提出了一种学习模板更新的**策略网络UpdateNet**。不同于历史模板的现行组合，这里把初始模板、累计模板和上一帧模板三者作为UpdateNet网络的输入，进UpdateNet决策生成一个适合下一帧用于相似匹配的最佳模板。模板更新更加平滑，而且残差学习使更新的模板不会与初始模板相差太大，保证了系统的鲁棒性。

**《GradNet: Gradient-Guided Network for Visual Object Tracking》** ICCV 2019
Li等人提出用**梯度引导的模板更新方法**，并采取样本泛化法提高了模型的自适应能力。现有的梯度下降法，如：momentum，Adagrad，ADAM等需要大量的计算和庞大的数据集。基于上述情况，设计了加速训练网络的GradNet，仅需要两次前向计算和一次反向求梯度，就能实现更新模板。

**《Discriminative and Robust Online Learning for Siamese Visual Tracking》** AAAI 2020
Zhou等人使用长短期模型交替的方式，来适应目标的形变。长期模型即初始帧模型，短期模型即在T帧内置信度最大的新模板。
$$z^{'}=\begin{cases}
z_s, & IoU(\hat{f}_M^{reg}(z_s), \hat{f}_M^{reg}(z^1))≥v_r\\
z^1, & otherwise
\end{cases}$$
根据长短期模板分别产生的回归响应图可靠度，进行模板的更替。每一帧都要进行两次目标匹配，影响跟踪效率。


## 端到端的训练
SiamRPN对于目标的定位：
1. 丢弃远离中心位置的proposal；
2. 使用余弦窗和尺度变化惩罚因子(沿用SiamFC)对proposal进行重新排序；
3. 使用NMS对最后预测输出的bbox进行筛选；
4. 用线性插值的方法更新目标尺寸。

ATOM有分类网络和评价网络两部分，评价网络估计目标的位置和尺度，具体操作如下：
1. 根据分类网络的置信图找到最大置信点，即粗略的候选目标区；
2. 结合上一帧的尺度生成初始bbox，加入噪声后又生成9个，10个bbox都给IoUNet预测模块；
3. 对于每个bbox，5次梯度下降迭代最大化IoU得到最优的bbox；
4. 去3个最高IoU的box平均值作为最终预测结果。




