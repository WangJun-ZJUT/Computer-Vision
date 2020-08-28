### ROI Align
ROI Align：解决原图到feature map之间的位置映射。

ROI Pooling在操作的时候存在两次量化：1. 奖候选框姐姐量化成整数点坐标值；2. 将量化后的边界区域平均分割成k*k个单元，对每一个单元的边界进行量化。
经过这两个量化之后，候选框与回归出来的位置已经存在一定的偏差。如在输入是800*800的图片上，有一个655*665的包围框，经过backbone提取特征后，得到feature map。backbone的步长是32，那么665在feature map上对应的长度是665/32=20.78，ROI Pooling量化成20。接下来还需要把20*20框内的特征池化成7*7大小，所以包围框又平均分割成了7*7个矩形区域，每个矩形区域变成是2.86，ROI Pooling再次量化成2。两次量化，累计出了较大的偏差。

ROI Align取消了ROI Pooling中的量化操作，用双线性内插的方式，将特征狙击转化成一个连续的操作。
步骤如下：
- 遍历每一个候选区域，保持浮点数边界不做量化。
- 将候选区域分割成k * k个单元，每个单元的边界也不做量化。
- 在每个单元中计算固定四个坐标位置，用双线性内插的方法计算出这四个位置的值，然后进行最大池化操作。

这里对上述步骤的第三点作一些说明：这个固定位置是指在每一个矩形单元（bin）中按照固定规则确定的位置。比如，如果采样点数是1，那么就是这个单元的中心点。如果采样点数是4，那么就是把这个单元平均分割成四个小方块以后它们分别的中心点。显然这些采样点的坐标通常是浮点数，所以需要使用插值的方法得到它的像素值。在相关实验中，作者发现将采样点设为4会获得最佳性能，甚至直接设为1在性能上也相差无几。

参考：https://zhuanlan.zhihu.com/p/73113289

代码实现：
```
import torch
import cv2
import numpy as np
from torchvision.ops import RoIAlign

# feature map上的包围框
feature_boxes = torch.Tensor([[3, 4, 35, 45]])
img = cv2.imread('D:\\Tracking\\OTB50\\Basketball\\img\\0001.jpg')
img_ = torch.tensor(img[np.newaxis,:,:,:]).permute(0, 3, 1, 2).float()

# feature map与原图的比例是1:8，即stride=8
# sampling_ratio=-1时自适应调整采样点的个数
output_size = 100
roi_align = RoIAlign((output_size, output_size), spatial_scale=8, sampling_ratio=-1)
img_align = roi_align(img_.cuda(), [feature_boxes.cuda()])
cv2.imshow('img_align', img_align[0].permute(1, 2, 0).data.cpu().numpy()/255.0)
cv2.waitKey(1)
```