# SmoothL1、IOU、GIOU、DIOU、CIOU的区别
参考：https://zhuanlan.zhihu.com/p/104236411
1. Smooth L1 Loss---Fast RCNN

$$smooth_{L1}(x)=\begin{cases}
0.5x^2 & if |x| < 1 \\
|x| - 0.5 & otherswise
\end{cases}
$$

2. IOU Loss
将4个点构成的box看成一个整体进行回归
$$IOU Loss = 1 - \frac{|B \bigcap B^{gt}|}{|B \bigcup B^{gt}|}$$
当目标框和预测框没有重叠的话，损失函数一直为1。

3. GIOU Loss
加入惩罚项，缓解**不重叠情况下梯度消失问题**。在预测框和目标框无重叠的情况下进行改进，使预测框会不断向目标框移动。
$$GIOU Loss = 1 - IOU + \frac{|C-B \bigcup B^{gt}|}{|C|}$$
其中，C表示能够同时覆盖预测框和目标框的最小边界框。
当一个框覆盖另外一个框的情况下，GIOU又退化成了IOU。

4. DIOU Loss
度量目标框和预测框之间中心点的距离，加快收敛速度。
$$DIOU Loss = 1-IOU + \frac{\rho^2(b,b^{gt})}{c^2}$$
其中，b和bgt表示预测框和目标框的中心，$\rho$代表欧氏距离，c表示覆盖两个框的最小边界框的对角线长度。

> - DIoU对比IoU和GIoU也有一些优越性:
    DIoU损失可以直接最小化两个盒子的距离，因此收敛速度比GIoU损失快得多对于目标框和预测框存在包含关系的情况，DIoU损失可以使回归非常快，而GIoU损失几乎退化为IoU损失

5. CIOU Loss
当目标框有重叠甚至包含关系时，回归速度更快。
$$CIOU Loss = 1 -  IOU + \frac{|C-B \bigcup B^{gt}|}{|C|} + \alpha v$$
其中，v用于测量长宽比的一致性，$\alpha$是平衡参数。
$$v = \frac{4}{\pi^2}(arctan \frac{w^{gt}}{h^{gt}} - arctan \frac{w}{h})^2$$
$$\alpha=\frac{v}{1-IOU+v}$$


实现代码：
参考：https://zhuanlan.zhihu.com/p/148548873


```
GIOU
def giou(preds, bbox, eps=1e-7, reduction='mean'):
    '''
   https://github.com/sfzhang15/ATSS/blob/master/atss_core/modeling/rpn/atss/loss.py#L36
    :param preds:[[x1,y1,x2,y2], [x1,y1,x2,y2],,,]
    :param bbox:[[x1,y1,x2,y2], [x1,y1,x2,y2],,,]
    :return: loss
    '''
    ix1 = torch.max(preds[:, 0], bbox[:, 0])
    iy1 = torch.max(preds[:, 1], bbox[:, 1])
    ix2 = torch.min(preds[:, 2], bbox[:, 2])
    iy2 = torch.min(preds[:, 3], bbox[:, 3])
​
    iw = (ix2 - ix1 + 1.0).clamp(0.)
    ih = (iy2 - iy1 + 1.0).clamp(0.)
​
    # overlap
    inters = iw * ih
​
    # union
    uni = (preds[:, 2] - preds[:, 0] + 1.0) * (preds[:, 3] - preds[:, 1] + 1.0) + (bbox[:, 2] - bbox[:, 0] + 1.0) * (
            bbox[:, 3] - bbox[:, 1] + 1.0) - inters + eps
​
    # ious
    ious = inters / uni
​
    ex1 = torch.min(preds[:, 0], bbox[:, 0])
    ey1 = torch.min(preds[:, 1], bbox[:, 1])
    ex2 = torch.max(preds[:, 2], bbox[:, 2])
    ey2 = torch.max(preds[:, 3], bbox[:, 3])
    ew = (ex2 - ex1 + 1.0).clamp(min=0.)
    eh = (ey2 - ey1 + 1.0).clamp(min=0.)
​
    # enclose erea
    enclose = ew * eh + eps
​
    giou = ious - (enclose - uni) / enclose
​
    loss = 1 - giou
​
    if reduction == 'mean':
        loss = torch.mean(loss)
    elif reduction == 'sum':
        loss = torch.sum(loss)
    else:
        raise NotImplementedError
    return loss
 ```

 ```
DIOU
def diou(preds, bbox, eps=1e-7, reduction='mean'):
    '''
    https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/loss/multibox_loss.py
    :param preds:[[x1,y1,x2,y2], [x1,y1,x2,y2],,,]
    :param bbox:[[x1,y1,x2,y2], [x1,y1,x2,y2],,,]
    :param eps: eps to avoid divide 0
    :param reduction: mean or sum
    :return: diou-loss
    '''
    ix1 = torch.max(preds[:, 0], bbox[:, 0])
    iy1 = torch.max(preds[:, 1], bbox[:, 1])
    ix2 = torch.min(preds[:, 2], bbox[:, 2])
    iy2 = torch.min(preds[:, 3], bbox[:, 3])
​
    iw = (ix2 - ix1 + 1.0).clamp(min=0.)
    ih = (iy2 - iy1 + 1.0).clamp(min=0.)
​
    # overlaps
    inters = iw * ih
​
    # union
    uni = (preds[:, 2] - preds[:, 0] + 1.0) * (preds[:, 3] - preds[:, 1] + 1.0) + (bbox[:, 2] - bbox[:, 0] + 1.0) * (
            bbox[:, 3] - bbox[:, 1] + 1.0) - inters
​
    # iou
    iou = inters / (uni + eps)
​
    # inter_diag
    cxpreds = (preds[:, 2] + preds[:, 0]) / 2
    cypreds = (preds[:, 3] + preds[:, 1]) / 2
​
    cxbbox = (bbox[:, 2] + bbox[:, 0]) / 2
    cybbox = (bbox[:, 3] + bbox[:, 1]) / 2
​
    inter_diag = (cxbbox - cxpreds) ** 2 + (cybbox - cypreds) ** 2
​
    # outer_diag
    ox1 = torch.min(preds[:, 0], bbox[:, 0])
    oy1 = torch.min(preds[:, 1], bbox[:, 1])
    ox2 = torch.max(preds[:, 2], bbox[:, 2])
    oy2 = torch.max(preds[:, 3], bbox[:, 3])
​
    outer_diag = (ox1 - ox2) ** 2 + (oy1 - oy2) ** 2
​
    diou = iou - inter_diag / outer_diag
    diou = torch.clamp(diou, min=-1.0, max=1.0)
​
    diou_loss = 1 - diou
​
    if reduction == 'mean':
        loss = torch.mean(diou_loss)
    elif reduction == 'sum':
        loss = torch.sum(diou_loss)
    else:
        raise NotImplementedError
    return loss
 ```

 ```
 def bbox_overlaps_ciou(bboxes1, bboxes2):
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    cious = torch.zeros((rows, cols))
    if rows * cols == 0:
        return cious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        cious = torch.zeros((cols, rows))
        exchange = True
​
    w1 = bboxes1[:, 2] - bboxes1[:, 0]
    h1 = bboxes1[:, 3] - bboxes1[:, 1]
    w2 = bboxes2[:, 2] - bboxes2[:, 0]
    h2 = bboxes2[:, 3] - bboxes2[:, 1]
​
    area1 = w1 * h1
    area2 = w2 * h2
​
    center_x1 = (bboxes1[:, 2] + bboxes1[:, 0]) / 2
    center_y1 = (bboxes1[:, 3] + bboxes1[:, 1]) / 2
    center_x2 = (bboxes2[:, 2] + bboxes2[:, 0]) / 2
    center_y2 = (bboxes2[:, 3] + bboxes2[:, 1]) / 2
​
    inter_max_xy = torch.min(bboxes1[:, 2:],bboxes2[:, 2:])
    inter_min_xy = torch.max(bboxes1[:, :2],bboxes2[:, :2])
    out_max_xy = torch.max(bboxes1[:, 2:],bboxes2[:, 2:])
    out_min_xy = torch.min(bboxes1[:, :2],bboxes2[:, :2])
​
    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    inter_area = inter[:, 0] * inter[:, 1]
    inter_diag = (center_x2 - center_x1)**2 + (center_y2 - center_y1)**2
    outer = torch.clamp((out_max_xy - out_min_xy), min=0)
    outer_diag = (outer[:, 0] ** 2) + (outer[:, 1] ** 2)
    union = area1+area2-inter_area
    u = (inter_diag) / outer_diag
    iou = inter_area / union
    with torch.no_grad():
        arctan = torch.atan(w2 / h2) - torch.atan(w1 / h1)
        v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(w2 / h2) - torch.atan(w1 / h1)), 2)
        S = 1 - iou
        alpha = v / (S + v)
    cious = iou - (u + alpha * v)
    cious = torch.clamp(cious,min=-1.0,max = 1.0)
    cious_loss=1-cious
    if reduction == 'mean':
        loss = torch.mean(cious_loss)
    elif reduction == 'sum':
        loss = torch.sum(cious_loss)
    else:
        raise NotImplementedError
    return cious_loss
```