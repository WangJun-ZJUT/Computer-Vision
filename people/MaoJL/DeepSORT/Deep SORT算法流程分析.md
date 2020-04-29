# Deep SORT 基于代码的算法流程分析

## 1.检测并筛选

１.１从txt中获取当前帧的检测结果。　

１.２筛选检测框，只保留检测框的h大于min height，.且 检测框的置信度大于min_ confidence 的。

１.３利用NMS（非极大值抑制）进行筛选，将属于同一个目标的bbox舍去。目的是消除一个目标身上多个框的情况。

 原理：将所有bbox按scorce排列，将其他的bbox与最大的bbox进行IOU，大于nms max overlap 则认为是同一个目标，舍去，依次比较。(程序中nms_max_overlap 设置为1，所以其实没有进行这步操作。)

## 2.跟踪
### 2.1 预测
- 预测：依据已经得到的trk（上一帧的检测结果，如果当前帧是第一帧，则trk＝［］）使用kalman滤波预测目标在当前帧的位置。

- 预测完之后，需要对每一个tracker 的self.time_ since_ update += 1。

### 2.2 匹配

首先进行检测结果和跟踪预测结果的匹配。包含如下步骤:

（１）将已存在的tracker分为confirmed tracks和unconfirmned tracks.

（２）针对之前已经confimed tracks，将它们与当前的检测结果进行级联匹配。这个匹配操作需要从刚刚匹配成功的trk循环遍历到最多已经有30次(cascade_ depth)没有 匹配的trk。这样做是为了对更加频繁出现的目标赋予优先权。

(注意，已经确认的trk连续max_ age帧(30 帧)没能匹配检测结果才会删除。所以，那种状态为confirmed,但却已经好多帧没有匹配到检测结果的trk是存在于deep sort中的。)

遍历: (最后合并匹配结果)

- 计算当前帧每个新检测结果的深度特征与这一层中每个trk已保存的特征集之间的余弦距离矩阵cost matrix。(取最小值作为该trk与检测结果之间的计算值。)
- 在cost matrix中，进行运动信息约束。对每个trk， 计算其预测结果和检测结果之间的马氏距离，并将cost_ matrix中，相应的trk的马氏距离大于阈值(gating_ threshold)的值置为无穷大。
- 将经由max_ distance 处理之后的cost matix 作为匈牙利算法的输入，得到线性匹配结果，并去除差距较大的匹配对。

（３）unconfirmed tracks和(2)中没有匹配的tracker (unmatched tracks_ a)一起组成iou track candidates,与还没有匹配.上的检测结果(unmatched detections) 进行IOU匹配。缓解因为表观突变或者部分遮挡导致的较大变化。这样做也有可能导致一些新产生的轨迹被连接到了一些旧的轨迹上。

（４）合并(2)和(3)的结果，得到最终的匹配结果。（match，unmatched_tracks,unmatched_detections）

### ２.３根据匹配情况进行后续相应操作
（１）针对match的，要用检测结果去更新相应的tracker的参数;

注意注意，更新包括以下几个操作:

- 更新卡尔曼滤波的一系列运动变量、命中次数以及重置time_ since_ _update。
- det的深度特征保存到这个trk的特征集中。
- 如果已经连续命中3帧，将trk的状态由tentative 改为confirmed。

（２）针对unmatched_ tracks:

- 如果这个trk是还没经过确认的，直接从trk列表中删除;
- 如果这个是之前经过确认的，但是已经连续max_ age 帧没能匹配检测结果了则该tk无效，从trk列表中删除。

（３）针对unmatched detections，为其创建新的tracker。

### ２.４更新已经确认的trk的特征集
trk最多保存最近与之匹配的100帧检测结果的feature。

输出已经确认的trk的跟踪预测结果。(画在图上and输出到txt)



***********************************************【示例】  *********************************************************

# /////////////////第1帧///////////////////

## 1. 对检测框进行筛选

只保留检测框的h大于min height，.且 检测框的置信度大于min_ confidence 的，并利用NMS去除重叠率较大的ｂｂｏｘ。
第１帧中剩下了９个检测框：

	boxes:[x,y,w,h]:
	 [[298. 189.  29.  72.]
	 [144. 187.  33.  72.]
	 [347. 184.  35.  89.]
	 [223. 197.  29.  61.]
	 [120. 194.  28.  66.]
	 [502. 187.  38.  87.]
	 [415. 192.  34.  79.]
	 [441. 188.  29.  77.]
	 [484. 199.  26.  71.]]

## 2.跟踪
 
### 2.1 预测

因为是第一帧，不需要预测。
### 2.2 匹配

确认track的各种状态正常，因为第一帧还没有tracker，所以没有进行状态检查操作。
```python
	tracker.predict()
```
```python
	def __init__(self, metric, max_iou_distance=0.7, max_age=30, n_init=3):
		self.metric = metric
		self.max_iou_distance = max_iou_distance
		self.max_age = max_age
		self.n_init = n_init

		self.kf = kalman_filter.KalmanFilter()
		self.tracks = []
		# print('*******************self.tracks：',self.tracks)
		self._next_id = 1
```
```python
    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        # print('---------------------------track',self.tracks)
        for track in self.tracks:
            track.predict(self.kf)    
```
### 2.3 利用检测结果更新tracker  
```python
	tracker.update(detections)
```
(1) 获取前后两帧的匹配状态：
```python
	matches, unmatched_tracks, unmatched_detections = self._match(detections)
	／／输出匹配状态，前一帧未匹配的ｔｒａｃｋｓ，当前帧未匹配的detection
```
因第一帧不存在tracker，因此所有的检测都匹配不到，输出为：

	输出：
	matches, unmatched_tracks, unmatched_detections: [] [] [0, 1, 2, 3, 4, 5, 6, 7, 8]

(2) 利用每个检测框来创建其对应的新的tracker:

初始化ｋａｌｍａｎ滤波的一系列运动变量:

（_next_id += 1：多一个tracker，id也就多一个）
```python
for detection_idx in unmatched_detections:
    	self._initiate_track(detections[detection_idx])
```
```python	
def _initiate_track(self, detection):
    	mean, covariance = self.kf.initiate(detection.to_xyah())
    	self.tracks.append(Track(
        	mean, covariance, self._next_id, self.n_init, self.max_age,
        	detection.feature))
```

注意：每个tracker在创建时，它的状态都是tentative。
```python	
    def __init__(self, mean, covariance, track_id, n_init, max_age,feature=None):
        self.mean = mean                # 初始的mean
        self.covariance = covariance        # 初始的covariance
        self.track_id = track_id            # id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0          # 初始值为0

        self.state = TrackState.Tentative   # 初始为Tentative状态
        self.features = []
        if feature is not None:
            self.features.append(feature)   # 相应的det特征存入特征库中
```
```python
	class TrackState:

		Tentative = 1
		Confirmed = 2
		Deleted = 3
```

此时第一帧的tracker已经创建，其结果与第一帧的检测结果一样。因为tracker的结果还未得到确认，所以不会把tracker的结果输出到txt中。
```python
	# Store results.
	for track in tracker.tracks:
    	if not track.is_confirmed() or track.time_since_update > 1:
        	continue
    	bbox = track.to_tlwh()
	results.append([frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])
```

同时，未确认的tracker的结果不会画在图上，所以第一帧只能看到检测结果（画出ｂｏｘ框，但是不会有id标号）。
```python
	def draw_trackers(self, tracks):
	    self.viewer.thickness = 2
	    for track in tracks:
	        if not track.is_confirmed() or track.time_since_update > 0:
	            continue
	        self.viewer.color = create_unique_color_uchar(track.track_id)
	        self.viewer.rectangle(
	            *track.to_tlwh().astype(np.int), label=str(track.track_id))
```
```python
	def draw_trackers(self, tracks):
		self.viewer.thickness = 2
        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 0:
                continue
            self.viewer.color = create_unique_color_uchar(track.track_id)
            self.viewer.rectangle(
                *track.to_tlwh().astype(np.int), label=str(track.track_id))
```

其实第一帧除了利用每个检测结果创建了其对应的track以外，没有其他操作。

# /////////////////第2帧///////////////

## 1. 检测并筛选
第二帧筛选得到了８个检测框：

	输出：
	boxes:[x,y,w,h]:
	[[297. 189.  30.  74.]
 	[147. 188.  32.  75.]
	[351. 188.  34.  87.]
	[121. 194.  28.  67.]
 	[417. 193.  34.  80.]
 	[222. 199.  28.  60.]
	[503. 188.  39.  86.]
 	[441. 187.  32.  79.]]

## ２.跟踪
### ２.１　预测

因为第一帧创建了９个tracker，所以利用ｋａｌｍａｎ滤波，可在当前帧得到９个跟踪预测结果。预测完之后，对每一个tracker的self.time_since_update += 1。

目前有８个det，９个trk，那么如何进行tracker的更新呢？

### ２.２匹配
#### ２.２.１　检测结果和预测结果的匹配。
```python
	matches, unmatched_tracks, unmatched_detections = self._match(detections)
```
 
步骤：

（１）将已经存在的tracker分为confirmed　trackers　和　unconfirmed　trackers。 
```python
	confirmed_tracks = [i for i, t in enumerate(self.tracks) if t.is_confirmed()]
	unconfirmed_tracks = [i for i, t in enumerate(self.tracks) if not t.is_confirmed()]
```
 
此时因为这些tracker还没有经过确认（匹配），所以它们的state是Tentative。

	
	输出：
	confirmed_tracks: []
	unconfirmed_tracks: [0, 1, 2, 3, 4, 5, 6, 7, 8]

（２）针对之前的confirmed　trackers，将它们与当前的检测结果进行级联匹配。 
```python
	# Associate confirmed tracks using appearance features.
	matches_a, unmatched_tracks_a, unmatched_detections = \
		linear_assignment.matching_cascade(
			gated_metric, self.metric.matching_threshold, self.max_age,
			self.tracks, detections, confirmed_tracks)
```
 
由于当前帧还没有confirmed　trackers，所以没有级联匹配。

	matches_a, unmatched_tracks_a, unmatched_detections: [] [] [0, 1, 2, 3, 4, 5, 6, 7]

（３）unmatched_tracks和unmatched_tracks_a一起组成iou_track_candidates，与还没有匹配上的检测结果（unmatched_detections）进行IOU匹配。

（意思是，当目标被遮挡时，tracker会断掉，在规定帧内，这些断掉的tracker被记为unmatched_tracks_a，跟随上一帧的unmatched_tracks与当前帧的unmatched_detections进行IOU匹配）
```python
	iou_track_candidates = unconfirmed_tracks + [
    	k for k in unmatched_tracks_a if
    	self.tracks[k].time_since_update == 1]
```
```python
	matches_b, unmatched_tracks_b, unmatched_detections = \
	    linear_assignment.min_cost_matching(
	        iou_matching.iou_cost, self.max_iou_distance, self.tracks,
	        detections, iou_track_candidates, unmatched_detections)
```

当前iou_track_candidates有９（上一帧９＋前几帧０）个，还没match的有８个：

	iou_track_candidates: [0, 1, 2, 3, 4, 5, 6, 7, 8]
	unmatched_detections: [0, 1, 2, 3, 4, 5, 6, 7]

＃　首先计算这些box两两之间的ｉｏｕ，经过１－ｉｏｕ得到ｃｏｓｔ——matrix：
```python
	bbox = tracks[track_idx].to_tlwh()
	candidates = np.asarray([detections[i].tlwh for i in detection_indices])
	cost_matrix[row, :] = 1. - iou(bbox, candidates)
```
 
cost_matrix：
![cost_matrix](https://github.com/WangJun-ZJUT/Computer-Vision/blob/master/people/MaoJL/DeepSORT/img/001.jpg)
 

程序中把ｃｏｓｔ大于阈值（０.７）的，都设置成了０.７.
 ![cost_matrix](https://github.com/WangJun-ZJUT/Computer-Vision/blob/master/people/MaoJL/DeepSORT/img/002.jpg)


＃　然后把cost_matrix作为匈牙利算法的输入，得到线性匹配结果：
```python
	indices = linear_assignment(cost_matrix)
```
	indices: [[0 0]
	 [1 1]
	 [2 2]
	 [3 5]
	 [4 3]
	 [5 6]
	 [6 4]
	 [7 7]]

注意：如果某个组合的ｃｏｓｔ值大于阈值，这样的组合还是认为是不match的，相应的，还会把组合中的检测框和跟踪框都踢到各自的unmatch列表中。
 
```python
	for row, col in indices:
	    track_idx = track_indices[row]
	    detection_idx = detection_indices[col]
	    if cost_matrix[row, col] > max_distance:
	        unmatched_tracks.append(track_idx)
	        unmatched_detections.append(detection_idx)
	    else:
	        matches.append((track_idx, detection_idx))
```
 
经过上述处理之后，依据IOU得到当前的匹配结果：
```python
	matches = matches_a + matches_b
	unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
	return matches, unmatched_tracks, unmatched_detections
```
```python
	输出：
	matches_b: [(0, 0), (1, 1), (2, 2), (3, 5), (4, 3), (5, 6), (6, 4), (7, 7)]
	unmatched_detections: []
```

经由以上，得到当前的匹配结果：
 
	输出：
	matches, unmatched_tracks, unmatched_detections: [(0, 0), (1, 1), (2, 2), (3, 5), (4, 3), (5, 6), (6, 4), (7, 7)] [8] []
 


### ２.３　根据匹配情况更新tracker。

（１）针对match的，要用检测结果去更新相应的tracker的参数；　
 
```python	
	for track_idx, detection_idx in matches:
    self.tracks[track_idx].update(
        self.kf, detections[detection_idx])
```
 
更新包括几个操作：

- 更新ｋａｌｍａｎ滤波的一系列运动变量，命中次数和重置time＿ｓｉｎｃｅ＿ｕｐｄａｔｅ．
- ｄｅｔ的深度特征保存到这个ｔｒｋ的特征集中。
- 如果已经连续命中３帧，将ｔｒｋ的状态由tentative改为confirmed。
 
```python
		self.mean, self.covariance = kf.update(
	    self.mean, self.covariance, detection.to_xyah())
		self.features.append(detection.feature)
	
		self.hits += 1
		self.time_since_update = 0
		if self.state == TrackState.Tentative and self.hits >= self._n_init:
	    self.state = TrackState.Confirmed
```
 

（２）针对unmatched_tracks：

- 　如果这个trk是还没经过确认的，直接从tk列表中删除;
- 　如果这个是之前经过确认的,但是已经连续max_ age 帧(程序中设置的3)没能匹配检测结果了，我们也认为这个turk无效了，需要从trk列表中删除。
```python
	for track_idx in unmatched_tracks:
    	self.tracks[track_idx].mark_missed()
```
```python
	def mark_missed(self):
    	if self.state == TrackState.Tentative:
        	self.state = TrackState.Deleted
    	elif self.time_since_update > self._max_age:
        	self.state = TrackState.Deleted
```
```python
	self.tracks = [t for t in self.tracks if not t.is_deleted()]
```

（３）针对unmatched_detections，要为其创建新的tracker。
```python
	for detection_idx in unmatched_detections:
	    self._initiate_track(detections[detection_idx])
```
```python
	def _initiate_track(self, detection):
	    mean, covariance = self.kf.initiate(detection.to_xyah())
	    self.tracks.append(Track(
	        mean, covariance, self._next_id, self.n_init, self.max_age,
	        detection.feature))
	    self._next_id += 1
```
```python
	def __init__(self, mean, covariance, track_id, n_init, max_age,
             feature=None):
	    self.mean = mean                # 初始的mean
	    self.covariance = covariance        # 初始的covariance
	    self.track_id = track_id            # id
	    self.hits = 1
	    self.age = 1
	    self.time_since_update = 0          # 初始值为0
	
	    self.state = TrackState.Tentative   # 初始为Tentative状态
	    self.features = []
	    if feature is not None:
	        self.features.append(feature)   # 相应的det特征存入特征库中

	    self._n_init = n_init
	    self._max_age = max_age
```

因为当前帧没有unmatched_detections，所以没有进行这些操作。

### ２.４更新已经确认的tracker的特征集。

```python
	active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
	# print('active_targets:',active_targets)
	features, targets = [], []
	for track in self.tracks:
	    if not track.is_confirmed():
	        continue
	    features += track.features
	    targets += [track.track_id for _ in track.features]
	    track.features = []
	self.metric.partial_fit(
	    np.asarray(features), np.asarray(targets), active_targets)
```
```python
	def partial_fit(self, features, targets, active_targets):
		# 每个activate的追踪器保留最近的self，budget的特征
	       for feature, target in zip(features, targets):
	        self.samples.setdefault(target, []).append(feature)
	        if self.budget is not None:
	            self.samples[target] = self.samples[target][-self.budget:]
		# 以dict的形式插入总库
	    self.samples = {k: self.samples[k] for k in active_targets}
```

因为当前还没有已经确认的trk，所以没有进行这项操作。

同样，这帧的结果还未得到确认，所以不会吧tracker的结果输出到ｔｘｔ中，未确认的tracker的结果不会画在图上，所以第二帧只能看到检测结果（画出ｂｏｘ框，但是不会有id标号）


# ///////////////////第3帧/////////////////
## 1. 检测并筛选

得到第三帧有１１个检测框：
 
	boxes:[x,y,w,h]:
	 [[296. 188.  31.  77.]
	 [148. 187.  33.  76.]
	 [508. 188.  37.  92.]
	 [419. 193.  35.  79.]
	 [122. 194.  28.  67.]
	 [357. 189.  39.  87.]
	 [217. 195.  26.  64.]
	 [443. 190.  30.  79.]
	 [237. 201.  19.  48.]
	 [324. 204.  19.  48.]
	 [488. 198.  32.  78.]]

当前有８个trk，还没有完成confirm（击中次数还没达到３），状态都还是tentative：

目前有１１个det，８个trk，如何进行匹配与跟踪，以及如何update　 tracker呢？

## ２.跟踪
### ２.１　预测

因为第二帧创建了８个tracker，所以利用ｋａｌｍａｎ滤波，可在当前帧得到８个跟踪预测结果。预测完之后，对每一个tracker的self.time_since_update += 1。

目前有１１个ｄｅｔ，８个ｔｒｋ，那么如何进行tracker的更新呢？

### ２.２　匹配
进行ｄｅｔ和ｔｒｋ的匹配，包含如下步骤：

（１）将已经存在的tracker分为confirmed　trackers　和　unconfirmed　trackers。

	输出：
	confirmed_tracks: []
	unconfirmed_tracks: [0, 1, 2, 3, 4, 5, 6, 7]
	self.tracks: 8

（２）针对之前的confirmed　trackers，将它们与当前的检测结果进行级联匹配。

由于当前帧还没有confirmed　trackers，所以没有级联匹配。

	输出：
	matches_a, unmatched_tracks_a, unmatched_detections: [] [] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

（３）unmatched_tracks和unmatched_tracks_a一起组成iou_track_candidates，与还没有匹配上的检测结果（unmatched_detections）进行IOU匹配。

当前iou_track_candidates有８（上一帧８＋前几帧０）个，还没match的有３个：
　
	输出：
	iou_track_candidates: [0, 1, 2, 3, 4, 5, 6, 7]
	unmatched_detections: [8, 9, 10]

- 首先计算这些ｂｏｘ两两之间的ｉｏｕ，经过１－ｉｏｕ得到cost_matrix：

- ｃｏｓｔ大于阈值（０.７）的，都设置成了０.７

		cost_matrix111: [[0.07215288 0.70001    0.70001    0.70001    0.70001    0.70001
		  0.70001    0.70001    0.70001    0.70001    0.70001   ]
		 [0.70001    0.09085532 0.70001    0.70001    0.70001    0.70001
		  0.70001    0.70001    0.70001    0.70001    0.70001   ]
		 [0.70001    0.70001    0.70001    0.70001    0.70001    0.37694781
		  0.70001    0.70001    0.70001    0.70001    0.70001   ]
		 [0.70001    0.70001    0.70001    0.70001    0.70001    0.70001
		  0.38936927 0.70001    0.68310482 0.70001    0.70001   ]
		 [0.70001    0.70001    0.70001    0.70001    0.06452229 0.70001
		  0.70001    0.70001    0.70001    0.70001    0.70001   ]
		 [0.70001    0.70001    0.23736137 0.70001    0.70001    0.70001
		  0.70001    0.70001    0.70001    0.70001    0.70001   ]
		 [0.70001    0.70001    0.70001    0.13923339 0.70001    0.70001
		  0.70001    0.70001    0.70001    0.70001    0.70001   ]
		 [0.70001    0.70001    0.70001    0.70001    0.70001    0.70001
		  0.70001    0.12470238 0.70001    0.70001    0.70001   ]]


- 然后把cost_matrix作为匈牙利算法的输入，得到线性匹配结果：

		indices: [[0 0]
		 [1 1]
		 [2 5]
		 [3 6]
		 [4 4]
		 [5 2]
		 [6 3]
		 [7 7]]


经过上述处理之后，依据IOU得到当前匹配结果：

		# 总的未匹配的结果 iou_track_candidates:
		 [0, 1, 2, 3, 4, 5, 6, 7]
		# 上一帧与当前帧匹配的结果 matches_b:
		 [(0, 0), (1, 1), (2, 5), (3, 6), (4, 4), (5, 2), (6, 3), (7, 7)]
		# 级联中未匹配的结果 unmatched_tracks_b:
		 []
		# 未匹配的检测 unmatched_detections:
		 [8, 9, 10]

经由以上，得到当前的匹配结果：

	
	输出：
	matches, unmatched_tracks, unmatched_detections: [(0, 0), (1, 1), (2, 5), (3, 6), (4, 4), (5, 2), (6, 3), (7, 7)] [] [8, 9, 10]

### ２.３　根据匹配情况更新tracker。　

（１）针对match的，要用检测结果去更新相应的tracker的参数；　（８个）
更新包括几个操作：

- 更新ｋａｌｍａｎ滤波的一系列运动变量，命中次数和重置time＿ｓｉｎｃｅ＿ｕｐｄａｔｅ．
- ｄｅｔ的深度特征保存到这个ｔｒｋ的特征集中。
- 如果已经连续命中３帧，将ｔｒｋ的状态由tentative改为confirmed。

因为有８个ｔｒｋ的命中帧数已经是３帧，所以它们的状态改为confirmed。
 
```python
	if self.state == TrackState.Tentative and self.hits >= self._n_init:
	    self.state = TrackState.Confirmed
```
 
（２）针对unmatched_tracks：

- 　如果这个trk是还没经过确认的，直接从tk列表中删除;（０个：unmatched_tracks＝［］）
- 　如果这个是之前经过确认的,但是已经连续max_ age 帧(程序中设置的3)没能匹配检测结果了，我们也认为这个turk无效了，需要从trk列表中删除。
- 　
（３）针对unmatched_detections，要为其创建新的tracker。（３个：unmatched_detections: [8, 9, 10]）　

 ### ２.４更新已经确认的tracker的特征集。

因为当前帧已经有了确认的ｔｒｋ（８个），所以有了activate　target。

	active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
	
> active_targets: [1, 2, 3, 4, 5, 6, 7, 8]

把这些activate target之前保存的feature (之前每帧只要能匹配上，都会把与之匹配的det的feature 保存下来)，用于更新卡尔曼滤波的distance metric。

【注意注意:程序中budget为100， 也就是trk最多保存最近与之匹配的100帧检测结果的feature。】
```python
	active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
	# print('active_targets:',active_targets)
	features, targets = [], []
	for track in self.tracks:
	    if not track.is_confirmed():
	        continue
	    features += track.features
	    targets += [track.track_id for _ in track.features]
	    track.features = []
	self.metric.partial_fit(
	    np.asarray(features), np.asarray(targets), active_targets)
```
```python
	def partial_fit(self, features, targets, active_targets):
		# 每个activate的追踪器保留最近的self，budget的特征
	       for feature, target in zip(features, targets):
	        self.samples.setdefault(target, []).append(feature)
	        if self.budget is not None:
	            self.samples[target] = self.samples[target][-self.budget:]
		# 以dict的形式插入总库
	    self.samples = {k: self.samples[k] for k in active_targets}
```

因为当前帧不仅有检测结果，还有已经确认的trk,这些trk的跟踪结果也会画在图上(有11个)。同时，已经确认的trk的跟踪结果也会保存在txt中。.
![frame03](https://github.com/WangJun-ZJUT/Computer-Vision/blob/master/people/MaoJL/DeepSORT/img/003.jpg)  
![txt03](https://github.com/WangJun-ZJUT/Computer-Vision/blob/master/people/MaoJL/DeepSORT/img/004.jpg)
 

 


# ///////////////////第4帧///////////////////

##１.检测并筛选

检测并筛选得到第４帧有１１个检测框：
 
		boxes:[x,y,w,h]:
		 [[296. 186.  31.  78.]
		 [149. 186.  32.  76.]
		 [123. 192.  27.  68.]
		 [362. 187.  38.  91.]
		 [217. 195.  24.  61.]
		 [421. 190.  35.  82.]
		 [510. 187.  37.  91.]
		 [447. 190.  26.  75.]
		 [326. 204.  18.  46.]
		 [491. 199.  29.  70.]
		 [238. 197.  20.  51.]]
## ２.跟踪
### ２.１　预测

因为第４帧创建了１１个tracker，所以利用ｋａｌｍａｎ滤波，可在当前帧得到１１个跟踪预测结果。预测完之后，对每一个tracker的self.time_since_update += 1。

		confirmed_tracks: []
		unconfirmed_tracks: [0, 1, 2, 3, 4, 5, 6, 7]
		self.tracks: 8

目前有１１个ｄｅｔ，１１个ｔｒｋ（８个confirmed，３个tentative），那么如何进行tracker的更新呢？

### ２.２　匹配

进行ｄｅｔ和ｔｒｋ的匹配，包含如下***步骤***：

（１）将已经存在的tracker分为confirmed　trackers　和　unconfirmed　trackers。

		confirmed_tracks: [0, 1, 2, 3, 4, 5, 6, 7]
		unconfirmed_tracks: [8, 9, 10]
		self.tracks: 11

（２）针对之前的confirmed　trackers，将它们与当前的检测结果进行级联匹配。

> 为啥叫做“级联匹配”? ?就是因为这个匹配操作需要从刚刚匹配成功的trk循环遍历到最多已经有30次(cascade_ _depth)没有匹配的trk。

终于，第4帧有confimed tracks了，可以进行级联匹配操作了...步骤如下:
 
```python
	for level in range(cascade_depth):
	    if len(unmatched_detections) == 0:  # No detections left
	        break
	
	    track_indices_l = [
	        k for k in track_indices
	        if tracks[k].time_since_update == 1 + level
	    ]
	    if len(track_indices_l) == 0:  # Nothing to match at this level
	        continue
	
	    matches_l, _, unmatched_detections = \
	        min_cost_matching(
	            distance_metric, max_distance, tracks, detections,
	            track_indices_l, unmatched_detections)
	    matches += matches_l
	unmatched_tracks = list(set(track_indices) - set(k for k, _ in matches))
```
 
／／／／／／级联匹配循环中／／／／／／

因为这８个ｔｒｋ在上一帧中都经历了update（time＿since＿update被重置为０），在 刚刚的predict 中time_ since_ update 增加1变为1。

所以当前而言，有效循环仅有level＝０这一层。

下面这个distance_ matric 包括外观(深度特征)和运动信息(马氏距离)。
 
```python
		cost_matrix = distance_metric(
    		tracks, detections, track_indices, detection_indices)
```
 
计算当前帧每个新检测结果的深度特征与这一层中每个trk已保存的特征集之间的余弦距离矩阵。
 
```python
	def gated_metric(tracks, dets, track_indices, detection_indices):
	    features = np.array([dets[i].feature for i in detection_indices])
	    targets = np.array([tracks[i].track_id for i in track_indices])   # 这个targets和c85的active_targets一样。
	    # print('targets:',targets)
	# 计算每个深度特征与这一层中每个trk已保存的特征集之间的余弦距离矩阵
	    cost_matrix = self.metric.distance(features, targets)
	    cost_matrix = linear_assignment.gate_cost_matrix(
	        self.kf, cost_matrix, tracks, dets, track_indices,
	        detection_indices)
	
	    return cost_matrix
```
 
具体过程是针对trk 的每个特征(因为每个trk都有一个特征集),计算它们与当前这14个det的特征之间的(1-余弦距离)。然后取最小值作为该trk与检测结果之间的计算值。

```python
	features = np.array([dets[i].feature for i in detection_indices])
	targets = np.array([tracks[i].track_id for i in track_indices])
	cost_matrix = self.metric.distance(features, targets)
```
```python
	cost_matrix = np.zeros((len(targets), len(features)))
	for i, target in enumerate(targets):
	    cost_matrix[i, :] = self._metric(self.samples[target], features)
	return cost_matrix
```
```python
	distances = _cosine_distance(x, y)
	return distances.min(axis=0)
```


由此，分别得到这８个ｔｒｋ和１４个ｄｅｔ之间的计算值。

		cost_matrix: [[0.02023339 0.29850411 0.44907928 0.31560659 0.20198858 0.3288002
		  0.31499684 0.31050617 0.287498   0.27524912 0.19508636]
		 [0.28607941 0.03209376 0.35560262 0.34028852 0.3200357  0.5144363
		  0.40016687 0.33700073 0.3482033  0.31202698 0.26062381]
		 [0.32741988 0.38263899 0.69244003 0.06043547 0.28002524 0.40995812
		  0.59532726 0.48511219 0.23045903 0.50101143 0.31813174]
		 [0.20582271 0.35502416 0.46645522 0.25115764 0.04201472 0.25663936
		  0.23206902 0.22747219 0.19357866 0.25943094 0.2108044 ]
		 [0.44074821 0.30449224 0.00944638 0.66983765 0.47476071 0.66336334
		  0.47939146 0.50026762 0.55963075 0.40713763 0.48527277]
		 [0.36658049 0.49771833 0.48334181 0.61574954 0.32726502 0.28979874
		  0.00359911 0.33039951 0.43519032 0.1887576  0.39907777]
		 [0.40084898 0.66898113 0.69848609 0.4452405  0.25479519 0.00766444
		  0.27265644 0.26321059 0.3365283  0.40368915 0.49623644]
		 [0.26410222 0.46481758 0.59198606 0.48767895 0.25756538 0.23508245
		  0.30532736 0.01520348 0.44101393 0.20466948 0.32043082]]

接着，在cost_matrix中，进行运动信息约束。

```python
	def gated_metric(tracks, dets, track_indices, detection_indices):
	    features = np.array([dets[i].feature for i in detection_indices])
	    targets = np.array([tracks[i].track_id for i in track_indices])
	    # print('targets:',targets)
	    # 计算当前帧每个新检测结果的深度特征与这一层中每个trk已保存的特征集之间的余弦距离矩阵
	    cost_matrix = self.metric.distance(features, targets)
	    # print('cost_matrix:',cost_matrix)
	    # 接着，在cost_matrix中，进行运动信息约束。
	    cost_matrix = linear_assignment.gate_cost_matrix(
	        self.kf, cost_matrix, tracks, dets, track_indices,
	        detection_indices)
	
	    return cost_matrix
```
```python
	gating_dim = 2 if only_position else 4
	gating_threshold = kalman_filter.chi2inv95[gating_dim]
	measurements = np.asarray(
	    [detections[i].to_xyah() for i in detection_indices])
	for row, track_idx in enumerate(track_indices):
	    track = tracks[track_idx]
	    gating_distance = kf.gating_distance(
	        track.mean, track.covariance, measurements, only_position)
	    cost_matrix[row, gating_distance > gating_threshold] = gated_cost
	return cost_matrix
```

展开来说，先将各检测结果由[x.y,w,h]转化为[center x,center y, aspect ration,height]:
 
```python
	measurements = np.asarray(
	    [detections[i].to_xyah() for i in detection_indices])
```
 
对每个trk，计算其预测结果和检测结果之间的马氏距离，并将cost_ matix中，相应的trk的马氏距离大于阈值(gating_ threshold)的值置 为100000 (gated_ cost，相当于设为无穷大) :

```python
	gating_distance = kf.gating_distance(
	    track.mean, track.covariance, measurements, only_position)
	cost_matrix[row, gating_distance > gating_threshold] = gated_cost
```
 
	输出：
	cost_matrix: [[2.02333927e-02 1.00000000e+05 1.00000000e+05 1.00000000e+05
	  1.00000000e+05 1.00000000e+05 1.00000000e+05 1.00000000e+05
	  1.00000000e+05 1.00000000e+05 1.00000000e+05]
	 [1.00000000e+05 3.20937634e-02 1.00000000e+05 1.00000000e+05
	  1.00000000e+05 1.00000000e+05 1.00000000e+05 1.00000000e+05
	  1.00000000e+05 1.00000000e+05 1.00000000e+05]
	 [1.00000000e+05 1.00000000e+05 1.00000000e+05 6.04354739e-02
	  1.00000000e+05 1.00000000e+05 1.00000000e+05 1.00000000e+05
	  1.00000000e+05 1.00000000e+05 1.00000000e+05]
	 [1.00000000e+05 1.00000000e+05 1.00000000e+05 1.00000000e+05
	  4.20147181e-02 1.00000000e+05 1.00000000e+05 1.00000000e+05
	  1.00000000e+05 1.00000000e+05 1.00000000e+05]
	 [1.00000000e+05 1.00000000e+05 9.44638252e-03 1.00000000e+05
	  1.00000000e+05 1.00000000e+05 1.00000000e+05 1.00000000e+05
	  1.00000000e+05 1.00000000e+05 1.00000000e+05]
	 [1.00000000e+05 1.00000000e+05 1.00000000e+05 1.00000000e+05
	  1.00000000e+05 1.00000000e+05 3.59910727e-03 1.00000000e+05
	  1.00000000e+05 1.00000000e+05 1.00000000e+05]
	 [1.00000000e+05 1.00000000e+05 1.00000000e+05 1.00000000e+05
	  1.00000000e+05 7.66444206e-03 1.00000000e+05 2.63210595e-01
	  1.00000000e+05 1.00000000e+05 1.00000000e+05]
	 [1.00000000e+05 1.00000000e+05 1.00000000e+05 1.00000000e+05
	  1.00000000e+05 2.35082448e-01 1.00000000e+05 1.52034760e-02
	  1.00000000e+05 1.00000000e+05 1.00000000e+05]]
 

将经过马氏距离处理的矩阵cost matix 继续经由max distance 处理，(程序中 max distance=0.2)得到处理过的cost＿matrix:

程序中把ｃｏｓｔ大于阈值（０.７）的，都设置成了０.７
 
```python
	cost_matrix[cost_matrix > max_distance] = max_distance + 1e-5
```
 
	cost_matrix111: [[0.02023339 0.20001    0.20001    0.20001    0.20001    0.20001
	  0.20001    0.20001    0.20001    0.20001    0.20001   ]
	 [0.20001    0.03209376 0.20001    0.20001    0.20001    0.20001
	  0.20001    0.20001    0.20001    0.20001    0.20001   ]
	 [0.20001    0.20001    0.20001    0.06043547 0.20001    0.20001
	  0.20001    0.20001    0.20001    0.20001    0.20001   ]
	 [0.20001    0.20001    0.20001    0.20001    0.04201472 0.20001
	  0.20001    0.20001    0.20001    0.20001    0.20001   ]
	 [0.20001    0.20001    0.00944638 0.20001    0.20001    0.20001
	  0.20001    0.20001    0.20001    0.20001    0.20001   ]
	 [0.20001    0.20001    0.20001    0.20001    0.20001    0.20001
	  0.00359911 0.20001    0.20001    0.20001    0.20001   ]
	 [0.20001    0.20001    0.20001    0.20001    0.20001    0.00766444
	  0.20001    0.20001    0.20001    0.20001    0.20001   ]
	 [0.20001    0.20001    0.20001    0.20001    0.20001    0.20001
	  0.20001    0.01520348 0.20001    0.20001    0.20001   ]]

把cost_matrix作为匈牙利算法的输入，得到线性匹配结果：

	indices: [[0 0]
	 [1 1]
	 [2 3]
	 [3 4]
	 [4 2]
	 [5 6]
	 [6 5]
	 [7 7]]

对匹配结果进行筛选，删去两者差距太大的：
 
```python
	matches, unmatched_tracks, unmatched_detections = [], [], []
	for col, detection_idx in enumerate(detection_indices):
	    if col not in indices[:, 1]:
	        unmatched_detections.append(detection_idx)
	for row, track_idx in enumerate(track_indices):
	    if row not in indices[:, 0]:
	        unmatched_tracks.append(track_idx)
	for row, col in indices:
	    track_idx = track_indices[row]
	    detection_idx = detection_indices[col]
	    if cost_matrix[row, col] > max_distance:
	        unmatched_tracks.append(track_idx)
	        unmatched_detections.append(detection_idx)
	    else:
	        matches.append((track_idx, detection_idx))
	return matches, unmatched_tracks, unmatched_detections
```
 
由此得到当前level的匹配结果：

	matches,  unmatched_detections: [(0, 0), (1, 1), (2, 3), (3, 4), (4, 2), (5, 6), (6, 5), (7, 7)] [8, 9, 10]

组合各层的匹配：

	matches += matches_l

／／／／／／／／／／结束循环／／／／／／／／／／／／

得到由级联匹配得到的匹配结果：

	matches_a, unmatched_tracks_a, unmatched_detections: [(0, 0), (1, 1), (2, 3), (3, 4), (4, 2), (5, 6), (6, 5), (7, 7)] [] [8, 9, 10]

（３）unmatched_tracks和unmatched_tracks_a一起组成iou_track_candidates，与还没有匹配上的检测结果（unmatched_detections）进行IOU匹配。

当前iou_track_candidates有３（上一帧３＋前几帧０）个，还没match的有０个：
　
	iou_track_candidates: [8, 9, 10]
	unmatched_detections: []

- 首先计算这些ｂｏｘ两两之间的ｉｏｕ，经过１－ｉｏｕ得到cost_matrix：

- ｃｏｓｔ大于阈值（０.７）的，都设置成了０.７

		cost_matrix111: [[0.02023339 0.20001    0.20001    0.20001    0.20001    0.20001
		0.20001    0.20001    0.20001    0.20001    0.20001   ]
		[0.20001    0.03209376 0.20001    0.20001    0.20001    0.20001
		0.20001    0.20001    0.20001    0.20001    0.20001   ]
		[0.20001    0.20001    0.20001    0.06043547 0.20001    0.20001
		0.20001    0.20001    0.20001    0.20001    0.20001   ]
		[0.20001    0.20001    0.20001    0.20001    0.04201472 0.20001
		0.20001    0.20001    0.20001    0.20001    0.20001   ]
		[0.20001    0.20001    0.00944638 0.20001    0.20001    0.20001
		0.20001    0.20001    0.20001    0.20001    0.20001   ]
		[0.20001    0.20001    0.20001    0.20001    0.20001    0.20001
		0.00359911 0.20001    0.20001    0.20001    0.20001   ]
		[0.20001    0.20001    0.20001    0.20001    0.20001    0.00766444
		0.20001    0.20001    0.20001    0.20001    0.20001   ]
		[0.20001    0.20001    0.20001    0.20001    0.20001    0.20001
		0.20001    0.01520348 0.20001    0.20001    0.20001   ]]

- 然后把cost_matrix作为匈牙利算法的输入，得到线性匹配结果：

		indices: [[0 0]
		 [1 1]
		 [2 3]
		 [3 4]
		 [4 2]
		 [5 6]
		 [6 5]
		 [7 7]]

经过上述处理之后，依据IOU得到当前匹配结果：

	matches_b: [(8, 10), (9, 8), (10, 9)]
	unmatched_tracks_b: []
	unmatched_detections: []

经由以上，得到当前的匹配结果：

	matches, unmatched_tracks, unmatched_detections: [(0, 0), (1, 1), (2, 3), (3, 4), (4, 2), (5, 6), (6, 5), (7, 7), (8, 10), (9, 8), (10, 9)] [] []

### ２.３　根据匹配情况更新tracker。　

（１）针对match（１１个）的，要用检测结果去更新相应的tracker的参数；　

这１１个match的trk中，有８个是之前已经confirmed（２）的，有３个是tentative（１）。
 
	self.state.before: 2		self.state.after: 2
	self.state.before: 2		self.state.after: 2
	self.state.before: 2		self.state.after: 2
	self.state.before: 2		self.state.after: 2
	self.state.before: 2		self.state.after: 2 
	self.state.before: 2		self.state.after: 2
	self.state.before: 2		self.state.after: 2
	self.state.before: 2		self.state.after: 2
	self.state.before: 1		self.state.after: 1
	self.state.before: 1		self.state.after: 1
	self.state.before: 1		self.state.after: 1
 
（２）针对unmatched_tracks：

- 　如果这个trk是还没经过确认的，直接从tk列表中删除;（０个：unmatched_tracks＝［］）
- 　如果这个是之前经过确认的,但是已经连续max_ age 帧(３０帧)没能匹配检测结果了，我们也认为这个trk无效了，需要从trk列表中删除。

（３）针对unmatched_detections，要为其创建新的tracker。（３个：		
> unmatched_detections: [8, 9, 10]）　

## ２.４更新已经确认的tracker的特征集。

因为当前帧已经有了确认的ｔｒｋ（８个，还是第３帧中的那８个）

	active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]

第３帧中的active_targets对应的id：

> active_targets: [1, 2, 3, 4, 5, 6, 7, 8]

第４帧中的active_targets对应的id：

> active_targets: [1, 2, 3, 4, 5, 6, 7, 8]

把这些activate target之前保存的feature (之前每帧只要能匹配上，都会把与之匹配的det的feature 保存下来)，用于更新卡尔曼滤波的distance metric。

因为当前帧不仅有检测结果，还有已经确认的trk,这些trk的跟踪结果也会画在图上(有11个)。同时，已经确认的trk的跟踪结果也会保存在txt中。.

 因为当前帧不仅有检测结果，还有已经确认的trk,这些trk的跟踪结果也会画在图上(有11个)。同时，已经确认的trk的跟踪结果也会保存在txt中。
![frame04](https://github.com/WangJun-ZJUT/Computer-Vision/blob/master/people/MaoJL/DeepSORT/img/005.jpg)

![txt04](https://github.com/WangJun-ZJUT/Computer-Vision/blob/master/people/MaoJL/DeepSORT/img/006.jpg)

