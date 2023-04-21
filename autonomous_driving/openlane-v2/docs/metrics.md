# Metrics

## OpenLane-V2 Score
To evaluate performances on different aspects of the task, several metrics are adopted:
- $\text{DET}_{l}$ for mAP on directed lane centerlines,
- $\text{DET}_{t}$ for mAP on traffic elements,
- $\text{TOP}_{ll}$ for mAP on topology among lane centerlines,
- $\text{TOP}_{lt}$ for mAP on topology between lane centerlines and traffic elements.

We consolidate the above metrics by computing an average of them, resulting in the **OpenLane-V2 Score (OLS)**.

### Lane Centerline
We adopt the average precision (AP) but define a match of lane centerlines by considering the discrete Frechet distance in the 3D space.
The mAP for lane centerlines is averaged over match thresholds of $\\{1.0, 2.0, 3.0\\}$ on the similarity measure.

### Traffic Element
Similarly, we use AP to evaluate the task of traffic element detection.
We consider IoU distance as the affinity measure with a match threshold of $0.75$.
Besides, traffic elements have their own attribute.
For instance, a traffic light can be red or green, which indicates the drivable state of the lane.
Therefore, the mAP is then averaged over attributes.

### Topology
The topology metrics estimate the goodness of the relationship among lane centerlines and the relationship between lane centerlines and traffic elements.
To formulate the task of topology prediction as a link prediction problem, we first determine a match of ground truth and predicted vertices (lane centerlines and traffic elements) in the relationship graph.
We choose Frechet and IoU distance for the lane centerline and traffic element respectively.
Also, the metric is average over different recalls.

We adopt mAP from link prediction, which is defined as a mean of APs over all vertices. 
Two vertices are regarded as connected if the predicted confidence of the edge is greater than $0.5$.
The AP of a vertex is obtained by ranking all predicted edges and calculating the accumulative mean of the precisions:

$$
mAP = \frac{1}{|V|} \sum_{v \in V} \frac{\sum_{\hat{n} \in \hat{N}(v)} P(\hat{n}) \mathbb{1}(\hat{n} \in N(v))}{|N(v)|},
$$

where $N(v)$ denotes ordered list of neighbors of vertex $v$ ranked by confidence and $P(v)$ is the precision of the $i$-th vertex $v$ in the ordered list.

Given ground truth and predicted connectivity of lane centerlines, the mAP is calculated on $G^{l} = (V^{l}, E^{l})$ and $\hat{G}^{l} = (\hat{V}^{l}, \hat{E}^{l})$.
As the given graphs are directed, e.g., the ending point of a lane centerline is connected to the starting point of the next lane centerline, we take the mean of mAP over graphs with only in-going or out-going edges.

To evaluate the predicted topology between lane centerlines and traffic elements, we ignore the relationship among lane centerlines.
The relationship among traffic elements is also not taken into consideration.
Thus this can be seen as a link prediction problem on a bipartite undirected graph that $G = (V^{l} \cup V^{t}, E)$ and $\hat{G} = (\hat{V}^{l} \cup \hat{V}^{t}, \hat{E})$.

## Distances
To measure the similarity between ground truth and predicted instances, we adopt Frechet and IoU distances for directed curves and 2D bounding boxes respectively.

### Frechet Distance
Discrete Frechet distance measures the geometric similarity of two ordered lists of points.
Given a pair of curves, namely a ground truth $v = (p_1, ..., p_n)$ and a prediction $\hat{v} = (\hat{p}_1, ..., \hat{p}_k)$, a coupling $L$ is a sequence of distinct pairs between $v$ and $\hat{v}$:

$$
(p_{a_1} \ , \ \hat{p}_{b_1} \ ), ..., (p_{a_m} \ , \ \hat{p}_{b_m} \ ),
$$

where $a_1, ..., a_m$ and $b_1, ..., b_m$ are nondecreasing surjection such that $1 = a_1 \leq a_i \leq a_j \leq a_m = n$ and $1 = b_1 \leq b_i \leq b_j \leq b_m = k$ for all $i < j$. Then the norm $||L||$ of a coupling $L$ is the distance of the most dissimilar pair in $L$ that:

$$
||L|| = \mathop{max}_{i=1, ..., m} D(p_{a_i} \ , \ \hat{p}_{b_i} \ ).
$$

The Frechet distance of a pair of curves is the minimum norm of all possible coupling that:

$$
D_{Frechet}(v, \hat{v}) = min\\{||L|| \ | \ for \ all \ possible \ coupling \ L\\}.
$$

### IoU Distance
To preserve consistency to the distance mentioned above, we modify the common IoU (Intersection over Union) measure that:

$$
D_{IoU}(X, \hat{X}) = 1 - \frac{|X \cap \hat{X}|}{|X \cup \hat{X}|},
$$

where $X$ and $\hat{X}$ is the ground truth and predicted bounding box respectively.
