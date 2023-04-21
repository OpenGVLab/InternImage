# ==============================================================================
# Binaries and/or source for the following packages or projects 
# are presented under one or more of the following open source licenses:
# f_score.py    The OpenLane-V2 Dataset Authors    Apache License, Version 2.0
#
# Contact wanghuijie@pjlab.org.cn if you have any issue.
#
# Adapted from:
# https://github.com/OpenPerceptionX/OpenLane/tree/0aaf62045e897d2b20ecf1357ae7742634b8f972/eval/LANE_evaluation/lane3d
#
# Copyright (c) 2023 The OpenLane-v2 Dataset Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
Description: This code is to evaluate 3D lane detection. The optimal matching between ground-truth set and predicted 
    set of lanes are sought via solving a min cost flow.
Evaluation metrics includes:
    F-scores
    x error close (0 - 40 m)
    x error far (0 - 100 m)
    z error close (0 - 40 m)
    z error far (0 - 100 m)
"""


import numpy as np
from scipy.interpolate import interp1d
from ortools.graph import pywrapgraph


def resample_laneline_in_x(input_lane, steps, out_vis=False):
    """
        Interpolate y, z values at each anchor grid, including those beyond the range of input lnae x range
    :param input_lane: N x 2 or N x 3 ndarray, one row for a point (x, y, z-optional).
                       It requires y values of input lane in ascending order
    :param steps: a vector of steps
    :param out_vis: whether to output visibility indicator which only depends on input y range
    :return:
    """

    # at least two points are included
    assert(input_lane.shape[0] >= 2)

    x_min = np.min(input_lane[:, 0])-5
    x_max = np.max(input_lane[:, 0])+5

    if input_lane.shape[1] < 3:
        input_lane = np.concatenate([input_lane, np.zeros([input_lane.shape[0], 1], dtype=np.float32)], axis=1)

    f_y = interp1d(input_lane[:, 0], input_lane[:, 1], fill_value="extrapolate")
    f_z = interp1d(input_lane[:, 0], input_lane[:, 2], fill_value="extrapolate")

    y_values = f_y(steps)
    z_values = f_z(steps)

    if out_vis:
        output_visibility = np.logical_and(steps >= x_min, steps <= x_max)
        return y_values, z_values, output_visibility.astype(np.float32) + 1e-9
    return y_values, z_values

def SolveMinCostFlow(adj_mat, cost_mat):
    """
        Solving an Assignment Problem with MinCostFlow"
    :param adj_mat: adjacency matrix with binary values indicating possible matchings between two sets
    :param cost_mat: cost matrix recording the matching cost of every possible pair of items from two sets
    :return:
    """

    # Instantiate a SimpleMinCostFlow solver.
    min_cost_flow = pywrapgraph.SimpleMinCostFlow()
    # Define the directed graph for the flow.

    cnt_1, cnt_2 = adj_mat.shape
    cnt_nonzero_row = int(np.sum(np.sum(adj_mat, axis=1) > 0))
    cnt_nonzero_col = int(np.sum(np.sum(adj_mat, axis=0) > 0))

    # prepare directed graph for the flow
    start_nodes = np.zeros(cnt_1, dtype=np.int32).tolist() +\
                  np.repeat(np.array(range(1, cnt_1+1)), cnt_2).tolist() + \
                  [i for i in range(cnt_1+1, cnt_1 + cnt_2 + 1)]
    end_nodes = [i for i in range(1, cnt_1+1)] + \
                np.repeat(np.array([i for i in range(cnt_1+1, cnt_1 + cnt_2 + 1)]).reshape([1, -1]), cnt_1, axis=0).flatten().tolist() + \
                [cnt_1 + cnt_2 + 1 for i in range(cnt_2)]
    capacities = np.ones(cnt_1, dtype=np.int32).tolist() + adj_mat.flatten().astype(np.int32).tolist() + np.ones(cnt_2, dtype=np.int32).tolist()
    costs = (np.zeros(cnt_1, dtype=np.int32).tolist() + cost_mat.flatten().astype(np.int32).tolist() + np.zeros(cnt_2, dtype=np.int32).tolist())
    # Define an array of supplies at each node.
    supplies = [min(cnt_nonzero_row, cnt_nonzero_col)] + np.zeros(cnt_1 + cnt_2, dtype=np.int32).tolist() + [-min(cnt_nonzero_row, cnt_nonzero_col)]
    # supplies = [min(cnt_1, cnt_2)] + np.zeros(cnt_1 + cnt_2, dtype=np.int).tolist() + [-min(cnt_1, cnt_2)]
    source = 0
    sink = cnt_1 + cnt_2 + 1

    # Add each arc.
    for i in range(len(start_nodes)):
        min_cost_flow.AddArcWithCapacityAndUnitCost(start_nodes[i], end_nodes[i],
                                                    capacities[i], costs[i])

    # Add node supplies.
    for i in range(len(supplies)):
        min_cost_flow.SetNodeSupply(i, supplies[i])

    match_results = []
    # Find the minimum cost flow between node 0 and node 10.
    if min_cost_flow.Solve() == min_cost_flow.OPTIMAL:
        # print('Total cost = ', min_cost_flow.OptimalCost())
        # print()
        for arc in range(min_cost_flow.NumArcs()):

            # Can ignore arcs leading out of source or into sink.
            if min_cost_flow.Tail(arc)!=source and min_cost_flow.Head(arc)!=sink:

                # Arcs in the solution have a flow value of 1. Their start and end nodes
                # give an assignment of worker to task.

                if min_cost_flow.Flow(arc) > 0:
                    # print('set A item %d assigned to set B item %d.  Cost = %d' % (
                    #     min_cost_flow.Tail(arc)-1,
                    #     min_cost_flow.Head(arc)-cnt_1-1,
                    #     min_cost_flow.UnitCost(arc)))
                    match_results.append([min_cost_flow.Tail(arc)-1,
                                          min_cost_flow.Head(arc)-cnt_1-1,
                                          min_cost_flow.UnitCost(arc)])
    else:
        print('There was an issue with the min cost flow input.')

    return match_results

class LaneEval(object):
    def __init__(self):        
        self.x_samples = np.linspace(-50, 50, num=100, endpoint=False)
        self.dist_th = 1.5
        self.ratio_th = 0.75

    def bench(self, pred_lanes, pred_category, gt_lanes, gt_category):
        """
            Matching predicted lanes and ground-truth lanes in their IPM projection, ignoring z attributes.
            x error, y_error, and z error are all considered, although the matching does not rely on z
            The input of prediction and ground-truth lanes are in ground coordinate, x-right, y-forward, z-up
            The fundamental assumption is: 1. there are no two points from different lanes with identical x, y
                                              but different z's
                                           2. there are no two points from a single lane having identical x, y
                                              but different z's
            If the interest area is within the current drivable road, the above assumptions are almost always valid.
        :param pred_lanes: N X 2 or N X 3 lists depending on 2D or 3D
        :param gt_lanes: N X 2 or N X 3 lists depending on 2D or 3D
        :return:
        """

        r_lane, p_lane, c_lane = 0., 0., 0.
        
        gt_lanes = [lane for lane in gt_lanes if lane.shape[0] > 1]

        # only consider those pred lanes overlapping with sampling range
        pred_category = [pred_category[k] for k, lane in enumerate(pred_lanes)
                        if lane[0, 0] < self.x_samples[-1] and lane[-1, 0] > self.x_samples[0]]
        pred_lanes = [lane for lane in pred_lanes if lane[0, 0] < self.x_samples[-1] and lane[-1, 0] > self.x_samples[0]]

        pred_category = [pred_category[k] for k, lane in enumerate(pred_lanes) if lane.shape[0] > 1]
        pred_lanes = [lane for lane in pred_lanes if lane.shape[0] > 1]

        # only consider those gt lanes overlapping with sampling range
        gt_category = [gt_category[k] for k, lane in enumerate(gt_lanes)
                        if lane[0, 0] < self.x_samples[-1] and lane[-1, 0] > self.x_samples[0]]
        gt_lanes = [lane for lane in gt_lanes if lane[0, 0] < self.x_samples[-1] and lane[-1, 0] > self.x_samples[0]]

        gt_category = [gt_category[k] for k, lane in enumerate(gt_lanes) if lane.shape[0] > 1]
        gt_lanes = [lane for lane in gt_lanes if lane.shape[0] > 1]

        cnt_gt = len(gt_lanes)
        cnt_pred = len(pred_lanes)

        gt_visibility_mat = np.zeros((cnt_gt, 100))
        pred_visibility_mat = np.zeros((cnt_pred, 100))

        # resample gt and pred at x_samples
        for i in range(cnt_gt):
            min_x = np.min(np.array(gt_lanes[i])[:, 0])
            max_x = np.max(np.array(gt_lanes[i])[:, 0])
            y_values, z_values, visibility_vec = resample_laneline_in_x(np.array(gt_lanes[i]), self.x_samples, out_vis=True)
            gt_lanes[i] = np.vstack([y_values, z_values]).T
            gt_visibility_mat[i, :] = np.logical_and(self.x_samples >= min_x, self.x_samples <= max_x)
            gt_visibility_mat[i, :] = np.logical_and(gt_visibility_mat[i, :], visibility_vec)

        for i in range(cnt_pred):
            # # ATTENTION: ensure y mono increase before interpolation: but it can reduce size
            # pred_lanes[i] = make_lane_y_mono_inc(np.array(pred_lanes[i]))
            # pred_lane = prune_3d_lane_by_range(np.array(pred_lanes[i]), self.x_min, self.x_max)
            min_x = np.min(np.array(pred_lanes[i])[:, 0])
            max_x = np.max(np.array(pred_lanes[i])[:, 0])
            y_values, z_values, visibility_vec = resample_laneline_in_x(np.array(pred_lanes[i]), self.x_samples, out_vis=True)
            pred_lanes[i] = np.vstack([y_values, z_values]).T
            pred_visibility_mat[i, :] = np.logical_and(self.x_samples >= min_x, self.x_samples <= max_x)
            pred_visibility_mat[i, :] = np.logical_and(pred_visibility_mat[i, :], visibility_vec)
            # pred_visibility_mat[i, :] = np.logical_and(x_values >= self.x_min, x_values <= self.x_max)

        # at least two-points for both gt and pred
        gt_lanes = [gt_lanes[k] for k in range(cnt_gt) if np.sum(gt_visibility_mat[k, :]) > 1]
        gt_category = [gt_category[k] for k in range(cnt_gt) if np.sum(gt_visibility_mat[k, :]) > 1]
        gt_visibility_mat = gt_visibility_mat[np.sum(gt_visibility_mat, axis=-1) > 1, :]
        cnt_gt = len(gt_lanes)

        pred_lanes = [pred_lanes[k] for k in range(cnt_pred) if np.sum(pred_visibility_mat[k, :]) > 1]
        pred_category = [pred_category[k] for k in range(cnt_pred) if np.sum(pred_visibility_mat[k, :]) > 1]
        pred_visibility_mat = pred_visibility_mat[np.sum(pred_visibility_mat, axis=-1) > 1, :]
        cnt_pred = len(pred_lanes)

        adj_mat = np.zeros((cnt_gt, cnt_pred), dtype=int)
        cost_mat = np.zeros((cnt_gt, cnt_pred), dtype=int)
        cost_mat.fill(1000)
        num_match_mat = np.zeros((cnt_gt, cnt_pred), dtype=float)

        # compute curve to curve distance
        for i in range(cnt_gt):
            for j in range(cnt_pred):
                y_dist = np.abs(gt_lanes[i][:, 0] - pred_lanes[j][:, 0])
                z_dist = np.abs(gt_lanes[i][:, 1] - pred_lanes[j][:, 1])

                # apply visibility to penalize different partial matching accordingly
                both_visible_indices = np.logical_and(gt_visibility_mat[i, :] >= 0.5, pred_visibility_mat[j, :] >= 0.5)
                both_invisible_indices = np.logical_and(gt_visibility_mat[i, :] < 0.5, pred_visibility_mat[j, :] < 0.5)
                other_indices = np.logical_not(np.logical_or(both_visible_indices, both_invisible_indices))
                
                euclidean_dist = np.sqrt(y_dist ** 2 + z_dist ** 2)
                euclidean_dist[both_invisible_indices] = 0
                euclidean_dist[other_indices] = self.dist_th

                # if np.average(euclidean_dist) < 2*self.dist_th: # don't prune here to encourage finding perfect match
                num_match_mat[i, j] = np.sum(euclidean_dist < self.dist_th) - np.sum(both_invisible_indices)
                adj_mat[i, j] = 1
                # ATTENTION: use the sum as int type to meet the requirements of min cost flow optimization (int type)
                # using num_match_mat as cost does not work?
                # make sure cost is not set to 0 when it's smaller than 1
                cost_ = np.sum(euclidean_dist)
                if cost_<1 and cost_>0:
                    cost_ = 1
                else:
                    cost_ = (cost_).astype(int)
                cost_mat[i, j] = cost_
                # cost_mat[i, j] = np.sum(euclidean_dist)
                # cost_mat[i, j] = num_match_mat[i, j]

        # solve bipartite matching vis min cost flow solver
        match_results = SolveMinCostFlow(adj_mat, cost_mat)
        match_results = np.array(match_results)

        # only a match with avg cost < self.dist_th is consider valid one
        match_gt_ids = []
        match_pred_ids = []
        match_num = 0
        if match_results.shape[0] > 0:
            for i in range(len(match_results)):
                if match_results[i, 2] < self.dist_th * self.x_samples.shape[0]:
                    match_num += 1
                    gt_i = match_results[i, 0]
                    pred_i = match_results[i, 1]
                    # consider match when the matched points is above a ratio
                    if num_match_mat[gt_i, pred_i] / np.sum(gt_visibility_mat[gt_i, :]) >= self.ratio_th:
                        r_lane += 1
                        match_gt_ids.append(gt_i)
                    if num_match_mat[gt_i, pred_i] / np.sum(pred_visibility_mat[pred_i, :]) >= self.ratio_th:
                        p_lane += 1
                        match_pred_ids.append(pred_i)
                    if pred_category != []:
                        if pred_category[pred_i] == gt_category[gt_i] or (pred_category[pred_i]==20 and gt_category[gt_i]==21):
                            c_lane += 1    # category matched num
        return r_lane, p_lane, c_lane, cnt_gt, cnt_pred, match_num

    def bench_one_submit(self, gts, preds):

        laneline_stats = []
        laneline_x_error_close = []
        laneline_x_error_far = []
        laneline_z_error_close = []
        laneline_z_error_far = []
        for token, pred in preds.items():

            pred_lanelines = pred['lane_centerline']
            pred_lanes = [lane['points'] for i, lane in enumerate(pred_lanelines)]
            pred_category = [np.int8(1) for i, lane in enumerate(pred_lanelines)]

            gt = gts[token]

            # extrinsic
            # evaluate lanelines
            # cam_extrinsics = np.array(gt['extrinsic'])
            # # Re-calculate extrinsic matrix based on ground coordinate
            # R_vg = np.array([[0, 1, 0],
            #                     [-1, 0, 0],
            #                     [0, 0, 1]], dtype=float)
            # R_gc = np.array([[1, 0, 0],
            #                     [0, 0, 1],
            #                     [0, -1, 0]], dtype=float)
            # cam_extrinsics[:3, :3] = np.matmul(np.matmul(
            #                             np.matmul(np.linalg.inv(R_vg), cam_extrinsics[:3, :3]),
            #                                 R_vg), R_gc)
            # cam_extrinsics[0:2, 3] = 0.0

            gt_lanes_packed = gt['lane_centerline']

            gt_lanes, gt_category = [], []
            for j, gt_lane_packed in enumerate(gt_lanes_packed):
                # A GT lane can be either 2D or 3D
                # if a GT lane is 3D, the height is intact from 3D GT, so keep it intact here too
                lane = gt_lane_packed['points']

                # extrinsic
                # lane = np.vstack((lane, np.ones((1, lane.shape[1]))))
                # cam_representation = np.linalg.inv(
                #                         np.array([[0, 0, 1, 0],
                #                                   [-1, 0, 0, 0],
                #                                   [0, -1, 0, 0],
                #                                   [0, 0, 0, 1]], dtype=float))
                # lane = np.matmul(cam_extrinsics, np.matmul(cam_representation, lane))
                # lane = lane[0:3, :].T

                gt_lanes.append(lane)
                gt_category.append(np.int8(1))

            # N to N matching of lanelines
            r_lane, p_lane, c_lane, cnt_gt, cnt_pred, match_num = self.bench(pred_lanes,
                                                                             pred_category, 
                                                                             gt_lanes,
                                                                             gt_category,
                                                                             )
            laneline_stats.append(np.array([r_lane, p_lane, c_lane, cnt_gt, cnt_pred, match_num]))
            # consider x_error z_error only for the matched lanes
            # if r_lane > 0 and p_lane > 0:

        output_stats = []
        laneline_stats = np.array(laneline_stats)
        laneline_x_error_close = np.array(laneline_x_error_close)
        laneline_x_error_far = np.array(laneline_x_error_far)
        laneline_z_error_close = np.array(laneline_z_error_close)
        laneline_z_error_far = np.array(laneline_z_error_far)

        if np.sum(laneline_stats[:, 3])!= 0:
            R_lane = np.sum(laneline_stats[:, 0]) / (np.sum(laneline_stats[:, 3]))
        else:
            R_lane = np.sum(laneline_stats[:, 0]) / (np.sum(laneline_stats[:, 3]) + 1e-6)   # recall = TP / (TP+FN)
        if np.sum(laneline_stats[:, 4]) != 0:
            P_lane = np.sum(laneline_stats[:, 1]) / (np.sum(laneline_stats[:, 4]))
        else:
            P_lane = np.sum(laneline_stats[:, 1]) / (np.sum(laneline_stats[:, 4]) + 1e-6)   # precision = TP / (TP+FP)
        if np.sum(laneline_stats[:, 5]) != 0:
            C_lane = np.sum(laneline_stats[:, 2]) / (np.sum(laneline_stats[:, 5]))
        else:
            C_lane = np.sum(laneline_stats[:, 2]) / (np.sum(laneline_stats[:, 5]) + 1e-6)   # category_accuracy
        if R_lane + P_lane != 0:
            F_lane = 2 * R_lane * P_lane / (R_lane + P_lane)
        else:
            F_lane = 2 * R_lane * P_lane / (R_lane + P_lane + 1e-6)

        output_stats.append(F_lane)

        return output_stats[0]

f1 = LaneEval()
