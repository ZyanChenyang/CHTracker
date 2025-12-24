import numpy as np

def iou_batch(bboxes1, bboxes2):
    """
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)
    
    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])                                      
        + (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1]) - wh)                                              
    return(o)

def cal_bc_dif_batch(bboxes1, bboxes2):  
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    bc2 = bboxes2[..., 4]
    bc1 = np.sqrt((bboxes1[..., 3] - bboxes1[..., 0]) ** 2 + (bboxes1[..., 2] - bboxes1[..., 1]) ** 2)
    return (abs(bc2 - bc1) / bc1)

def x_fake_iou(bboxes1, bboxes2, confidence, track_confidence):   
    for i in range(len(confidence)):
        if confidence[i] < 0.8:
            w_actual = bboxes1[i, 2] - bboxes1[i, 0]
            bboxes1[i, 0] -= w_actual * abs(0.8 - confidence[i]) / 4
            bboxes1[i, 2] += w_actual * abs(0.8 - confidence[i]) / 4

    for j in range(len(track_confidence)):
        if track_confidence[j] < 0.8:
            w_actual = bboxes2[j, 2] - bboxes2[j, 0]
            bboxes2[j, 0] -= w_actual * abs(0.8 - track_confidence[j]) / 4
            bboxes2[j, 2] += w_actual * abs(0.8 - track_confidence[j]) / 4

    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)
    
    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])                                      
        + (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1]) - wh)                                              
    return(o)

def y_fake_iou(bboxes1, bboxes2, confidence, track_confidence):   # detections, tracks， confidence
    for i in range(len(confidence)):
        if confidence[i] < 0.8:
            h_actual = bboxes1[i, 3] - bboxes1[i, 1]
            bboxes1[i, 1] -= h_actual * abs(0.8 - confidence[i]) / 4
            bboxes1[i, 3] += h_actual * abs(0.8 - confidence[i]) / 4

    for j in range(len(track_confidence)):
        if track_confidence[j] < 0.8:
            h_actual = bboxes2[j, 3] - bboxes2[j, 1]
            bboxes2[j, 1] -= h_actual * abs(0.8 - track_confidence[j]) / 4
            bboxes2[j, 3] += h_actual * abs(0.8 - track_confidence[j]) / 4

    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)
    
    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])                                      
        + (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1]) - wh)                                              
    return(o)

def hmiou(bboxes1, bboxes2):
    """
    Height_Modulated_IoU
    """
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    yy11 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    yy12 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])

    yy21 = np.minimum(bboxes1[..., 1], bboxes2[..., 1])
    yy22 = np.maximum(bboxes1[..., 3], bboxes2[..., 3])
    o = (yy12 - yy11) / (yy22 - yy21)

    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o *= wh / ((bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
        + (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1]) - wh)
    return (o)

def linear_assignment(cost_matrix, thresh=0.):
    try:        # [hgx0411] goes here!
        import lap
        if thresh != 0:
            _, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
        else:
            _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))

def cost_vel(Y, X, trackers, velocities, detections, previous_obs, vdc_weight):
    # Y, X = speed_direction_batch(detections, previous_obs)
    inertia_Y, inertia_X = velocities[:, 0], velocities[:, 1]
    inertia_Y = np.repeat(inertia_Y[:, np.newaxis], Y.shape[1], axis=1)
    inertia_X = np.repeat(inertia_X[:, np.newaxis], X.shape[1], axis=1)
    diff_angle_cos = inertia_X * X + inertia_Y * Y
    diff_angle_cos = np.clip(diff_angle_cos, a_min=-1, a_max=1)
    diff_angle = np.arccos(diff_angle_cos)
    diff_angle = (np.pi / 2.0 - np.abs(diff_angle)) / np.pi

    valid_mask = np.ones(previous_obs.shape[0])
    valid_mask[np.where(previous_obs[:, 4] < 0)] = 0

    # iou_matrix = iou_batch(detections, trackers)
    scores = np.repeat(detections[:, -1][:, np.newaxis], trackers.shape[0], axis=1)
    # iou_matrix = iou_matrix * scores # a trick sometiems works, we don't encourage this
    valid_mask = np.repeat(valid_mask[:, np.newaxis], X.shape[1], axis=1)

    angle_diff_cost = (valid_mask * diff_angle) * vdc_weight
    angle_diff_cost = angle_diff_cost.T
    angle_diff_cost = angle_diff_cost * scores
    return angle_diff_cost

def speed_direction_batch(dets, tracks):
    """
    batch formulation of function 'speed_direction', compute normalized speed from batch bboxes
    @param dets:
    @param tracks:
    @return: normalized speed in batch
    """
    tracks = tracks[..., np.newaxis]
    CX1, CY1 = (dets[:,0] + dets[:,2])/2.0, (dets[:,1]+dets[:,3])/2.0
    CX2, CY2 = (tracks[:,0] + tracks[:,2]) /2.0, (tracks[:,1]+tracks[:,3])/2.0
    dx = CX1 - CX2 
    dy = CY1 - CY2 
    norm = np.sqrt(dx**2 + dy**2) + 1e-6
    dx = dx / norm 
    dy = dy / norm
    return dy, dx # size: num_track x num_det

def speed_direction_batch_lt(dets, tracks):
    tracks = tracks[..., np.newaxis]
    CX1, CY1 = dets[:,0], dets[:,1]
    CX2, CY2 = tracks[:,0], tracks[:,1]
    dx = CX1 - CX2
    dy = CY1 - CY2
    norm = np.sqrt(dx**2 + dy**2) + 1e-6
    dx = dx / norm
    dy = dy / norm
    return dy, dx # size: num_track x num_det

def speed_direction_batch_rt(dets, tracks):
    tracks = tracks[..., np.newaxis]
    CX1, CY1 = dets[:,0], dets[:,3]
    CX2, CY2 = tracks[:,0], tracks[:,3]
    dx = CX1 - CX2
    dy = CY1 - CY2
    norm = np.sqrt(dx**2 + dy**2) + 1e-6
    dx = dx / norm
    dy = dy / norm
    return dy, dx # size: num_track x num_det

def speed_direction_batch_lb(dets, tracks):
    tracks = tracks[..., np.newaxis]
    CX1, CY1 = dets[:,2], dets[:,1]
    CX2, CY2 = tracks[:,2], tracks[:,1]
    dx = CX1 - CX2
    dy = CY1 - CY2
    norm = np.sqrt(dx**2 + dy**2) + 1e-6
    dx = dx / norm
    dy = dy / norm
    return dy, dx # size: num_track x num_det

def speed_direction_batch_rb(dets, tracks):
    tracks = tracks[..., np.newaxis]
    CX1, CY1 = dets[:,2], dets[:,3]
    CX2, CY2 = tracks[:,2], tracks[:,3]
    dx = CX1 - CX2
    dy = CY1 - CY2
    norm = np.sqrt(dx**2 + dy**2) + 1e-6
    dx = dx / norm
    dy = dy / norm
    return dy, dx # size: num_track x num_det

def associate_4_points_with_score(detections, trackers, iou_threshold, lt, rt, lb, rb, previous_obs, vdc_weight, iou_type=None, args=None):
    if (len(trackers) == 0):
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

    Y1, X1 = speed_direction_batch_lt(detections, previous_obs)
    Y2, X2 = speed_direction_batch_rt(detections, previous_obs)
    Y3, X3 = speed_direction_batch_lb(detections, previous_obs)
    Y4, X4 = speed_direction_batch_rb(detections, previous_obs)
    cost_lt = cost_vel(Y1, X1, trackers, lt, detections, previous_obs, vdc_weight)
    cost_rt = cost_vel(Y2, X2, trackers, rt, detections, previous_obs, vdc_weight)
    cost_lb = cost_vel(Y3, X3, trackers, lb, detections, previous_obs, vdc_weight)
    cost_rb = cost_vel(Y4, X4, trackers, rb, detections, previous_obs, vdc_weight)

    angle_diff_l = abs(cost_lt) + abs(cost_lb) 
    angle_diff_r = abs(cost_rt) + abs(cost_rb)
    angle_diff_t = abs(cost_lt) + abs(cost_rt)
    angle_diff_b = abs(cost_lb) + abs(cost_rb)
    angle_diff = np.minimum(np.minimum(np.minimum(angle_diff_l, angle_diff_r), angle_diff_t), angle_diff_b)

    iou_matrix = iou_type(detections, trackers)

    iou_matrix_x = x_fake_iou(detections, trackers, detections[:, -1], trackers[:, -1])
    iou_matrix_y = y_fake_iou(detections, trackers, detections[:, -1], trackers[:, -1])
    iou_matrix_soft = np.maximum(iou_matrix_x, iou_matrix_y)

    bc_dif = cal_bc_dif_batch(detections, trackers)

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-(iou_matrix - bc_dif * 0.5 + iou_matrix_soft * 0.05 - angle_diff * 0.3))
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if (d not in matched_indices[:, 0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if (t not in matched_indices[:, 1]):
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if (iou_matrix[m[0], m[1]] < iou_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if (len(matches) == 0):
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

def associate_4_points_with_score_with_reid(detections, trackers, iou_threshold, lt, rt, lb, rb, previous_obs, vdc_weight,
                                            iou_type=None, args=None,emb_cost=None, weights=(1.0, 0), thresh=0.8,
                                            long_emb_dists=None, with_longterm_reid=False,
                                            longterm_reid_weight=0.0, with_longterm_reid_correction=False,
                                            longterm_reid_correction_thresh=0.0, dataset="dancetrack"):
    if (len(trackers) == 0):
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

    Y1, X1 = speed_direction_batch_lt(detections, previous_obs)
    Y2, X2 = speed_direction_batch_rt(detections, previous_obs)
    Y3, X3 = speed_direction_batch_lb(detections, previous_obs)
    Y4, X4 = speed_direction_batch_rb(detections, previous_obs)
    cost_lt = cost_vel(Y1, X1, trackers, lt, detections, previous_obs, vdc_weight)
    cost_rt = cost_vel(Y2, X2, trackers, rt, detections, previous_obs, vdc_weight)
    cost_lb = cost_vel(Y3, X3, trackers, lb, detections, previous_obs, vdc_weight)
    cost_rb = cost_vel(Y4, X4, trackers, rb, detections, previous_obs, vdc_weight)

    angle_diff_l = abs(cost_lt) + abs(cost_lb) 
    angle_diff_r = abs(cost_rt) + abs(cost_rb)
    angle_diff_t = abs(cost_lt) + abs(cost_rt)
    angle_diff_b = abs(cost_lb) + abs(cost_rb)
    angle_diff = np.minimum(np.minimum(np.minimum(angle_diff_l, angle_diff_r), angle_diff_t), angle_diff_b)

    iou_matrix = iou_type(detections, trackers)

    iou_matrix_x = x_fake_iou(detections, trackers, detections[:, -1], trackers[:, -1])
    iou_matrix_y = y_fake_iou(detections, trackers, detections[:, -1], trackers[:, -1])
    iou_matrix_soft = np.maximum(iou_matrix_x, iou_matrix_y)

    bc_dif = cal_bc_dif_batch(detections, trackers)

    if min(iou_matrix.shape) > 0:
        if emb_cost is None:
            a = (iou_matrix > iou_threshold).astype(np.int32)
            if a.sum(1).max() == 1 and a.sum(0).max() == 1:
                matched_indices = np.stack(np.where(a), axis=1)
            else:
                matched_indices = linear_assignment(-(iou_matrix - bc_dif * 0.5 + iou_matrix_soft * 0.05 - angle_diff * 0.3))
        else:
            if not with_longterm_reid:
                matched_indices = linear_assignment(weights[0] * (-(iou_matrix  - bc_dif * 0.5 + iou_matrix_soft * 0.05 - angle_diff * 0.3)) + 
                                                    weights[1] * emb_cost)
            else:   # long-term reid feats
                matched_indices = linear_assignment(weights[0] * (-(iou_matrix  - bc_dif * 0.5 + iou_matrix_soft * 0.05 - angle_diff * 0.3)) +
                                                    weights[1] * emb_cost + longterm_reid_weight * long_emb_dists)

        if matched_indices.size == 0:
            matched_indices = np.empty(shape=(0, 2))
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if (d not in matched_indices[:, 0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if (t not in matched_indices[:, 1]):
            unmatched_trackers.append(t)

    # filter out matched with low IOU (and long-term ReID feats)
    matches = []
    iou_matrix_thre = iou_matrix
    if with_longterm_reid_correction:
        for m in matched_indices:
            if (emb_cost[m[0], m[1]] > longterm_reid_correction_thresh) and (iou_matrix_thre[m[0], m[1]] < iou_threshold):
                print("correction:", emb_cost[m[0], m[1]])
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1, 2))
    else:
        for m in matched_indices:
            if (iou_matrix_thre[m[0], m[1]] < iou_threshold):
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1, 2))

    if (len(matches) == 0):
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

# compute embedding distance and gating, borrowed and modified from FairMOT
from scipy.spatial.distance import cdist
def embedding_distance(tracks_feat, detections_feat, metric='cosine'):
    """
    :param tracks: list[KalmanBoxTracker]
    :param detections: list[KalmanBoxTracker]
    :param metric:
    :return: cost_matrix np.ndarray
    """

    cost_matrix = np.zeros((len(tracks_feat), len(detections_feat)), dtype=float)
    if cost_matrix.size == 0:
        return cost_matrix
    cost_matrix = np.maximum(0.0, cdist(tracks_feat, detections_feat, metric))  # Nomalized features, metric: cosine, [track_num, detection_num]
    return cost_matrix

chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919}

# [hgx0411] compute embedding distance and gating, borrowed and modified from FairMOT
def fuse_motion(cost_matrix, tracks, detections, only_position=False, lambda_=0.98):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = chi2inv95[gating_dim]
    for row, track in enumerate(tracks):
        gating_distance = track.kf.gating_distance(detections, only_position, metric='maha')
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
        cost_matrix[row] = lambda_ * cost_matrix[row] + (1 - lambda_) * gating_distance
    return cost_matrix

# [hgx0411] compute embedding distance and gating, borrowed and modified from FairMOT
import lap
def linear_assignment_appearance(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b

def fuse_score(cost_matrix, det_scores):
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = - cost_matrix
    det_scores = np.expand_dims(det_scores, axis=1).repeat(cost_matrix.shape[1], axis=1)
    fuse_sim = iou_sim * det_scores
    fuse_cost = - fuse_sim
    return fuse_cost