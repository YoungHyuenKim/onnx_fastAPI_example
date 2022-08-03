import numpy as np


def xywh2xyxy(x):
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def basic_soft_nms(bbox: np.ndarray, score: np.ndarray, iou_thd=0.5, method="soft", sigma=0.5):
    """

    :param bbox: N x 4
    :param score: N
    :param iou_thd: float
    :param method: str ["soft","linear", "hard"]
    :param sigma: float
    :return:
    """
    N = len(bbox)
    y1 = bbox[:, 0]
    x1 = bbox[:, 1]
    y2 = bbox[:, 2]
    x2 = bbox[:, 3]
    scores = score
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    for i in range(N):
        # intermediate parameters for later parameters exchange
        tscore = scores[i].copy()
        pos = i + 1

        if i != N - 1:
            maxpos = np.argmax(score[pos:], axis=1)
            maxscore = np.max(scores[pos:], axis=1)
            if tscore < maxscore:
                bbox[i], bbox[maxpos + i + 1] = bbox[maxpos + i + 1].copy(), bbox[i].copy()
                scores[i], scores[maxpos + i + 1] = scores[maxpos + i + 1].copy(), scores[i].copy()
                areas[i], areas[maxpos + i + 1] = areas[maxpos + i + 1].copy(), areas[i].copy()

        # IoU calculate
        yy1 = np.maximum(bbox[i, 0], bbox[pos:, 0])
        xx1 = np.maximum(bbox[i, 1], bbox[pos:, 1])
        yy2 = np.minimum(bbox[i, 2], bbox[pos:, 2])
        xx2 = np.minimum(bbox[i, 3], bbox[pos:, 3])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[pos:] - inter)

        # Gaussian decay
        if method == "soft":
            weight = np.exp(-(ovr * ovr) / sigma)
        elif method == "linear":
            weight = 1 - ovr
        else:
            weight = 0
        scores[pos:] = weight * scores[pos:]

    # select the boxes and keep the corresponding indexes
    keep = bbox[scores > iou_thd].int()

    return keep


def nms(model_outputs, conf_thd=0.3, iou_thd=0.5):
    """

    :param model_outputs: N x (5[xywh objectness]  +classes)
    :param conf_thd: float (0~1)
    :param iou_thd: float (0~1)
    :return: np.ndraary [M, 6] , xyxy conf id
    """
    max_wh = 7680 # for offset_by_classes
    nc = model_outputs.shape[-1] - 5

    assert 0 <= conf_thd <= 1, f'Invalid Confidence threshold {conf_thd}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thd <= 1, f'Invalid IoU {iou_thd}, valid values are between 0.0 and 1.0'

    xc = model_outputs[..., 4] > conf_thd

    x = model_outputs[xc]  # objectness threshold confidence

    x[:, 5:] *= x[:, 4:5]

    box = xywh2xyxy(x[:, :4])
    score = np.argmax(x[:, 4:5], axis=1)
    c = np.argmax(x[:, 5:], axis=1)

    x = np.concatenate([box, score, c], axis=0)

    offset_by_classes = x[:, 5:6] * max_wh
    bboxes, scores = x[:, :4] + offset_by_classes, x[:, 4]
    i = basic_soft_nms(bboxes, scores, iou_thd=iou_thd)

    return x[i]
