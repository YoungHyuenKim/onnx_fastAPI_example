import numpy as np
import cv2
import random


def xywh2xyxy(x):
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def basic_nms(bbox: np.ndarray, conf_thd=0.3, iou_thd=0.5, method="soft", sigma=0.5):
    """

    :param bbox: N x 6, xyxy confidence id
    :param conf_thd: float
    :param iou_thd: float
    :param method: str ["soft","linear", "hard"]
    :param sigma: float, use in soft nms
    :return: filtered object
    """
    N = len(bbox)
    x1 = bbox[:, 0]
    y1 = bbox[:, 1]
    x2 = bbox[:, 2]
    y2 = bbox[:, 3]
    scores = bbox[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    for i in range(N):
        # intermediate parameters for later parameters exchange
        tscore = scores[i].copy()
        pos = i + 1

        if i != N - 1:
            maxpos = np.argmax(scores[pos:])
            maxscore = np.max(scores[pos:])
            if tscore < maxscore:
                bbox[i], bbox[maxpos + pos] = bbox[maxpos + pos].copy(), bbox[i].copy()
                scores[i], scores[maxpos + pos] = scores[maxpos + pos].copy(), scores[i].copy()
                areas[i], areas[maxpos + pos] = areas[maxpos +pos].copy(), areas[i].copy()

        # IoU calculate
        xx1 = np.maximum(bbox[i, 0], bbox[pos:, 0])
        yy1 = np.maximum(bbox[i, 1], bbox[pos:, 1])
        xx2 = np.minimum(bbox[i, 2], bbox[pos:, 2])
        yy2 = np.minimum(bbox[i, 3], bbox[pos:, 3])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[pos:] - inter)

        # Gaussian decay
        if method == "soft":
            _weight = np.exp(-(ovr * ovr) / sigma)
            weight = np.where(ovr > iou_thd, 0, _weight)
        elif method == "linear":
            _weight = 1 - ovr
            weight = np.where(ovr > iou_thd, 0, _weight)
        else:
            weight = np.where(ovr > iou_thd, 0, 1)
        scores[pos:] = weight * scores[pos:]

    # select the boxes and keep the corresponding indexes
    keep_index = scores > conf_thd

    return bbox[keep_index]


def nms(model_outputs, conf_thd=0.3, iou_thd=0.5):
    """

    :param model_outputs: N x (5[xywh objectness]  +classes)
    :param conf_thd: float (0~1)
    :param iou_thd: float (0~1)
    :return: np.ndraary [M, 6] , xyxy conf id
    """
    max_wh = 7680  # for offset_by_classes

    assert 0 <= conf_thd <= 1, f'Invalid Confidence threshold {conf_thd}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thd <= 1, f'Invalid IoU {iou_thd}, valid values are between 0.0 and 1.0'

    xc = model_outputs[..., 4] > conf_thd

    x = model_outputs[xc]  # objectness threshold confidence

    x[:, 5:] *= x[:, 4:5]

    box = xywh2xyxy(x[:, :4])
    score = np.max(x[:, 4:5], axis=1).reshape(-1, 1)
    class_idx = np.argmax(x[:, 5:], axis=1).reshape(-1, 1)
    offset_by_classes = class_idx * max_wh
    shift_object = np.concatenate([box+offset_by_classes, score, class_idx], axis=1)

    nms_object = basic_nms(shift_object, iou_thd=iou_thd, method="soft")
    nms_object[:,:4] = nms_object[:,:4] - nms_object[:, 5:] * max_wh # restore offset
    return nms_object


def draw_bboxes(img, bboxes, conf=None, label_idx=None, color=None, thickness=3):
    """

    :param img:
    :param bboxes: np.ndarray
    :param conf: np.ndarray or list [N,1] or [N]
    :param label_idx: np.ndarray or list [N,1] or [N]
    :return:
    """
    bboxes = bboxes.astype(int)
    for bbox in bboxes:
        x1, y1, x2, y2 = map(lambda x: int(x), bbox)
        color = tuple(random.choices(range(256), k=3)) if color is None else color
        img = cv2.rectangle(img, pt1=(x1, y1), pt2=(x2, y2), color=color, thickness=thickness)
    if label_idx is not None:
        # TODO:: put text...
        pass
    if conf is not None:
        # TODO:: put text...
        pass
    return img
