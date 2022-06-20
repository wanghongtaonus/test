import torch
from torch.nn import functional as F
import torch.nn as nn

try:
    from itertools import ifilterfalse
except ImportError:  # py3k
    from itertools import filterfalse as ifilterfalse
import numpy as np


def balanced_cross_entropy(logit_pixel, truth_pixel, balanced=True):
    logit = logit_pixel.view(-1)
    truth = truth_pixel.view(-1)
    assert (logit.shape == truth.shape)

    loss = F.binary_cross_entropy_with_logits(logit, truth, reduction='none')

    if balanced:
        pos = (truth > 0.5).float()
        neg = (truth < 0.5).float()
        pos_weight = pos.sum().item() + 1e-12
        neg_weight = neg.sum().item() + 1e-12
        loss = (0.25 * pos * loss / pos_weight + 0.75 * neg * loss / neg_weight).sum()
    else:
        loss = loss.mean()

    return loss


def binary_sigmoid_focal_loss(pred,
                              target,
                              weight=None,
                              gamma=2.0,
                              alpha=0.25,
                              reduction='mean',
                              avg_factor=None):
    # pred_sigmoid = pred.sigmoid()
    pred_sigmoid = pred
    target = target.type_as(pred)
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * pt.pow(gamma)
    if True:
        for idx in range(target.shape[0]):
            up_lines = 20
            _loss_weight = target[idx].clone().detach()
            point_y_top = None
            point_y_bot = None
            try:
                point_y_top = int(torch.nonzero(torch.sum(target[idx], axis=1))[0][0])
                point_y_bot = int(torch.nonzero(torch.sum(target[idx], axis=1))[-1][0])
                point_y_top = point_y_top - up_lines if point_y_top - up_lines > 0 else 0
                point_y_bot = point_y_bot + up_lines if point_y_bot + up_lines < target.shape[1] else target.shape[1]
            except Exception as e:
                continue
            if point_y_bot and point_y_top:
                _loss_weight[point_y_top: point_y_top+up_lines*2, :] *= 2
                _loss_weight[point_y_bot-up_lines * 2: point_y_bot, :] *= 2
                focal_weight[idx] = focal_weight[idx] * _loss_weight

    loss = F.binary_cross_entropy_with_logits(
        pred, target, reduction='none') * focal_weight
    # loss = F.binary_cross_entropy(
    #     pred, target, reduction='none') * focal_weight
    if weight is not None:
        loss = loss * weight

    if avg_factor is None:
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            loss = loss.sum() / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')

    return loss


def binary_dice_loss(logits, targets, smooth=1, reduction='mean'):
    logits = logits.sigmoid()
    batch_size = logits.shape[0]
    logit = logits.contiguous().view(batch_size, -1)
    target = targets.contiguous().view(batch_size, -1).float()

    intersection = torch.sum(logit * target, dim=1) + smooth
    union = torch.sum(logit.pow(2) + target.pow(2), dim=1) + smooth
    loss = 1.0 - 2.0 * intersection / union
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'none':
        return loss
    else:
        raise Exception('Unexpected reduction {}'.format(reduction))


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def iou_binary(preds, labels, EMPTY=1., ignore=None, per_image=True):
    """
    IoU for foreground class
    binary: 1 foreground, 0 background
    """
    if not per_image:
        preds, labels = (preds,), (labels,)
    ious = []
    for pred, label in zip(preds, labels):
        intersection = ((label == 1) & (pred == 1)).sum()
        union = ((label == 1) | ((pred == 1) & (label != ignore))).sum()
        if not union:
            iou = EMPTY
        else:
            iou = float(intersection) / float(union)
        ious.append(iou)
    iou = mean(ious)  # mean accross images if per_image
    return 100 * iou


def iou(preds, labels, C, EMPTY=1., ignore=None, per_image=False):
    """
    Array of IoU for each (non ignored) class
    """
    if not per_image:
        preds, labels = (preds,), (labels,)
    ious = []
    for pred, label in zip(preds, labels):
        iou = []
        for i in range(C):
            if i != ignore:  # The ignored label is sometimes among predicted classes (ENet - CityScapes)
                intersection = ((label == i) & (pred == i)).sum()
                union = ((label == i) | ((pred == i) & (label != ignore))).sum()
                if not union:
                    iou.append(EMPTY)
                else:
                    iou.append(float(intersection) / float(union))
        ious.append(iou)
    ious = [mean(iou) for iou in zip(*ious)]  # mean accross images if per_image
    return 100 * np.array(ious)


# --------------------------- BINARY LOSSES ---------------------------


def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        loss = mean(lovasz_hinge_flat(*flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore))
                    for log, lab in zip(logits, labels))
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.
    signs = 2. * labels.float() - 1.
    errors = (1. - logits * signs)
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), grad)
    return loss


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = (labels != ignore)
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels


# --------------------------- MULTICLASS LOSSES ---------------------------


def lovasz_softmax(probas, labels, classes='present', per_image=False, ignore=None):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
              Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
    """
    if per_image:
        loss = mean(lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes)
                    for prob, lab in zip(probas, labels))
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), classes=classes)
    return loss


def lovasz_softmax_flat(probas, labels, classes='present'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    if probas.numel() == 0:
        # only void pixels, the gradients should be 0
        return probas * 0.
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float()  # foreground for class c
        if (classes is 'present' and fg.sum() == 0):
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (fg - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, lovasz_grad(fg_sorted)))
    return mean(losses)


def flatten_probas(probas, labels, ignore=None):
    """
    Flattens predictions in the batch
    """
    if probas.dim() == 3:
        # assumes output of a sigmoid layer
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = (labels != ignore)
    vprobas = probas[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels


# --------------------------- HELPER FUNCTIONS ---------------------------
def isnan(x):
    return x != x


def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


def focal_dice_lovasz_binary_loss_with_logit(logits, targets):
    assert logits.shape == targets.shape, 'output & target shape do not match'

    focal_loss = binary_sigmoid_focal_loss(logits, targets)

    dice_loss = binary_dice_loss(logits, targets)

    lovasz_loss = lovasz_hinge(logits, targets)

    return (focal_loss*1.5 + dice_loss + lovasz_loss) / 3


def focal_dice_binary_loss_with_logit(logits, targets, loss_weight):
    assert len(loss_weight) == 2
    assert logits.shape == targets.shape, 'output & target shape do not match'

    focal_loss = binary_sigmoid_focal_loss(logits, targets)

    dice_loss = binary_dice_loss(logits, targets)

    return focal_loss*loss_weight[0] + dice_loss*loss_weight[1]


class FDLMultiLabelComboLoss(nn.Module):
    def __init__(self, loss_weight, ignore_index=[]):
        super(FDLMultiLabelComboLoss, self).__init__()
        self.ignore_index = ignore_index
        self.loss_weight = loss_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.permute(dims=(0, 3, 1, 2))
        targets = targets.contiguous()
        assert logits.shape == targets.shape, 'output & target shape do not match'
        total_loss = 0
        num_class = targets.shape[1]
        for i in range(num_class):
            if i not in self.ignore_index:
                # loss = focal_dice_binary_loss_with_logit(logits[:, i], targets[:, i], self.loss_weight)
                loss = focal_dice_lovasz_binary_loss_with_logit(logits[:, i], targets[:, i])
                total_loss += loss
        loss = total_loss / (num_class - len(self.ignore_index))
        return loss
