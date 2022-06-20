import torch
import torchcontrib
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

import pandas as pd
import numpy as np

from tqdm import tqdm
import heapq
from pathlib import Path
import collections

from .MulitLosses import FocalLoss, MultiDiceLoss, soft_dice_loss_multi
from .BinaryLosses import FDLMultiLabelComboLoss


class Learning():
    def __init__(self,
                 optimizer,
                 loss_fn,
                 loss_args,
                 seg_classes_num,
                 device,
                 n_epoches,
                 scheduler,
                 freeze_model,
                 grad_clip,
                 grad_accum,
                 early_stopping,
                 validation_frequency,
                 calculation_name,
                 best_checkpoint_folder,
                 checkpoints_history_folder,
                 checkpoints_topk,
                 main_logger,
                 fold_logger,
                 tf_logger=None,
                 use_fp_16=False,
                 use_swa=False,
                 ):
        self.main_logger = main_logger
        self.fold_logger = fold_logger
        self.tf_logger = tf_logger

        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.loss_args = loss_args
        self.seg_classes_num = seg_classes_num
        self.device = device
        self.n_epoches = n_epoches
        self.scheduler = scheduler
        self.freeze_model = freeze_model
        self.grad_clip = grad_clip
        self.grad_accum = grad_accum
        self.early_stopping = early_stopping
        self.validation_frequency = validation_frequency
        self.use_fp16 = use_fp_16
        self.use_swa = use_swa

        if self.use_swa:
            self.opt = torchcontrib.optim.SWA(self.optimizer)

        self.calculation_name = calculation_name
        self.best_checkpoint_path = Path(
            best_checkpoint_folder,
            '{}.pth'.format(self.calculation_name)
        )
        self.checkpoints_history_folder = Path(checkpoints_history_folder)
        self.checkpoints_topk = checkpoints_topk
        self.score_heap = []
        self.summary_file = Path(self.checkpoints_history_folder, 'summary_part0.csv')
        if self.summary_file.is_file():
            self.best_score = pd.read_csv(self.summary_file).best_metric.max()
            self.main_logger.info('Pretrained best score is {:.5}'.format(self.best_score))
            self.fold_logger.info('Pretrained best score is {:.5}'.format(self.best_score))
        else:
            self.best_score = 0
        self.best_epoch = -1

    def train_epoch(self, epoch, model, loader, loss_hist):
        tqdm_loader = tqdm(loader)
        continuous_loss_mean = 0

        for batch_idx, (img_id, imgs, labels,) in enumerate(tqdm_loader):
            loss, predicted = self.batch_train(model, imgs, labels, batch_idx)
            # just slide average
            continuous_loss_mean = (continuous_loss_mean * batch_idx + loss) / (batch_idx + 1)
            # print(current_loss_mean)
            current_lr = self.optimizer.param_groups[0]['lr']
            tqdm_loader.set_description('loss: {:.4} lr:{:.6}'.format(
                continuous_loss_mean, current_lr))

            loss_hist.append(loss)

            msg = 'epoch: {}/{}, batch: {}/{}, lr: {}, loss: {:.6f}, ' \
                  'continuous_loss_mean: {:.6f}, sliding_loss_mean: {:.6f}'
            msg = msg.format(epoch, self.n_epoches, batch_idx, len(tqdm_loader),
                             current_lr, loss, continuous_loss_mean, np.mean(loss_hist))
            self.fold_logger.info(msg)

            # make visualization with tensorboardX
            if self.tf_logger is not None:
                lr_log = {'learning_rate': current_lr}
                self.tf_logger.add_scalars('logs_lr', lr_log, (epoch - 1) * len(tqdm_loader) + batch_idx)
                loss_log = {'training_loss': np.mean(loss_hist)}
                self.tf_logger.add_scalars('logs_sliding_loss', loss_log, (epoch - 1) * len(tqdm_loader) + batch_idx)

        return continuous_loss_mean

    def batch_train(self, model, batch_imgs, batch_labels, batch_idx):
        batch_imgs, batch_labels = batch_imgs.to(self.device), batch_labels.to(self.device)
        predicted = model(batch_imgs)
        # loss = self.loss_fn(predicted, batch_labels)
        if self.loss_fn == 'CE_loss':
            loss = nn.CrossEntropyLoss()(predicted, batch_labels)
        elif self.loss_fn == 'Focal_loss':
            loss_func = FocalLoss(class_num=self.seg_classes_num,
                                  alpha=self.loss_args['obj_weights'],
                                  gamma=self.loss_args['gamma'])
            loss = loss_func(predicted, batch_labels)
        elif self.loss_fn == 'Dice_loss':
            # loss_func = MultiDiceLoss(weights=self.loss_args['obj_weights'],
            #                           num_class=self.seg_classes_num)
            # loss, _ = loss_func(predicted, batch_labels)
            loss, _ = soft_dice_loss_multi(predicted, batch_labels)
        elif self.loss_fn == 'Combo_loss':
            double_loss_weight = self.loss_args['loss_weights'] if self.loss_args['loss_weights'] else [1 / 2, 1 / 2]
            focal_loss_func = FocalLoss(class_num=self.seg_classes_num,
                                        alpha=self.loss_args['obj_weights'],
                                        gamma=self.loss_args['gamma'])
            focal_loss = focal_loss_func(predicted, batch_labels)

            # dice_loss_func = MultiDiceLoss(weights=self.loss_args['obj_weights'],
            #                                num_class=self.seg_classes_num)
            # dice_loss, _ = dice_loss_func(predicted, batch_labels)
            # dice_loss = dice_loss.cuda()
            dice_loss, _ = soft_dice_loss_multi(predicted, batch_labels)

            double_loss_weight = torch.FloatTensor(double_loss_weight)
            double_loss_weight = double_loss_weight.unsqueeze(1)
            double_loss_weight = double_loss_weight / double_loss_weight.sum()
            double_loss_weight = double_loss_weight.cuda()

            loss = double_loss_weight[0] * focal_loss + double_loss_weight[1] * dice_loss

        elif self.loss_fn == 'Multilabel_loss':
            double_loss_weight = self.loss_args['loss_weights'] if self.loss_args['loss_weights'] else [1 / 2, 1 / 2]
            loss_func = FDLMultiLabelComboLoss(double_loss_weight)
            loss = loss_func(predicted, batch_labels)

        if self.use_fp16:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        if batch_idx % self.grad_accum == self.grad_accum - 1:
            clip_grad_norm_(model.parameters(), self.grad_clip)
            self.optimizer.step()
            self.optimizer.zero_grad()

            # may need fixed
            if self.use_swa:
                self.opt.step()
                self.opt.zero_grad()
                assert hasattr(self.scheduler, 'eta_min')
                if self.optimizer.param_groups[0]['lr'] <= self.scheduler.eta_min:
                    self.opt.update_swa()

        return loss.item(), predicted

    def get_validation_score(self, predicted, labels):
        score_list = []
        predicted = np.argmax(predicted, axis=1)
        for i_label in range(1, self.seg_classes_num):
            score_list.append(self.dice_score_fn(predicted, labels, i_label))
            # print(score_list)

        concat_score = np.array(score_list)
        return np.mean(concat_score), np.mean(concat_score, axis=1)

    def get_validation_score_multi_label(self, predicted, labels):
        score_list = []
        # predicted = np.argmax(predicted, axis=1)
        predicted = (predicted > 0.5).astype(np.uint8)
        for i_label in range(0, self.seg_classes_num):
            score_list.append(self.dice_score_fn(predicted[:, i_label], labels[..., i_label], 1))
            # print(score_list)

        concat_score = np.array(score_list)
        return np.mean(concat_score), np.mean(concat_score, axis=1)

    @staticmethod
    def dice_score_fn(predicted, ground_truth, label):
        eps = 1e-4
        batch_size = predicted.shape[0]

        predicted_bool = np.zeros_like(predicted)
        predicted_bool[predicted == label] = 1
        ground_truth_bool = np.zeros_like(ground_truth)
        ground_truth_bool[ground_truth == label] = 1

        predicted_bool = predicted_bool.reshape(batch_size, -1).astype(np.bool)
        ground_truth_bool = ground_truth_bool.reshape(batch_size, -1).astype(np.bool)

        intersection = np.logical_and(predicted_bool, ground_truth_bool).sum(axis=1)
        union = predicted_bool.sum(axis=1) + ground_truth_bool.sum(axis=1) + eps
        score = (2. * intersection + eps) / union  # use eps make same results
        return score

    def valid_epoch(self, epoch, model, loader):
        tqdm_loader = tqdm(loader)
        current_score_mean = 0
        current_score_each_class = np.zeros((self.seg_classes_num,))
        eval_list = []
        for batch_idx, (case_name, imgs, labels) in enumerate(tqdm_loader):
            with torch.no_grad():
                predicted = self.batch_valid(model, imgs)
                labels = labels.numpy()
                eval_list.append((predicted, labels))
                score, score_each_class = self.get_validation_score_multi_label(predicted, labels)
                current_score_mean = (current_score_mean * batch_idx + score) / (batch_idx + 1)
                current_score_each_class = (current_score_each_class * batch_idx + score_each_class) / (batch_idx + 1)
                tqdm_loader.set_description('score: {:.5}'.format(np.mean(current_score_mean)))
                # print(batch_idx)

        # todo each class
        msg = 'epoch: {}/{}, score: {:.6f}, score_each_class: ' + '{:.4f},' * (self.seg_classes_num)
        msg = msg.format(epoch, self.n_epoches, current_score_mean, *current_score_each_class)
        self.main_logger.info(msg)
        self.fold_logger.info(msg)

        # make visualization with tensorboardX
        if self.tf_logger is not None:
            log_score = {'score': current_score_mean}
            self.tf_logger.add_scalars('logs_score_sesssion_{}', log_score, epoch)

        return eval_list, current_score_mean

    def batch_valid(self, model, batch_imgs):
        batch_imgs = batch_imgs.to(self.device)
        predicted = model(batch_imgs)
        # predicted = torch.softmax(predicted, dim=1)
        predicted = torch.sigmoid(predicted)
        return predicted.cpu().detach().numpy()

    @staticmethod
    def get_state_dict(model):
        if type(model) == torch.nn.DataParallel:
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()
        return state_dict

    def post_processing(self, score, epoch, model):
        if self.freeze_model:
            return

        checkpoints_history_path = Path(
            self.checkpoints_history_folder,
            '{}_epoch_{}.pth'.format(self.calculation_name, epoch)
        )

        torch.save(self.get_state_dict(model), checkpoints_history_path)
        heapq.heappush(self.score_heap, (score, checkpoints_history_path))
        if len(self.score_heap) > self.checkpoints_topk:
            _, removing_checkpoint_path = heapq.heappop(self.score_heap)
            removing_checkpoint_path.unlink()
            msg = 'epoch: {}/{}, removed checkpoint is {}'.format(epoch, self.n_epoches, removing_checkpoint_path)
            self.main_logger.info(msg)
            self.fold_logger.info(msg)
        if score > self.best_score:
            self.best_score = score
            self.best_epoch = epoch
            torch.save(self.get_state_dict(model), self.best_checkpoint_path)
            msg = 'epoch: {}/{}, best model: --> {:.5}'.format(epoch, self.n_epoches, score)
            self.main_logger.info(msg)
            self.fold_logger.info(msg)

        if self.scheduler.__class__.__name__ == 'ReduceLROnPlateau':
            self.scheduler.step(score)
        else:
            self.scheduler.step()

    def run_train(self, model, train_dataloader, valid_dataloader):
        # model.to(self.device)
        loss_hist = collections.deque(maxlen=10)
        for epoch in range(self.n_epoches):
            if not self.freeze_model:
                msg = 'epoch: {}/{}, start training...'.format(epoch, self.n_epoches)
                self.main_logger.info(msg)
                self.fold_logger.info(msg)
                model.train()
                train_loss_mean = self.train_epoch(epoch, model, train_dataloader, loss_hist)
                msg = 'epoch: {}/{}, calculated train loss: {:.5}'.format(epoch, self.n_epoches, train_loss_mean)
                self.main_logger.info(msg)
                self.fold_logger.info(msg)

            if epoch % self.validation_frequency != (self.validation_frequency - 1):
                msg = 'epoch: {}/{}, skip validation....'.format(epoch, self.n_epoches)
                self.main_logger.info(msg)
                self.fold_logger.info(msg)
                continue

            msg = 'epoch: {}/{}, start validation....'.format(epoch, self.n_epoches)
            self.main_logger.info(msg)
            self.fold_logger.info(msg)
            model.eval()
            eval_list, val_score = self.valid_epoch(epoch, model, valid_dataloader)

            selected_score = val_score
            self.post_processing(selected_score, epoch, model)

            if epoch - self.best_epoch > self.early_stopping:
                msg = 'epoch: {}/{}, EARLY STOPPING'.format(epoch, self.n_epoches)
                self.main_logger.info(msg)
                self.fold_logger.info(msg)
                break

        if self.use_swa:
            self.opt.swap_swa_sgd()

        return self.best_score, self.best_epoch
