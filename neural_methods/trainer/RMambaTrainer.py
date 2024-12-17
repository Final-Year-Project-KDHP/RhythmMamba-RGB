"""Trainer for RMamba."""
import os
import numpy as np
import torch
import torch.optim as optim
import random
from tqdm import tqdm
from evaluation.post_process import calculate_hr
from evaluation.metrics import calculate_metrics
from neural_methods.model.RMamba import RMamba
from neural_methods.trainer.BaseTrainer import BaseTrainer
from neural_methods.loss.TorchLossComputer import Hybrid_Loss

class RMambaTrainer(BaseTrainer):

    def __init__(self, config, data_loader):
        super().__init__()
        self.device = torch.device(config.DEVICE)
        self.max_epoch_num = config.TRAIN.EPOCHS
        self.model_dir = config.MODEL.MODEL_DIR
        self.model_file_name = config.TRAIN.MODEL_FILE_NAME
        self.batch_size = config.TRAIN.BATCH_SIZE
        self.num_of_gpu = config.NUM_OF_GPU_TRAIN
        self.chunk_len = config.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH
        self.config = config
        self.min_valid_loss = None
        self.best_epoch = 0
        self.diff_flag = 0
        if config.TRAIN.DATA.PREPROCESS.LABEL_TYPE == "DiffNormalized":
            self.diff_flag = 1
        if config.TOOLBOX_MODE == "train_and_test":
            self.model = RMamba().to(self.device)
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(config.NUM_OF_GPU_TRAIN)))
            self.num_train_batches = len(data_loader["train"])
            self.criterion = Hybrid_Loss()
            self.optimizer = optim.AdamW(
                self.model.parameters(), lr=config.TRAIN.LR, weight_decay=0)
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=config.TRAIN.LR, epochs=config.TRAIN.EPOCHS, steps_per_epoch=self.num_train_batches)
        elif config.TOOLBOX_MODE == "only_test":
            self.model = RMamba().to(self.device)
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(config.NUM_OF_GPU_TRAIN)))
        else:
            raise ValueError("RMamba trainer initialized in incorrect toolbox mode!")
    
    def save_model(self, index):
        """
        Save the model checkpoint at the specified epoch.
        Args:
            index: Epoch index for naming the checkpoint file.
        """
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        model_path = os.path.join(
            self.model_dir, self.model_file_name + '_Epoch' + str(index) + '.pth')
        torch.save(self.model.state_dict(), model_path)
        print('Saved Model Path: ', model_path)

    def preprocess_red_channel(self, data):
        """
        Extract the Red (R) channel from the input data.
        Args:
            data: Input tensor [N, D, 3, H, W] (RGB format).
        Returns:
            data_r: Tensor containing only the R channel [N, D, 1, H, W].
        """
        data_r = data[:, :, 0:1, :, :]  # Extract R channel (index 0)
        return data_r

    def data_augmentation(self, data, labels):
        """
        Perform data augmentation by applying temporal transformations.
        Args:
            data: Input data tensor [N, D, C, H, W].
            labels: Ground truth labels [N, D].
        Returns:
            Augmented data and labels.
        """
        N, D, C, H, W = data.shape
        data_aug = np.zeros((N, D, C, H, W))
        labels_aug = np.zeros((N, D))
        for idx in range(N):
            gt_hr_fft, _ = calculate_hr(labels[idx], labels[idx], diff_flag=self.diff_flag, fs=self.config.VALID.DATA.FS)
            rand1 = random.random()
            rand2 = random.random()
            rand3 = random.randint(0, D // 2 - 1)

            if rand1 < 0.5:
                if gt_hr_fft > 90:
                    for tt in range(rand3, rand3 + D):
                        if tt % 2 == 0:
                            data_aug[idx, tt - rand3, :, :, :] = data[idx, tt // 2, :, :, :]
                            labels_aug[idx, tt - rand3] = labels[idx, tt // 2]
                        else:
                            data_aug[idx, tt - rand3, :, :, :] = (
                                data[idx, tt // 2, :, :, :] / 2 + data[idx, tt // 2 + 1, :, :, :] / 2
                            )
                            labels_aug[idx, tt - rand3] = (
                                labels[idx, tt // 2] / 2 + labels[idx, tt // 2 + 1] / 2
                            )
                elif gt_hr_fft < 75:
                    for tt in range(D):
                        if tt < D / 2:
                            data_aug[idx, tt, :, :, :] = data[idx, tt * 2, :, :, :]
                            labels_aug[idx, tt] = labels[idx, tt * 2]
                        else:
                            data_aug[idx, tt, :, :, :] = data_aug[idx, tt - D // 2, :, :, :]
                            labels_aug[idx, tt] = labels_aug[idx, tt - D // 2]
                else:
                    data_aug[idx] = data[idx]
                    labels_aug[idx] = labels[idx]
            else:
                data_aug[idx] = data[idx]
                labels_aug[idx] = labels[idx]

        data_aug = torch.tensor(data_aug).float()
        labels_aug = torch.tensor(labels_aug).float()
        if rand2 < 0.5:
            data_aug = torch.flip(data_aug, dims=[4])
        return data_aug, labels_aug

    def train(self, data_loader):
        """Training routine for model."""
        if data_loader["train"] is None:
            raise ValueError("No data for train")

        for epoch in range(self.max_epoch_num):
            print('')
            print(f"====Training Epoch: {epoch}====")
            self.model.train()

            tbar = tqdm(data_loader["train"], ncols=80)
            for idx, batch in enumerate(tbar):
                tbar.set_description("Train epoch %s" % epoch)
                data, labels = batch[0].float(), batch[1].float()

                # Preprocess to extract R channel
                data = self.preprocess_red_channel(data)

                if self.config.TRAIN.AUG:
                    data, labels = self.data_augmentation(data, labels)

                data = data.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                pred_ppg = self.model(data)
                pred_ppg = (pred_ppg - torch.mean(pred_ppg, axis=-1).view(-1, 1)) / torch.std(pred_ppg, axis=-1).view(-1, 1)

                loss = 0.0
                for ib in range(data.shape[0]):
                    loss += self.criterion(pred_ppg[ib], labels[ib], epoch, self.config.TRAIN.DATA.FS, self.diff_flag)
                loss = loss / data.shape[0]
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                tbar.set_postfix(loss=loss.item())

            self.save_model(epoch)
