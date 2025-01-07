import os
import numpy as np
import torch
import torch.optim as optim
import random
from tqdm import tqdm

# Local imports (adjust paths as needed)
from evaluation.post_process import calculate_hr
from evaluation.metrics import calculate_metrics
from neural_methods.model.RGBMamba import RGBMamba
from neural_methods.trainer.BaseTrainer import BaseTrainer
from neural_methods.loss.TorchLossComputer import Hybrid_Loss


class RGBMambaTrainer(BaseTrainer):

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

        # Track the best validation loss
        self.min_valid_loss = None
        self.best_epoch = 0

        # Whether the target signals are difference-normalized
        self.diff_flag = 1 if config.TRAIN.DATA.PREPROCESS.LABEL_TYPE == "DiffNormalized" else 0

        # Depending on toolbox mode, initialize model and training specifics
        if config.TOOLBOX_MODE == "train_and_test":
            self.model = RGBMamba().to(self.device)
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(self.num_of_gpu)))

            # Number of training batches for the OneCycleLR scheduler
            self.num_train_batches = len(data_loader["train"])

            self.criterion = Hybrid_Loss()
            self.optimizer = optim.AdamW(
                self.model.parameters(), lr=config.TRAIN.LR, weight_decay=0
            )
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=config.TRAIN.LR,
                epochs=config.TRAIN.EPOCHS,
                steps_per_epoch=self.num_train_batches
            )

        elif config.TOOLBOX_MODE == "only_test":
            self.model = RGBMamba().to(self.device)
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(self.num_of_gpu)))
        else:
            raise ValueError("RGBMambaTrainer initialized in incorrect toolbox mode!")

    def data_augmentation(self, data, labels):
        """
        Example data augmentation, consistent with your prior approach:
        - Possibly up/downsample frames
        - Possibly spatial flip
        - This is purely optional or can be replaced with any augmentation strategy.
        """
        N, D, C, H, W = data.shape
        data_aug = np.zeros((N, D, C, H, W))
        labels_aug = np.zeros((N, D))

        for idx in range(N):
            # Calculate approximate HR just to decide augmentation
            gt_hr_fft, _ = calculate_hr(
                labels[idx], labels[idx],
                diff_flag=self.diff_flag,
                fs=self.config.VALID.DATA.FS
            )
            rand1 = random.random()
            rand2 = random.random()
            rand3 = random.randint(0, D // 2 - 1)

            # Example logic
            if rand1 < 0.5:
                if gt_hr_fft > 90:
                    # Doubling frames
                    for tt in range(rand3, rand3 + D):
                        if tt % 2 == 0:
                            data_aug[idx, tt - rand3, :, :, :] = data[idx, tt // 2, :, :, :]
                            labels_aug[idx, tt - rand3] = labels[idx, tt // 2]
                        else:
                            data_aug[idx, tt - rand3, :, :, :] = (
                                data[idx, tt // 2, :, :, :] / 2 +
                                data[idx, (tt // 2) + 1, :, :, :] / 2
                            )
                            labels_aug[idx, tt - rand3] = (
                                labels[idx, tt // 2] / 2 +
                                labels[idx, (tt // 2) + 1] / 2
                            )
                elif gt_hr_fft < 75:
                    # Halving frames, then repeating
                    for tt in range(D):
                        if tt < D / 2:
                            data_aug[idx, tt, :, :, :] = data[idx, tt * 2, :, :, :]
                            labels_aug[idx, tt] = labels[idx, tt * 2]
                        else:
                            data_aug[idx, tt, :, :, :] = data_aug[idx, tt - D // 2, :, :, :]
                            labels_aug[idx, tt] = labels_aug[idx, tt - D // 2]
                else:
                    # No augmentation
                    data_aug[idx] = data[idx]
                    labels_aug[idx] = labels[idx]
            else:
                data_aug[idx] = data[idx]
                labels_aug[idx] = labels[idx]

            # Random horizontal flip
            if rand2 < 0.5:
                data_aug[idx] = np.flip(data_aug[idx], axis=3)  # axis=3 => flip width

        # Convert to tensors
        data_aug = torch.tensor(data_aug, dtype=torch.float32)
        labels_aug = torch.tensor(labels_aug, dtype=torch.float32)
        return data_aug, labels_aug

    def train(self, data_loader):
        """Main training loop."""
        if data_loader["train"] is None:
            raise ValueError("No data for 'train'.")

        for epoch in range(self.max_epoch_num):
            print(f"\n====Training Epoch: {epoch}====")
            self.model.train()

            tbar = tqdm(data_loader["train"], ncols=80)
            tbar.set_description(f"Train epoch {epoch}")
            for idx, batch in enumerate(tbar):
                data, labels = batch[0].float(), batch[1].float()

                # Optional data augmentation
                if self.config.TRAIN.AUG:
                    data, labels = self.data_augmentation(data, labels)

                data = data.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()

                # Forward pass
                pred_ppg = self.model(data)  # => [B, T]

                # Simple normalization (already done in model, but you could do more if you want)
                # pred_ppg = pred_ppg - torch.mean(pred_ppg, axis=-1, keepdim=True)
                # pred_ppg = pred_ppg / (torch.std(pred_ppg, axis=-1, keepdim=True) + 1e-5)

                # Compute loss
                # We assume each batch item has shape [T], and labels have shape [B, T].
                loss = 0.0
                for ib in range(data.shape[0]):
                    loss += self.criterion(
                        pred_ppg[ib],
                        labels[ib],
                        epoch,
                        self.config.TRAIN.DATA.FS,
                        self.diff_flag
                    )
                loss = loss / data.shape[0]

                # Backprop
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                # TQDM progress
                tbar.set_postfix(loss=loss.item())

            # Save checkpoint after each epoch
            self.save_model(epoch)

            # If we are not using "last epoch only", do validation
            if not self.config.TEST.USE_LAST_EPOCH:
                valid_loss = self.valid(data_loader)
                print("validation loss:", valid_loss)

                # Update best model if improved
                if self.min_valid_loss is None:
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    print(f"Update best model! Best epoch: {self.best_epoch}")
                elif valid_loss < self.min_valid_loss:
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    print(f"Update best model! Best epoch: {self.best_epoch}")

        if not self.config.TEST.USE_LAST_EPOCH:
            print(f"best trained epoch: {self.best_epoch}, min_val_loss: {self.min_valid_loss}")

    def valid(self, data_loader):
        if data_loader["valid"] is None:
            raise ValueError("No data for 'valid'.")

        print("\n===Validating===")
        valid_loss = []
        self.model.eval()

        with torch.no_grad():
            vbar = tqdm(data_loader["valid"], ncols=80)
            vbar.set_description("Validation")
            for valid_idx, valid_batch in enumerate(vbar):
                data_valid, labels_valid = valid_batch[0].float(), valid_batch[1].float()
                data_valid = data_valid.to(self.device)
                labels_valid = labels_valid.to(self.device)

                pred_ppg_valid = self.model(data_valid)

                N = data_valid.shape[0]
                for ib in range(N):
                    loss = self.criterion(
                        pred_ppg_valid[ib],
                        labels_valid[ib],
                        self.config.TRAIN.EPOCHS,
                        self.config.VALID.DATA.FS,
                        self.diff_flag
                    )
                    valid_loss.append(loss.item())
                    vbar.set_postfix(loss=loss.item())

        return np.mean(np.array(valid_loss))

    def test(self, data_loader):
        if data_loader["test"] is None:
            raise ValueError("No data for 'test'.")

        print("\n===Testing===")
        # Load checkpoint if needed
        if self.config.TOOLBOX_MODE == "only_test":
            if not os.path.exists(self.config.INFERENCE.MODEL_PATH):
                raise ValueError("Inference model path error! Check INFERENCE.MODEL_PATH in your yaml.")
            self.model.load_state_dict(torch.load(self.config.INFERENCE.MODEL_PATH))
            print("Testing uses a pretrained model from:", self.config.INFERENCE.MODEL_PATH)
        else:
            if self.config.TEST.USE_LAST_EPOCH:
                last_epoch_model_path = os.path.join(
                    self.model_dir,
                    self.model_file_name + '_Epoch' + str(self.max_epoch_num - 1) + '.pth'
                )
                print("Testing uses the last epoch's checkpoint:", last_epoch_model_path)
                self.model.load_state_dict(torch.load(last_epoch_model_path))
            else:
                best_model_path = os.path.join(
                    self.model_dir,
                    self.model_file_name + '_Epoch' + str(self.best_epoch) + '.pth'
                )
                print("Testing uses the best epoch checkpoint:", best_model_path)
                self.model.load_state_dict(torch.load(best_model_path))

        self.model.to(self.device)
        self.model.eval()

        with torch.no_grad():
            predictions = dict()
            labels = dict()
            tbar = tqdm(data_loader["test"], ncols=80)
            tbar.set_description("Test")
            for _, test_batch in enumerate(tbar):
                data_test, labels_test = test_batch[0], test_batch[1]
                batch_size = data_test.shape[0]

                data_test = data_test.to(self.device)
                labels_test = labels_test.to(self.device)

                pred_ppg_test = self.model(data_test)   # => [B, T]

                # Flatten for easier storage
                chunk_len = self.chunk_len
                labels_test = labels_test.view(-1, 1)
                pred_ppg_test = pred_ppg_test.view(-1, 1)

                for ib in range(batch_size):
                    subj_index = test_batch[2][ib]  # subject identifier
                    sort_index = int(test_batch[3][ib])  # chunk ordering index

                    if subj_index not in predictions:
                        predictions[subj_index] = {}
                        labels[subj_index] = {}

                    predictions[subj_index][sort_index] = pred_ppg_test[
                        ib * chunk_len : (ib + 1) * chunk_len
                    ]
                    labels[subj_index][sort_index] = labels_test[
                        ib * chunk_len : (ib + 1) * chunk_len
                    ]

            # Evaluate final metrics
            calculate_metrics(predictions, labels, self.config)

    def save_model(self, index):
        """Save the model checkpoint at the specified epoch."""
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        model_path = os.path.join(
            self.model_dir, f"{self.model_file_name}_Epoch{index}.pth"
        )
        torch.save(self.model.state_dict(), model_path)
        print("Saved Model Path:", model_path)
