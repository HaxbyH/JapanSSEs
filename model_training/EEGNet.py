import torch
from torch import nn
from torch.nn import functional as F
import copy
import glob

def binary_acc(y_pred, y_test):
    # print(f"Y pred: {y_pred}")
    # print(f"y_test: {y_test}")
    prediction = torch.round(y_pred)
    # print(f"Prediction: {prediction}")
    correct_pred = (prediction == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    acc = torch.round(acc * 100)
    return acc

'''
https://github.com/s4rduk4r/eegnet_pytorch
    EEGNet PyTorch implementation
    Original implementation - https://github.com/vlawhern/arl-eegmodels
    Original paper: https://iopscience.iop.org/article/10.1088/1741-2552/aace8c

    ---
    EEGNet Parameters:

      nb_classes      : int, number of classes to classify
      Chans           : number of channels in the EEG data
      Samples         : sample frequency (Hz) in the EEG data
      dropoutRate     : dropout fraction
      kernLength      : length of temporal convolution in first layer. 
                        ARL recommends to set this parameter to be half of the sampling rate. 
                        For the SMR dataset in particular since the data was high-passed at 4Hz ARL used a kernel length of 32.
      F1, F2          : number of temporal filters (F1) and number of pointwise
                        filters (F2) to learn. Default: F1 = 8, F2 = F1 * D.
      D               : number of spatial filters to learn within each temporal
                        convolution. Default: D = 2
'''

class EEGNet(nn.Module):
    def __init__(self, nb_classes: int, Chans: int = 64, Samples: int = 128,
                 dropoutRate: float = 0.5, kernLength: int = 63,
                 F1:int = 8, D:int = 2):
        super().__init__()

        F2 = F1 * D

        # Make kernel size and odd number
        try:
            assert kernLength % 2 != 0
        except AssertionError:
            raise ValueError("ERROR: kernLength must be odd number")

        # In: (B, Chans, Samples, 1)
        # Out: (B, F1, Samples, 1)
        self.conv1 = nn.Conv1d(Chans, F1, kernLength, padding=(kernLength // 2))
        self.bn1 = nn.BatchNorm1d(F1) # (B, F1, Samples, 1)
        # In: (B, F1, Samples, 1)
        # Out: (B, F2, Samples - Chans + 1, 1)
        self.conv2 = nn.Conv1d(F1, F2, Chans, groups=F1)
        self.bn2 = nn.BatchNorm1d(F2) # (B, F2, Samples - Chans + 1, 1)
        # In: (B, F2, Samples - Chans + 1, 1)
        # Out: (B, F2, (Samples - Chans + 1) / 4, 1)
        self.avg_pool = nn.AvgPool1d(1)
        self.dropout = nn.Dropout(dropoutRate)

        # In: (B, F2, (Samples - Chans + 1) / 4, 1)
        # Out: (B, F2, (Samples - Chans + 1) / 4, 1)
        
        
        # self.conv3 = SeparableConv1d(F2, F2, kernel_size=3, padding=2)
        self.depthwise_conv = nn.Conv1d(F2, F2, kernel_size=3, padding=2, groups=F2)
        self.conv1d_1x1 = nn.Conv1d(F2, F2, kernel_size=1)



        self.bn3 = nn.BatchNorm1d(F2)
        # In: (B, F2, (Samples - Chans + 1) / 4, 1)
        # Out: (B, F2, (Samples - Chans + 1) / 32, 1)
        self.avg_pool2 = nn.AvgPool1d(1)
        # In: (B, F2 *  (Samples - Chans + 1) / 32)
        # 32x2960 (incomming) -> Outgoing = 1
        # self.fc = nn.Linear(F2 * ((Samples - Chans + 1) // 32), nb_classes)
        self.flatten = torch.flatten
        self.fc = nn.Linear(496, nb_classes)
        # self.fc = nn.Linear(2992, nb_classes)


    def forward(self, x: torch.Tensor):
        # Block 1
        # print("BLOCK 1")
        # print("-------------------------")
        # print()
        # print("INPUT: ", x.shape)
        y1 = self.conv1(x)
        # print("AFTER CONV1: ", y1.shape)

        y1 = self.bn1(y1)
        # print("AFTER BATCH1: ", y1.shape)
        y1 = self.conv2(y1)
        # print("AFTER CONV2: ", y1.shape)
        y1 = F.relu(self.bn2(y1))
        # print("AFTER RELU ACTIVATION: ", y1.shape)
        y1 = self.avg_pool(y1)
        # print("AFTER AVEPOOL", y1.shape)
        y1 = self.dropout(y1)
        # print("AFTER DROPOUT: ", y1.shape)
        # print()
        # print("BLOCK 2")
        # print("-------------------------")
        # print()

        # Block 2
        # y2 = self.conv3(y1)
        y2 = self.depthwise_conv(y1)
        y2 = self.conv1d_1x1(y2)

        # print("AFTER CONV3: ", y2.shape)
        y2 = F.relu(self.bn3(y2))
        # print("AFTER RELU ACTIVATION: ", y2.shape)
        y2 = self.avg_pool2(y2)
        # print("AFTER AVE2: ", y2.shape)
        y2 = self.dropout(y2)
        # print("dropout", y2.shape)
        y2 = self.flatten(y2, 1)
        # print("AFTER FLATTEN: ", y2.shape)
        y2 = torch.sigmoid(self.fc(y2))
        # print("fc", y2.shape)

        return y2

def train(model, optimizer, criterion, dataloader,val_dataloader, epochs, device):
    start_epoch = 0

    best_model_wts = copy.deepcopy(model.state_dict())
    lowest_loss = 100

    accuracy_stats = {
        'train': [],
        "val": []
    }
    loss_stats = {
        'train': [],
        "val": []
    }

    check_path = glob.glob("./checkpoints/*.tar")
    if check_path:
        checkpoint = torch.load(check_path[0])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        loss_stats = checkpoint['loss_stats']
        accuracy_stats = checkpoint['accuracy_stats']
        best_model_wts = checkpoint["best_wts"]
        lowest_loss = checkpoint["lowest_loss"]
        model.train()
        print(f"Found checkpoint. Epoch: {start_epoch-1} | Train acc: {accuracy_stats['train'][-1]} | Val acc: {accuracy_stats['val'][-1]}")


    for epoch in range(start_epoch, epochs):
        # Training
        train_loss = 0
        train_acc = 0
        data_size = 0
        iteration = 0
        for features, labels in dataloader:
            if iteration % 100 == 0:
                print(f"Iteration: {iteration} / {len(dataloader)}")
            iteration += 1
            features, labels = features.to(device), labels.to(device)
            features = features.float()
            labels = labels.float()

            optimizer.zero_grad()
            # print(features.shape)
            # print(features)
            # print(model)
            # y_pred = torch.squeeze(y_pred)
            y_pred = model(features)
            y_pred = y_pred.squeeze(1)
            # print(y_pred)
            # print(labels)
            loss = criterion(y_pred, labels)
            acc = binary_acc(y_pred, labels)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_acc += acc.item()

            # data_size += 1
            # # if data_size % 100 == 0:
            # print(f"{data_size} / {len(dataloader)}")
        
        # Validating
        with torch.no_grad():
            val_loss = 0
            val_acc = 0
            for val_features, val_labels in val_dataloader:
                val_features, val_labels = val_features.to(device), val_labels.to(device)
                # val_features = torch.unsqueeze(val_features, 1)
                val_features = val_features.float()
                val_labels = val_labels.float()

                # print(val_labels.shape)
                val_pred = model(val_features)
                val_pred = torch.squeeze(val_pred)
                val_loss_item = criterion(val_pred, val_labels)
                val_acc_item = binary_acc(val_pred, val_labels)
                

                val_loss += val_loss_item.item()
                val_acc += val_acc_item.item()
                # break;
                if val_loss < lowest_loss:
                    lowest_loss = val_loss  
                    best_model_wts = copy.deepcopy(model.state_dict())


        loss_stats['train'].append(train_loss/len(dataloader))
        loss_stats['val'].append(val_loss/len(val_dataloader))
        accuracy_stats['train'].append(train_acc/len(dataloader))
        accuracy_stats['val'].append(val_acc/len(val_dataloader))

        
        print(f'Epoch {epoch+0:03}: | Train Loss: {train_loss/len(dataloader):.5f} | Val Loss: {val_loss/len(val_dataloader):.5f} | Train Acc: {train_acc/len(dataloader):.3f} | Val Acc: {val_acc/len(val_dataloader):.3f}')
        # path = f"./checkpoints/check_e.tar"
        # torch.save({
        #     "epoch": (epoch+1),
        #     "model_state_dict": model.state_dict(),
        #     "optimizer_state_dict": optimizer.state_dict(),
        #     "loss_stats": loss_stats,
        #     "accuracy_stats": accuracy_stats,
        #     "best_wts": best_model_wts,
        #     "lowest_loss": lowest_loss,
        # }, path)
    return best_model_wts, accuracy_stats, loss_stats
