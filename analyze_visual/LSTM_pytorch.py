import warnings
import argparse
import os
import numpy as np
import sys
import torch
import random
import shutil
import torch.nn as nn
from pathlib import Path
from collections import OrderedDict
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from sklearn.metrics import f1_score
from torch.nn.utils.rnn import pad_sequence as pad
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


"""
RUN:
apo seagate:
python3 LSTM_pytorch.py -v /media/ubuntu/Seagate/test/t/Non_Static_5 /media/ubuntu/Seagate/test/t/Static_5

big dataset:
python3 LSTM_pytorch.py -v /home/ubuntu/LSTM/binary_data/data/Non_Static_4 /home/ubuntu/LSTM/binary_data/data/Static_4
"""


def parse_arguments():
    """Parse arguments for real time demo.
    """
    parser = argparse.ArgumentParser(description="Create Shot "
                                                 "Classification Dataset")
    parser.add_argument("-v", "--videos_path", required=True, action='append',
                        nargs='+', help="Videos folder path")

    return parser.parse_args()


def seed_all(seed):
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def create_dataset(videos_path):
    """
    It returns the list "videos_dataset" which contains tuples.
    In each tuple the first element is the video's full_path_name
    and the second one is the corresponding y label
    """

    videos_dataset = []
    label_int = -1
    print("\n")
    for folder in videos_path:
        label = os.path.basename(folder)
        label_int += 1
        print(label, "=", label_int)

        for filename in os.listdir(folder):
            if filename.endswith(".mp4.npy"):
                full_path_name = folder + "/" + filename
                videos_dataset.append(tuple((full_path_name, label_int)))
    print("\n")

    return videos_dataset


def my_collate(batch):
    """
    Different padding in each batch depending on
    the longest sequence in the batch
    batch: list of tuples (data, label)
    """

    # batch's elements in descending order
    sorted_batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
    sequences = [x[0] for x in sorted_batch]

    sequences_padded = pad(sequences, batch_first=True)

    # store the original length of each sequence
    # before padding (to use it for packing and unpacking)
    lengths = torch.Tensor([len(x) for x in sequences])

    # labels of the sorted batch
    sorted_labels = [item[1] for item in sorted_batch]
    labels = torch.Tensor(sorted_labels)

    return sequences_padded, labels, lengths


def load_data(X, y, check_train, scaler):
    # for train/val/test sets
    split_dataset = []
    for i, j in zip(X, y):
        split_dataset.append(tuple((i, j)))

    x_len = []
    X = []
    labels = []
    for index, data in enumerate(split_dataset):
        # data[0] corresponds to .npy names
        # data[1] corresponds to y labels
        X_to_tensor = np.load(data[0])

        # keep only specific features
        #X_to_tensor = X_to_tensor[:, 45:89]

        y = data[1]
        labels.append(y)
        x_len.append(X_to_tensor.shape[0])

        # data normalization
        if check_train == True:
            X_to_tensor = scaler.fit_transform(X_to_tensor)
        else:
            X_to_tensor = scaler.transform(X_to_tensor)

        X.append(torch.Tensor(X_to_tensor))

    return X, labels, x_len


class LSTMDataset(Dataset):
    def __init__(self, X, y, lengths):
        self.X = X
        self.y = y
        self.lengths = lengths
        #self.max_length = max(lengths)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.X[index], self.y[index], self.lengths[index]


def data_preparation(videos_dataset, batch_size):

    # TODO: train/test split with StratifiedKFold ?

    X = [x[0] for x in videos_dataset]
    y = [x[1] for x in videos_dataset]

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                      test_size=0.15, stratify=y_train)

    # Define Scaler
    min_max_scaler = MinMaxScaler()

    X_train, y_train, train_lengths = load_data(X_train, y_train, True, scaler=min_max_scaler)
    train_dataset = LSTMDataset(X_train, y_train, train_lengths)

    X_val, y_val, val_lengths = load_data(X_val, y_val, False, scaler=min_max_scaler)
    val_dataset = LSTMDataset(X_val, y_val, val_lengths)

    X_test, y_test, test_lengths = load_data(X_test, y_test, False, scaler=min_max_scaler)
    test_dataset = LSTMDataset(X_test, y_test, test_lengths)

    # Define a DataLoader for each set
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              collate_fn=my_collate, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                              collate_fn=my_collate, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                              collate_fn=my_collate, shuffle=True)

    return train_loader, val_loader, test_loader


def F_score(logit, label, threshold=0.5, beta=2):

    #prob = torch.sigmoid(logit)
    #prob = prob > threshold

    prob = logit > threshold
    label = label > threshold

    TP = (prob & label).sum().float()
    TN = ((~prob) & (~label)).sum().float()
    FP = (prob & (~label)).sum().float()
    FN = ((~prob) & label).sum().float()

    accuracy = (TP+TN)/(TP+TN+FP+FN)
    precision = torch.mean(TP / (TP + FP + 1e-12))
    recall = torch.mean(TP / (TP + FN + 1e-12))
    #F2 = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-12)
    F2 = (2 * precision * recall) / (precision + recall + 1e-12)
    cm = np.array([[TP, FP], [FN, TN]])
    #print(cm)

    return accuracy, precision, recall, F2.mean(0), cm


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,
                 output_size, dropout_prob):
        super(LSTMModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout_prob)

        self.drop = nn.Dropout(p=dropout_prob)
        self.fc = nn.Linear(hidden_size, output_size)
        self.m = nn.Sigmoid()
        # self.fnn = nn.Sequential(OrderedDict([
        #     ('gelu1', nn.LeakyReLU()),
        #     ('fc1', nn.Linear(self.hidden_size, 512)),
        #     ('bn1', nn.BatchNorm1d(512)),
        #     ('gelu2', nn.LeakyReLU()),
        #     ('fc2', nn.Linear(512, 512)),
        #     ('bn2', nn.BatchNorm1d(512)),
        #     ('gelu3', nn.LeakyReLU()),
        #     ('fc3', nn.Linear(512, 128)),
        #     ('bn3', nn.BatchNorm1d(128)),
        #     ('gelu4', nn.LeakyReLU()),
        #     ('fc4', nn.Linear(128, 64)),
        #     ('bn4', nn.BatchNorm1d(64)),
        #     ('gelu5', nn.LeakyReLU()),
        #     ('fc5', nn.Linear(64, 32)),
        #     ('bn5', nn.BatchNorm1d(32)),
        #     ('gelu6', nn.LeakyReLU()),
        #     ('fc6', nn.Linear(32, 1))
        # ]))

        # self.fnn = nn.Sequential(OrderedDict([
        #     ('gelu1', nn.ReLU()),
        #     ('fc1', nn.Linear(self.hidden_size, 128)),
        #     ('bn1', nn.BatchNorm1d(128)),
        #     ('gelu2', nn.ReLU()),
        #     ('fc2', nn.Linear(128, 128)),
        #     ('bn2', nn.BatchNorm1d(128)),
        #     ('gelu3', nn.ReLU()),
        #     ('fc3', nn.Linear(128, 64)),
        #     ('bn3', nn.BatchNorm1d(64)),
        #     ('gelu4', nn.ReLU()),
        #     ('fc4', nn.Linear(64, 32)),
        #     ('bn4', nn.BatchNorm1d(32)),
        #     ('gelu5', nn.ReLU()),
        #     ('fc5', nn.Linear(32, 1))
        # ]))

        # self.fnn = nn.Sequential(OrderedDict([
        #     ('relu1', nn.GELU()),
        #     ('fc1', nn.Linear(self.hidden_size, 64)),
        #     ('bn1', nn.BatchNorm1d(64)),
        #     ('relu2', nn.GELU()),
        #     ('fc2', nn.Linear(64, 64)),
        #     ('bn2', nn.BatchNorm1d(64)),
        #     ('relu3', nn.GELU()),
        #     ('fc3', nn.Linear(64, 32)),
        #     ('bn3', nn.BatchNorm1d(32)),
        #     ('relu4', nn.GELU()),
        #     ('fc4', nn.Linear(32, 1))
        # ]))

        self.fnn = nn.Sequential(OrderedDict([
            ('relu1', nn.LeakyReLU()),
            ('fc1', nn.Linear(self.hidden_size, 32)),
            ('bn1', nn.BatchNorm1d(32)),
            ('relu2', nn.LeakyReLU()),
            ('fc2', nn.Linear(32, 32)),
            ('bn2', nn.BatchNorm1d(32)),
            ('relu3', nn.LeakyReLU()),
            ('fc3', nn.Linear(32, 1))
        ]))

    def forward(self, X, lengths):
        """
        The forward() function is executed sequentially, passing the
        inputs and both zero-initialized hidden and cell state.
        The function backward() is created automatically
        and it computes the backpropagation.
        """

        # Initialize (with zeros) both hidden and cell state
        # for the first input
        h0 = torch.zeros(self.num_layers, X.size(0),
                         self.hidden_size).requires_grad_()

        # The cell state determines which information is relevant or not
        # through input and forget gates respectively
        c0 = torch.zeros(self.num_layers, X.size(0),
                         self.hidden_size).requires_grad_()

        # Forward propagate LSTM
        output, (hn, cn) = self.lstm(X, (h0.detach(), c0.detach()))
        #output, _ = self.lstm(X)

        last_states = self.last_by_index(output, lengths)

        # output shape: (batch_size, seq_length, hidden_size)

        #output = self.fc(last_states)
        #last_states = self.drop(last_states)
        output = self.fnn(last_states)
        output = self.m(output)
        #output = self.fnn(output)

        return output

    @staticmethod
    def last_by_index(outputs, lengths):
        # Index of the last output for each sequence.
        idx = (lengths - 1).view(-1, 1).expand(outputs.size(0),
                                               outputs.size(2)).unsqueeze(1)

        return outputs.gather(1, idx.type(torch.int64)).squeeze()


def save_ckp(checkpoint, is_best_val, checkpoint_path, best_model_path):
    """
    state: checkpoint we want to save
    is_best: is this the best checkpoint; min validation loss
    """

    # save checkpoint data
    torch.save(checkpoint, checkpoint_path)

    # if it is the best model
    if is_best_val:
        best_check_path = best_model_path

        # copy that checkpoint file to best path
        shutil.copyfile(checkpoint_path, best_check_path)


def load_ckp(checkpoint_path, model, optimizer):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into
    optimizer: optimizer we defined in previous training
    """

    checkpoint = torch.load(checkpoint_path)

    # initialize state_dict, optimizer, validation_min_loss (from checkpoint)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    validation_min_loss = checkpoint['validation_min_loss']

    return model, optimizer, checkpoint['epoch'], validation_min_loss.item()


class Optimization:
    def __init__(self, model, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_loss = []
        self.val_loss = []

    def train(self, train_loader, val_loader, n_epochs):

        #validation_min_loss = float('inf')
        f1_max = -1.0
        counter_epoch = 0

        for epoch in range(1, n_epochs + 1):
            counter_epoch += 1

            train_losses = []
            val_losses = []

            acc_list = []
            f1_score_list = []
            #batch_predictions = []
            #batch_values = []

            # enumerate mini batches
            for batch_idx, batch_info in enumerate(train_loader):
                # batch_idx -------> batch id
                # batch_info[0] ---> padded arrays in each batch
                # batch_info[1] ---> labels (y) in each batch
                # batch_info[2] ---> original length of each sequence

                self.optimizer.zero_grad()

                X_train = batch_info[0]
                y_train = batch_info[1]
                X_train_original_len = batch_info[2]

                #X_train_packed = pack(X_train, X_train_original_len, batch_first=True)

                # print(X_train_packed[0].shape)
                # print(X_train_packed.data)
                # print(X_train_packed.batch_sizes)
                with torch.set_grad_enabled(True):
                    self.model.train()

                    # Compute the model output
                    out = self.model(X_train.float(), X_train_original_len)

                    # Calculate loss
                    output = out.squeeze().float()

                    #print("output: ", output)
                    loss = self.loss_fn(output, y_train.float())

                    # Computes the gradients
                    loss.backward()

                    # Updates parameters and zero gradients
                    self.optimizer.step()
                    train_losses.append(loss.item())

            train_step_loss = np.mean(train_losses)
            self.train_loss.append(train_step_loss)

            # validation process
            with torch.no_grad():

                for val_batch_idx, val_batch_info in enumerate(val_loader):
                    X_val = val_batch_info[0]
                    y_val = val_batch_info[1]
                    X_val_original_len = val_batch_info[2]

                    #X_val_packed = pack(X_val, X_val_original_len, batch_first=True)

                    self.model.eval()
                    y_hat = self.model(X_val.float(), X_val_original_len)
                    y_hat = y_hat.squeeze().float()

                    val_loss = self.loss_fn(y_hat, y_val.float())
                    val_losses.append(val_loss)

                    #batch_predictions.append(y_hat)
                    #batch_values.append(y_val)

                    accuracy, precision, recall, F1_score, cm = F_score(y_hat, y_val.float())

                    acc_list.append(accuracy)
                    f1_score_list.append(F1_score)

                #batch_values = np.concatenate(batch_values).ravel()
                #batch_predictions = np.concatenate(batch_predictions).ravel()
                #values_tens = (torch.Tensor(batch_values))
                #predictions_tens = (torch.Tensor(batch_predictions))
                #accuracy, precision, recall, f1_score = F_score(predictions_tens, values_tens.float())

                accuracy = np.mean(acc_list)
                f1_score = np.mean(f1_score_list)

                validation_loss = np.mean(val_losses)
                self.val_loss.append(validation_loss)

            # create checkpoint variable and add important data
            checkpoint = {
                'epoch': epoch + 1,
                'validation_min_loss': validation_loss,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }

            print(
                f"[{epoch}/{n_epochs}] Training loss: {train_step_loss:.4f}\t Validation loss: {validation_loss:.4f}"
            )
            print("accuracy: {:0.2f}%,".format(accuracy * 100), "f1_score: {:0.2f}%".format(f1_score * 100))
            # save checkpoint
            check_path = Path('checkpoint.pt')
            best_check_path = Path('best_checkpoint.pt')

            save_ckp(checkpoint, False, check_path, best_check_path)

            if f1_score >= f1_max:
                counter_epoch = 0
                print('f1_score increased({:.6f} --> {:.6f}).'.format(f1_max, f1_score))
                # save checkpoint as best model
                save_ckp(checkpoint, True, check_path, best_check_path)
                f1_max = f1_score

            if (epoch > 20) & (counter_epoch >= 15):
                break

            # if validation_loss <= validation_min_loss:
            #     print('Validation loss decreased ({:.6f} --> {:.6f}).'.format(validation_min_loss,
            #                                                                   validation_loss))
            #     # save checkpoint as best model
            #     save_ckp(checkpoint, True, check_path, best_check_path)
            #     validation_min_loss = validation_loss
            print("\n")

    def plot_losses(self):
        plt.plot(self.train_loss, label="Training loss")
        plt.plot(self.val_loss, label="Validation loss")
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        #plt.margins(1, 1)
        plt.legend()
        plt.title("LSTM Losses")
        plt.show()
        plt.savefig("LSTM_binary_class_losses.png")
        plt.close()

    def evaluate(self, test_loader, best_model):

        best_model.eval()
        with torch.no_grad():
            predictions = []
            values = []

            for test_batch_idx, test_batch_info in enumerate(test_loader):
                X_test = test_batch_info[0]
                y_test = test_batch_info[1] # actual values
                X_test_original_len = test_batch_info[2]

                class_labels = list(set(y_test))

                y_pred = best_model(X_test.float(), X_test_original_len)
                y_pred = y_pred.squeeze().float()

                # retrieve numpy array
                y_pred = y_pred.detach().numpy()

                # round to class values
                #y_pred = y_pred.round()

                predictions.append(y_pred)

                y_test = y_test.detach().numpy()
                values.append(y_test)

        values = np.concatenate(values).ravel()
        predictions = np.concatenate(predictions).ravel()

        #print(predictions.shape)
        values_tensor = (torch.Tensor(values))
        predictions_tensor = (torch.Tensor(predictions))

        print('\nClassification Report:')
        accuracy, precision, recall, F1_score, cm = F_score(predictions_tensor, values_tensor)
        print(cm)
        print("accuracy: {:0.2f}%,".format(accuracy*100),
              "precision: {:0.2f}%,".format(precision*100),
              "recall: {:0.2f}%,".format(recall*100),
              "F1_score: {:0.2f}%".format(F1_score*100))

        #print('Classification Report:')
        #print(classification_report(values, predictions, labels=[1, 0], digits=4))

        #cm = confusion_matrix(values, predictions, labels=[1, 0])
        return predictions, values


def get_model(model, model_params):

    # for multiple models
    models = {
        "lstm": LSTMModel
    }

    return models.get(model.lower())(**model_params)


if __name__ == "__main__":
    warnings.filterwarnings('ignore')

    args = parse_arguments()
    videos_path = args.videos_path

    # parameters
    input_size = 88 # num of features
    #input_size = 43  # num of features
    output_size = 1
    hidden_size = 32
    num_layers = 1
    batch_size = 64
    dropout = 0.5
    n_epochs = 100
    learning_rate = 1e-3
    weight_decay = 1e-1

    model_params = {'input_size': input_size,
                    'hidden_size': hidden_size,
                    'num_layers': num_layers,
                    'output_size': output_size,
                    'dropout_prob': dropout}

    videos_path = [item for sublist in videos_path for item in sublist]

    seed_all(41)
    dataset = create_dataset(videos_path)
    train_loader, val_loader, test_loader = data_preparation(
        dataset, batch_size=batch_size)

    model = get_model('lstm', model_params)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(),
                           lr=learning_rate, weight_decay=weight_decay)

    opt = Optimization(model=model, loss_fn=criterion, optimizer=optimizer)

    opt.train(train_loader, val_loader, n_epochs=n_epochs)
    opt.plot_losses()

    ckp_path = "best_checkpoint.pt"
    best_model, optimizer, start_epoch, \
    validation_min_loss = load_ckp(ckp_path, model, optimizer)

    # print("\nAfter validation:")
    # print("model = ", model)
    # print("optimizer = ", optimizer)
    # print("start_epoch = ", start_epoch)
    # print("validation_min_loss = ", validation_min_loss)
    # print("validation_min_loss = {:.6f}".format(validation_min_loss), "\n")

    predictions, values = opt.evaluate(test_loader, best_model)

