import os
import sys
import torch
import random
import shutil
import warnings
import argparse
import numpy as np
import torch.nn as nn
from pathlib import Path
from torch.nn import init
import torch.optim as optim
import sklearn.metrics as metrics
from collections import OrderedDict
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report
from torch.nn.utils.rnn import pad_sequence as pad
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score, accuracy_score
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from matplotlib import pyplot
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


"""
RUN (binary classification):
big dataset (941 Static VS 583 Non Static):
python3 LSTM_pytorch.py -v /media/ubuntu/Seagate/ChromeDownloads/dataset_annotated_4/Non_Static_88_feat /media/ubuntu/Seagate/ChromeDownloads/dataset_annotated_4/Static_88_feat /media/ubuntu/Seagate/ChromeDownloads/dataset_annotated_4/Handled
"""


def parse_arguments():
    """
    Parse arguments for real time demo.
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
    #random.seed(seed)
    #np.random.seed(seed)


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


class TimeSeriesStandardScaling():
    def __init__(self):
        pass

    def fit(self, X):
        self.X = X
        k = 0
        mean_std_tuples = []
        while k < self.X[0].shape[1]:
            temp = np.concatenate([i[:, k] for i in self.X])
            mean_std_tuples.append(
                tuple((np.mean(temp), np.std(temp))))
            k += 1

        self.scale_params = mean_std_tuples

    def transform(self, X):
        X_scaled = []
        tmp = list(zip(*self.scale_params))
        mu = torch.Tensor(list(tmp[0]))
        std = torch.Tensor(list(tmp[1]))

        for inst in X:
            v = (inst - mu) / std
            X_scaled.append(v)
        self.X_scaled = X_scaled

        return X_scaled

    def fit_transform(self, X):
        self.fit(X)
        self.transform(self.X)
        return self.X_scaled


class TimeSeriesMinMaxScaling():
    def __init__(self):
        pass

    def fit(self, X):
        self.X = X
        k = 0
        min_max_tuples = []
        while k < self.X[0].shape[1]:
            temp = np.concatenate([i[:, k] for i in self.X])
            min_max_tuples.append(
                tuple((np.min(temp), np.max(temp))))
            k += 1

        self.scale_params = min_max_tuples

    def transform(self, X):
        X_scaled = []
        tmp = list(zip(*self.scale_params))
        min_feature = torch.Tensor(list(tmp[0]))
        max_feature = torch.Tensor(list(tmp[1]))

        for inst in X:
            v = (inst - min_feature) / max_feature - min_feature
            X_scaled.append(v)
        self.X_scaled = X_scaled

        return X_scaled

    def fit_transform(self, X):
        self.fit(X)
        self.transform(self.X)
        return self.X_scaled


def load_data(X, y, check_train, scaler):
    # for train/val/test sets
    split_dataset = []
    for i, j in zip(X, y):
        split_dataset.append(tuple((i, j)))

    x_len = []
    X = []
    labels = []

    for index, data in enumerate(split_dataset):
        """
        data[0] corresponds to ---> .npy shot names
        data[1] corresponds to ---> y labels
        """

        X_to_tensor = np.load(data[0])
        # keep only specific features (remove RGB histogram-based features)
        X_to_tensor = X_to_tensor[:, 45:89]

        y = data[1]
        labels.append(y)
        x_len.append(X_to_tensor.shape[0])
        X.append(torch.Tensor(X_to_tensor))

    # data normalization
    if check_train == True:
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    return X_scaled, labels, x_len


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

    X = [x[0] for x in videos_dataset]
    y = [x[1] for x in videos_dataset]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42,
                                                        test_size=0.2, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state=42,
                                                      test_size=0.13, stratify=y_train)

    # Define Scaler
    scaler = TimeSeriesStandardScaling()
    #scaler = TimeSeriesMinMaxScaling()

    X_train, y_train, train_lengths = load_data(X_train, y_train, True, scaler=scaler)
    train_dataset = LSTMDataset(X_train, y_train, train_lengths)

    X_val, y_val, val_lengths = load_data(X_val, y_val, False, scaler=scaler)
    val_dataset = LSTMDataset(X_val, y_val, val_lengths)

    X_test, y_test, test_lengths = load_data(X_test, y_test, False, scaler=scaler)
    test_dataset = LSTMDataset(X_test, y_test, test_lengths)

    # Define a DataLoader for each set
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              collate_fn=my_collate, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                              collate_fn=my_collate, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                              collate_fn=my_collate, shuffle=True)

    return train_loader, val_loader, test_loader


def plot_roc_curve(fpr, tpr, roc_auc):
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.05, 1])
    plt.ylim([0, 1.05])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    plt.savefig("ROC Curve - Binary Classification.png")
    plt.close()


def plot_precision_recall_curve(precision, recall, y_test):
    no_skill = len(y_test[y_test == 1]) / len(y_test)
    pyplot.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    pyplot.plot(recall, precision, marker='.', label='LSTM')
    pyplot.xlabel('Recall')
    pyplot.ylabel('Precision')
    pyplot.legend()
    pyplot.show()
    plt.savefig("Precision-Recall Curve - Binary Classification.png")
    plt.close()


def calculate_metrics(y_pred, y_test):



    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

    y_pred = y_pred_tags.float()

    cm = confusion_matrix(y_test, y_pred)
    f1_score_macro = f1_score(y_test, y_pred, average='macro')
    acc = accuracy_score(y_test, y_pred)
    precision_recall_fscore = precision_recall_fscore_support(y_test, y_pred, average='macro')

    # # ROC Curve
    # fpr, tpr, threshold = metrics.roc_curve(actual_values, predicted_values)
    # precision, recall, thresholds = metrics.precision_recall_curve(actual_values, predicted_values)
    # roc_auc = metrics.auc(fpr, tpr)
    #
    # # print("fpr", fpr, "tpr", tpr, "threshold", threshold)
    # plot_roc_curve(fpr, tpr, roc_auc)
    # plot_precision_recall_curve(precision, recall, actual_values)

    return acc, f1_score_macro, cm, precision_recall_fscore


def weight_init(m):
    if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
        #init.normal_(m.weight.data, 0.0, 0.02)
        init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01) # or 0
        #print("VS m: ", m.weight, "\n")
    else:
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                init.xavier_uniform_(param.data)
                #print(param.data)
            elif 'weight_hh' in name:
                # for LSTM layer
                init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,
                 output_size, dropout_prob):
        super(LSTMModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True)#, dropout=dropout_prob)

        self.drop = nn.Dropout(p=dropout_prob)

        self.fnn = nn.Sequential(OrderedDict([
            ('relu1', nn.ReLU()),
            ('drop1', nn.Dropout(0.4)),
            ('fc1', nn.Linear(self.hidden_size, 512)),
            ('bn1', nn.BatchNorm1d(512)),
            ('relu2', nn.ReLU()),
            ('drop2', nn.Dropout(0.4)),
            ('fc2', nn.Linear(512, 256)),
            ('bn2', nn.BatchNorm1d(256)),
            ('relu3', nn.ReLU()),
            ('drop3', nn.Dropout(0.3)),
            ('fc3', nn.Linear(256, 128)),
            ('bn3', nn.BatchNorm1d(128)),
            ('relu4', nn.ReLU()),
            ('drop4', nn.Dropout(0.3)),
            ('fc4', nn.Linear(128, 64)),
            ('bn4', nn.BatchNorm1d(64)),
            ('relu5', nn.ReLU()),
            ('drop5', nn.Dropout(0.2)),
            ('fc5', nn.Linear(64, 32)),
            ('bn5', nn.BatchNorm1d(32)),
            ('relu6', nn.ReLU()),
            ('drop6', nn.Dropout(0.2)),
            ('fc6', nn.Linear(32, output_size))
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
        output, (hn, cn) = self.lstm(X, (h0.detach(), c0.detach()))  # output shape:(batch_size,seq_length,hidden_size)
        last_states = self.last_by_index(output, lengths)

        #last_states = self.drop(last_states)

        output = self.fnn(last_states)

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
    def __init__(self, model, loss_fn, optimizer, scheduler):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loss = []
        self.val_loss = []

    def train(self, train_loader, val_loader, n_epochs):
        f1_max = -1.0
        counter_epoch = 0

        for epoch in range(1, n_epochs + 1):
            counter_epoch += 1

            train_losses = []
            val_losses = []

            val_predictions = []
            val_values = []

            scheduler.step(epoch)

            # enumerate mini batches
            for batch_idx, batch_info in enumerate(train_loader):
                """
                batch_idx -------> batch id
                batch_info[0] ---> padded arrays in each batch
                batch_info[1] ---> labels (y) in each batch
                batch_info[2] ---> original length of each sequence
                """

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
                    output = out.squeeze().float()

                    # Calculate loss
                    loss = self.loss_fn(output, y_train.type(torch.LongTensor))

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

                    val_loss = self.loss_fn(y_hat, y_val.type(torch.LongTensor))
                    val_losses.append(val_loss)

                    val_predictions.append(y_hat)
                    val_values.append((y_val.float()))

                val_values = np.concatenate(val_values).ravel()
                val_predictions = np.concatenate(val_predictions)

                val_values_tensor = (torch.Tensor(val_values))
                val_predictions_tensor = (torch.Tensor(val_predictions))

                accuracy, f1_score_macro, cm, _ = calculate_metrics(val_predictions_tensor, val_values_tensor)

                validation_loss = np.mean(val_losses)
                self.val_loss.append(validation_loss)
                print(cm)

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
            print("accuracy: {:0.2f}%,".format(accuracy * 100), "f1_score: {:0.2f}%".format(f1_score_macro * 100))
            # save checkpoint
            check_path = Path('checkpoint.pt')
            best_check_path = Path('best_checkpoint.pt')

            save_ckp(checkpoint, False, check_path, best_check_path)

            if f1_score_macro > f1_max:
                counter_epoch = 0
                print('f1_score increased({:.6f} --> {:.6f}).'.format(f1_max, f1_score_macro))
                # save checkpoint as best model
                save_ckp(checkpoint, True, check_path, best_check_path)
                f1_max = f1_score_macro

            if counter_epoch >= 20:
                break

            print("\n")

    def plot_losses(self):
        #plt.figure(figsize=(10, 10))
        plt.title("LSTM Training and Validation Loss")
        plt.plot(self.train_loss, label="Training loss")
        plt.plot(self.val_loss, label="Validation loss")
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        #plt.margins(1, 1)
        plt.legend()
        #plt.title("LSTM Losses")
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

                y_pred = best_model(X_test.float(), X_test_original_len)
                y_pred = y_pred.squeeze().float()

                # retrieve numpy array
                y_pred = y_pred.detach().numpy()
                predictions.append(y_pred)

                y_test = y_test.detach().numpy()
                values.append(y_test)


        values = np.concatenate(values).ravel()
        predictions = np.concatenate(predictions)

        print("VALUES: ", len(values))
        print("VS PREDICTIONS: ", len(predictions))

        values_tensor = (torch.Tensor(values))
        predictions_tensor = (torch.Tensor(predictions))

        accuracy, f1_score_macro, cm, _ = calculate_metrics(predictions_tensor, values_tensor)

        print("\nClassification Report:\n"
              "accuracy: {:0.2f}%,".format(accuracy * 100),
              #"precision: {:0.2f}%,".format(precision_recall[0] * 100),
              #"recall: {:0.2f}%,".format(precision_recall[1]*100),
              "f1_score (macro): {:0.2f}%".format(f1_score_macro * 100))
        print("\nConfusion matrix\n", cm)
        #
        # print("y_test: ", values_tensor)
        # print("y_pred: ", predictions_tensor)

        return predictions, values, cm


def plot_binary_cm(conf_matrix):
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center', size='xx-large')

    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.show()
    plt.savefig('binary_confusion_matrix.png')


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
    seed_all(42)

    # LSTM parameters
    n_epochs = 1
    num_layers = 1
    batch_size = 64

    # for 3-class classification
    output_size = 3

    input_size = 43
    hidden_size = 256
    dropout = 0.2
    learning_rate = 1e-2
    weight_decay = 1e-4

    model_params = {'input_size': input_size,
                    'hidden_size': hidden_size,
                    'num_layers': num_layers,
                    'output_size': output_size,
                    'dropout_prob': dropout}

    videos_path = [item for sublist in videos_path for item in sublist]

    dataset = create_dataset(videos_path)
    train_loader, val_loader, test_loader = data_preparation(
        dataset, batch_size=batch_size)

    model = get_model('lstm', model_params)

    #initialize weights for both LSTM and Sequential

    model.lstm.apply(weight_init)
    for submodule in model.fnn:
        submodule.apply(weight_init)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),
                           lr=learning_rate, weight_decay=weight_decay)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True)
    opt = Optimization(model=model, loss_fn=criterion, optimizer=optimizer, scheduler=scheduler)

    opt.train(train_loader, val_loader, n_epochs=n_epochs)
    opt.plot_losses()

    ckp_path = "best_checkpoint.pt"
    best_model, optimizer, start_epoch, \
    best_f1_score = load_ckp(ckp_path, model, optimizer)

    predictions, values, binary_confusion_matrix = opt.evaluate(test_loader, best_model)
    #plot_binary_cm(binary_confusion_matrix)