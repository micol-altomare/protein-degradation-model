import os 
import pickle
import progressbar
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib

class CNN(nn.Module):
    def __init__(self, nb_filter=1, channel=1, num_classes=1, kernel_size=(1, 10), pool_size=(1, 3), window_size=101 + 6, hidden_size=256, stride=(1, 1), padding=0):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(channel, nb_filter, kernel_size, stride=stride, padding=padding), 
            nn.BatchNorm2d(nb_filter),
            nn.ReLU()) 
        self.pool1 = nn.MaxPool2d(pool_size, stride=stride) 
        out1_size = int((window_size + 2*padding -
                        (kernel_size[1] - 1) - 1)/stride[1] + 1) 
        maxpool_size = int(
            (out1_size + 2*padding - (pool_size[1] - 1) - 1)/stride[1] + 1) 
        self.layer2 = nn.Sequential(
            nn.Conv2d(nb_filter, nb_filter, kernel_size=(
                1, 10), stride=stride, padding=1),
            nn.BatchNorm2d(nb_filter), 
            nn.ReLU(), 
            nn.MaxPool2d(pool_size, stride=stride)) 
        out2_size = int((maxpool_size + 2*padding -
                        (kernel_size[1] - 1) - 1)/stride[1] + 1) 
        maxpool2_size = int((out2_size + 2*padding - (pool_size[1] - 1) - 1)/stride[1] + 1)
        self.drop1 = nn.Dropout(p=0.25) 
        # print('maxpool_size', maxpool_size)
        self.fc1 = nn.Linear(window_size * 2 + 4, hidden_size) # eg 804 for 400 ;;; max_len *2 + 4
        self.drop2 = nn.Dropout(p=0.25)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes) #
        self.cuda = torch.cuda.is_available()

    def forward(self, x):
        out = self.layer1(x)
        out = self.pool1(out)
        out = self.layer2(out)
        out = out.flatten()
        out = self.drop1(out)
        # print('out', out.size())
        out = self.fc1(out)
        out = self.drop2(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = torch.softmax(out, -1) #****
        return out

    def layer1out(self, x):
        if type(x) is np.ndarray:
            x = torch.from_numpy(x.astype(np.float32))
        with torch.no_grad():
            x = torch.autograd.Variable(x)
        # x = Variable(x, volatile=True)
        if self.cuda:
            x = x.cuda()
        out = self.layer1(x)
        temp = out.data.cpu().numpy()
        return temp

    def predict_proba(self, x):
        if type(x) is np.ndarray:
            x = torch.from_numpy(x.astype(np.float32))
        with torch.no_grad():
            x = torch.autograd.Variable(x)
        # x = Variable(x, volatile=True)
        if self.cuda:
            x = x.cuda()
        y = self.forward(x)
        temp = y.data.cpu().numpy()
        return temp[:, 1]

class VGG16(nn.Module):
    def __init__(self, num_classes=1):
        super(VGG16, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=1, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer7 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer8 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer9 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer10 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer11 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer12 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer13 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size = 2, stride = 2)
        )
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(12800, 4096),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(4096, num_classes))

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.layer11(out)
        out = self.layer12(out)
        out = self.layer13(out)
        out = out.flatten()
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

class getDataset(Dataset):

    def __init__(self, x, y):
        self.X_Train = torch.Tensor(x).unsqueeze(1) #.cuda()
       
        s = list(set(y))
        for i in range(len(y)):
            y[i] = s.index(y[i])
        self.Y_Train = torch.LongTensor(y) #.cuda()

    def __len__(self):
        return self.X_Train.size(0)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx #.tolist()

        x = self.X_Train[idx]
        y = self.Y_Train[idx]

        return x, y

if __name__ == '__main__':
    # get pickled train data
    with open('dataset1200_onehot.pickle', 'rb') as f:
        sequences, labels = pickle.load(f)
        f.close()

    # initialize hyperparameters
    num_classes = len(set(labels))
    batch_size = 1
    num_epochs = 500
    learning_rate = 0.0001
    shuffle = True
    weight_decay = 0.0001
    device = 'cpu'
    num_workers = 0

    # Shuffle data non vectorized form
    if shuffle:
        indices = np.arange(len(sequences))
        np.random.shuffle(indices)
        sequences = [sequences[i] for i in indices]
        labels = [labels[i] for i in indices]


    splitIdx = int(0.8*len(sequences)) # 4127 

    # load data
    trainDataset = getDataset(sequences[:splitIdx], labels[:splitIdx])
    testDataset = getDataset(sequences[splitIdx:], labels[splitIdx:])

    # load data loader
    trainLoader = DataLoader(trainDataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    testLoader = DataLoader(testDataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


    # create model
    model = CNN(num_classes=num_classes, window_size=len(sequences[0])) #.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss() #.cuda(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # decrease learning rate if loss does not decrease
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', verbose=True)

    # set cuda to optimize
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # save statistics for analysis
    train_loss = []
    test_loss = []
    test_acc = []

    # load the model checkpoint if it exists
    if os.path.exists('model'):
        # find path to latest model by enumerating all files in the directory
        files = os.listdir('model') 
        for i in range(len(files)): 
            if 'model' in files[i]: 
                files[i] = os.path.join('model', files[i]) 
        latest_file = max(files, key=os.path.getctime) 
        # load the model
        model.load_state_dict(torch.load(latest_file)) # what if we load from the 500th epoch? 
        print('Loaded model from ' + latest_file)

    # Train the model
    for epoch in range(num_epochs):
        # initialize progress bar
        pbar = progressbar.ProgressBar(maxval=len(trainLoader), widgets=["Training model: ", progressbar.Percentage(
        ), " ", progressbar.Bar(), " ", progressbar.ETA()]).start()

        totalLoss = 0
        for i, (sequence, label) in enumerate(trainLoader):
            # Move tensors to the configured device
            sequence = sequence #.to(device)
            label = label #.to(device)

            # Forward pass
            outputs = model(sequence)
            loss = criterion(outputs.unsqueeze(0), label)
            totalLoss += loss.item()
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # progress bar
            pbar.update(i)

        # decrease learning rate if loss does not decrease
        scheduler.step(totalLoss/len(trainLoader))
        pbar.finish()

        print('Epoch [{}/{}], Loss: {:.4f}'
              .format(epoch+1, num_epochs, totalLoss/len(trainLoader)))

        # save statistics for analysis
        train_loss.append(totalLoss/len(trainLoader))

        # Validation
        totalLoss = 0
        with torch.no_grad():
            correct = 0
            total = 0
            for sequence, label in testLoader:
                sequence = sequence #.to(device)
                label = label #.to(device)
                outputs = model(sequence)
                loss = criterion(outputs.unsqueeze(0), label)
                totalLoss += loss.item()
                _, predicted = torch.max(outputs.data, -1)
                total += label.size(0)
                correct += (predicted == label).sum().item()
                del sequence, label, outputs, predicted

            print('Accuracy of the network on the {} validation sequences: {} %'.format(
                len(testLoader)*batch_size, 100 * correct / total))
            print('Loss of the network on the {} validation sequences: {:.4f}'.format(
                len(testLoader)*batch_size, totalLoss/len(testLoader)))

            # save statistics for analysis
            test_loss.append(totalLoss/len(testLoader))
            test_acc.append(100 * correct / total)

            del total, correct

        # Save the model checkpoint
        # if folder does not exist, create it
        if not os.path.exists('model'):
            os.makedirs('model')
        torch.save(model.state_dict(),
                   'model/model_{}.ckpt'.format(epoch+1))

        # plot loss and accuracy
        matplotlib.interactive(False)
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_loss, label='train loss')
        plt.plot(test_loss, label='test loss')
        plt.legend()
        plt.title('Loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.subplot(1, 2, 2)
        plt.plot(test_acc, label='test acc')
        plt.legend()
        plt.title('Accuracy')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.savefig('loss_acc.png')
        plt.close()

    print('Finished Training')


# Note on training: the following code works to train on the CPU. 
# To train on GPU, uncomment the lines with .to(device) .

# Note on learning rate: if cumulatively training on multiple epoch
# rounds, make sure to carry through the learning rate.















