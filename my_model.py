import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
import datetime


class MyModel:
    def __init__(self, arch='resnet50', hidden_layers=512, gpu=False):
        self.arch = arch
        self.hidden_layers = hidden_layers
        if arch == 'resnet50':
            self.__model = models.resnet50(pretrained=True)
            classifier_input_nodes = 2048
        elif arch == 'densenet121':
            self.__model = models.densenet121(pretrained=True)
            classifier_input_nodes = 1024
        else:
            raise Exception(f"Unsupported architecture {arch}. Please choose from resnet50 or densenet121")

        for param in self.__model.parameters():
            param.requires_grad = False

        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(classifier_input_nodes, self.hidden_layers)),
            ('relu', nn.ReLU()),
            ('dropout', nn.Dropout(0.2)),
            ('fc2', nn.Linear(self.hidden_layers, 102)),
            ('logsoftmax', nn.LogSoftmax(dim=1))
        ]))

        if arch == 'resnet50':
            self.__model.fc = classifier
        else:
            self.__model.classifier = classifier

        self.device = torch.device("cuda" if gpu else "cpu")
        self.__model.to(self.device)

        self.criterion = self.optimizer = None
        print(f'Created model based on {arch} with {hidden_layers} hidden layers')

    def train(self, epochs=20, lr=0.003, trainloader=None, validloader=None):
        self.criterion = nn.NLLLoss()
        if self.arch == 'resnet50':
            self.optimizer = optim.Adam(self.__model.fc.parameters(), lr=lr)
        else:
            self.optimizer = optim.Adam(self.__model.classifier.parameters(), lr=lr)

        train_losses, valid_losses, valid_accuracies = [], [], []
        dt = datetime.datetime.now()
        print(f"{dt} - Start training on {self.device} for {epochs} epochs, lr = {lr}")

        for e in range(epochs):
            self.__model.train()
            train_loss = 0
            for images, labels in trainloader:
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()

                logps = self.__model.forward(images)
                loss = self.criterion(logps, labels)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            train_losses.append(train_loss / len(trainloader))

            valid_loss, valid_accuracy = self.test(validloader)
            valid_losses.append(valid_loss / len(validloader))
            valid_accuracies.append(valid_accuracy / len(validloader))

            dt = datetime.datetime.now()

            print(f'{dt} - Epoch {e + 1}, '
                  f'train_loss = {train_loss / len(trainloader)}, '
                  f'valid_loss = {valid_loss / len(validloader)}, '
                  f'valid_accuracy = {valid_accuracy / len(validloader)}')
        return train_losses, valid_losses, valid_accuracies

    def test(self, data_loader=None):
        test_loss = 0
        test_accuracy = 0
        with torch.no_grad():
            self.__model.eval()
            for images, labels in data_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                logps = self.__model.forward(images)
                loss = self.criterion(logps, labels)

                test_loss += loss.item()

                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                test_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        return test_loss, test_accuracy

    def save(self, directory, train_data, epochs, lr):
        self.__model.class_to_idx = train_data.class_to_idx
        data = {
            'arch': self.arch,
            'class_to_idx': self.__model.class_to_idx,
            'state_dict': self.__model.state_dict(),
            'saved_on_device': self.device.type,
            'hidden_layers': self.hidden_layers,
            'epochs': epochs,
            'lr': lr
        }
        # filename contains base architecture and the device on which it was trained
        filename = f'{directory}/{self.arch}_{self.device.type}_checkpoint.pth'
        torch.save(data, filename)

    def get_model(self):
        return self.__model

