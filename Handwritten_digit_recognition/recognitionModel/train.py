import argparse

import torch
import torchvision

from recognitionModel.models.lenet import LeNet
from recognitionModel.utils import pre_process

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def get_data_loader(batch_size):
    # MNIST dataset
    train_dataset = torchvision.datasets.MNIST(root='data/',
                                               train=True,
                                               transform=pre_process.data_augment_transform(),
                                               download=True)

    test_dataset = torchvision.datasets.MNIST(root='data/',
                                              train=False,
                                              transform=pre_process.normal_transform())

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)

    return train_loader, test_loader


def evaluate(model, test_loader, device):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Test Accuracy of the model is: {} %'.format(100 * correct / total))

    return 100 * correct / total


def save_model(model, save_path):
    ckpt_dict = {
        'state_dict': model.state_dict()
    }
    torch.save(ckpt_dict, save_path)


def train(epochs, batch_size, learning_rate, num_classes, name):
    # Fetch data
    train_loader, test_loader = get_data_loader(batch_size)

    # Loss and optimizer
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = LeNet(num_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Start training
    total_step = len(train_loader)
    loss_epoch = {}
    accuracy_epoch = {}

    for epoch in range(epochs):
        loss_all = 0
        loss_num = 0

        for i, (images, labels) in enumerate(train_loader):
            # Get image and label
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_all += loss.item()
            loss_num = i

            if (i + 1) % 100 == 0:
                # Print information for each epoch
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, epochs, i + 1, total_step, loss.item()))

        # Record the loss of the current epoch
        loss_epoch.update({f"epoch {epoch + 1}": round(loss_all / loss_num, 4)})

        # Evaluate after training
        ac = evaluate(model, test_loader, device)
        accuracy_epoch.update({f"epoch {epoch + 1}": round(ac, 4)})

    # Save the trained model
    save_model(model, save_path=f'recognitionModel/models/{name}.pth')

    return loss_epoch, accuracy_epoch
