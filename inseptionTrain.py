import torch
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, datasets, models
from torchvision.datasets import ImageFolder
import time
from pytorchClf import superSimpleNet
from simplePytorch import clfNN
import pandas as pd
from imageStackTest import get_loaders, AppDataset

batch_size = 32  # defines the batch size that we will be working with


# gets the PyTorch loader functions for training and validation datasets.
train_loader, valid_loader = get_loaders()

dataset_sizes = {
    'train': len(train_loader.dataset),
    'test': len(valid_loader.dataset)
}  # defines the dataset size for training and test/validation


def train_model(dataloaders, model, criterion, optimizer, scheduler, num_epochs=10):
    """Trains the NN model that is passed to it as an argument.

    Arguments:
        dataloaders {torch.utils.data.DataLoader} -- The dataloader that will give us the batched data that the model will be trained on
        model {torch.nn.Module} -- The model that needs to be trained.
        criterion {function : torch.nn.CrossEntropyLoss} -- This describes the loss function that is used.
        optimizer {one of torch.optim} -- The optimizer that is used.
        scheduler  -- The scheduler used to control the learning rate.

    Keyword Arguments:
        num_epochs {int} -- The number epochs that the model needs to be trained. (default: {10})

    Return:
        The most accurate model that was trained.
    """
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            running_batch = 0

            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                cinputs, vinputs, labels = data
                labels = labels.view(-1)

                # wrap them in Variable
                if use_gpu:
                    cinputs = Variable(cinputs.cuda())
                    vinputs = Variable(vinputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    cinputs, vinputs, labels = Variable(
                        cinputs), Variable(vinputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(cinputs, vinputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.item()
                running_corrects += torch.sum(preds == labels.data)
                running_batch += 1

            epoch_loss = running_loss / running_batch
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


use_gpu = torch.cuda.is_available()

model = clfNN(11)  # Create a clfNN model to classify 11 classes of data.
if use_gpu:
    model = model.cuda()

# Using a cross entrophy loss function.
criterion = torch.nn.CrossEntropyLoss()
# SGD is the optimizer used for this model.Learning rate that starts with 0.001.
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# The loaders converted into a dict for ease of use.
loaders = {'train': train_loader, 'test': valid_loader}


def makeSubmission(modelFile, csvFile):
    """This builds the submission.csv file based on the prediction of the best model trained.

    Arguments:
        modelFile {str} -- Path to the model file on disk.
        csvFile {str} -- Path to the submission.csv file on disk.
    """
    pred_lables = []
    pdf = pd.read_csv(
        "D:\\Work\\Progarmming\\Datasets\\data-release\\submission_format.csv")
    model = torch.load(modelFile)
    test_set = AppDataset(
        pdf, "D:\\Work\\Progarmming\Datasets\\data-release\\pytest_processed_unsplit")
    valid_loader = DataLoader(test_set, batch_size=1, num_workers=4)
    for cimages, vimages, lable in valid_loader:
        if use_gpu:
            cimages = Variable(cimages.cuda())
            vimages = Variable(vimages.cuda())
        # Predict classes using images from the test set
        outputs = model(cimages, vimages)
        _, prediction = torch.max(outputs.data, 1)
        pred_lables.append(prediction.data)

    lables = pd.DataFrame({"appliance": pred_lables})
    pdf.update(lables)
    pdf.to_csv(csvFile, index=False)


if __name__ == "__main__":
    model = train_model(loaders, model, criterion, optimizer,
                        exp_lr_scheduler, num_epochs=25)
    #torch.save(model, "BestSimplePytorch_stackedv2.model")
    makeSubmission("BestSimplePytorch_stackedv2.model",
                   "submission_pytorch_stackedv2.csv")
