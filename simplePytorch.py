import torch
import torch.nn as nn


class imgNN(nn.Module):
    def __init__(self):
        """imgNN class defines the model that is used to process images.

        This model is cnn -> batch normalization -> relu -> dropout -> maxpool -> cnn -> batch normalization -> relu -> dropout -> maxpool

        This network produces just an intermediate tensor that is used by another network to classify the input images.
        """
        super(imgNN,self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=3, out_channels = 16, padding=1, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(num_features=16)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout2d(p=0.3)
        self.maxPool = nn.MaxPool2d(2)
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels = 32, kernel_size = 3)
        self.bn2 = nn.BatchNorm2d(num_features = 32)
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout2d(p=0.12)
        self.maxPool1 = nn.MaxPool2d(2)

    def forward(self, input):
        """The forward propogation in the NN. This is a simple NN and doesn't need much processing.
        
        Arguments:
            input {Pytorch tensor} -- The tensor image that is passed to the network to be processed.
        """
        output = self.cnn1(input)
        output = self.bn1(output)
        output = self.relu1(output)
        output = self.dropout(output)
        output = self.maxPool(output)
        output = self.cnn2(output)
        output = self.bn2(output)
        output = self.relu2(output)
        output = self.dropout1(output)
        output = self.maxPool1(output)
        return output

class clfNN(nn.Module):
    def __init__(self, numberofClasses = 11):
        """This is the classification NN. This NN uses two image to classify an appliance. 
        Classifies the applications by building two NN using the imgNN model. 
        Then stacks the output tensors of the basic NN and then builds two DNNs to classify the images.
        
        
        Keyword Arguments:
            numberofClasses {int} -- [Number of classes that are going to be classified] (default: {11})
        """
        super(clfNN,self).__init__()
        self.cimg = imgNN()
        self.vimg = imgNN()
        self.imgnet = nn.Sequential(self.cimg, self.vimg)
        self.fc1 = nn.Linear(in_features = 64*31*28, out_features = 120)
        self.bn3 = nn.BatchNorm1d(120)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features = 120, out_features= 40)
        self.fc = nn.Linear(in_features = 40, out_features = numberofClasses)

    def forward(self, cinput, vinput):
        """Classif forward propogation in PyTorch. 
        Here the tensors from the basic NN are concatinated i.e. stacked, flattened and passed through two DNNs to classify the image.
        
        Arguments:
            cinput {[Pytorch tensor]} -- [The tensor for the processed image that represents the currrent spectrum]
            vinput {[Pytorch tensor]} -- [The tensor for the processed image that represents the voltage spectrum]
        """
        coutput = self.cimg(cinput)
        voutput = self.vimg(vinput)
        output = torch.cat((coutput, voutput),1)
        output = output.view(output.size(0), -1)
        output = self.fc1(output)
        output = self.bn3(output)
        output = self.fc2(output)
        output = self.fc(output)
        return output