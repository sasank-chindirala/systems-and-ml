# COSD598_Warmup -- MNIST Classification
This is a warmup task to let you get familiar with running deep learning model using Pytorch.

### Model and Dataset

In this experiment, you will train convolutional neural networks (CNN) for MNIST dataset classification. MNIST dataset contains hand-writing digits 0-9. When you run the code, data will be automatically downloaded and split into training and testing datasets. Training data will be used for CNN training and testing data will be used for evaluation. 

### CNN archietecture

You are required to learn how to build your own deep learning model. We offered a CNN with two Conv2D layers and two FC layers as defined below. Let us call the default model as 'model1'.
```
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
```
- [ ] Please add one more Conv2D layer with `channel size  (number of kernels) = 32` between layer `self.conv1` and layer `self.conv2` to model1. You need to modify the `forward` function correspondingly. Let us call this model as 'model2'.
- [ ] Please add one more FC layer with `channel size  (number of kernels) = 128` between layer `self.fc1` and layer `self.fc2` to model1. You need to modify the `forward` function correspondingly. Let us call this model as 'model3'.

Please plot `training loss vs. epoch`, `testing loss vs. epoch`, `training accuracy vs. epoch`, and `testing accuracy vs. epoch` for model1, model2 and model3. You need to write a function(s) to record the results in `main.py`.

### How to run?
```bash
pip install -r requirements.txt
python main.py
```
