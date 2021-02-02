# COSD598_Warmup -- MNIST Classification
This is a warmup task to let you get familiar with Azure Cloud Platform and running deep learning model using Pytorch.

## Task1

###Get started with Azura

To get started with Azure, I’d recommend you to view videos 1 – 5 in the Azure for Academic series found here:

<a href="https://channel9.msdn.com/Series/Azure-for-Academics">Azure for Academics | Channel 9 (msdn.com)</a>. You might find the rest of the series useful, but they can skip the 6th video as they will have already have an Azure account.

### Virtual Machine

As for resources, here’s step-by-step guided lessons from Microsoft Learn on virtual machines:
<a href="https://docs.microsoft.com/en-us/learn/modules/intro-to-azure-virtual-machines/">Introduction to Azure virtual machines - Learn | Microsoft Docs</a>

<a href="https://docs.microsoft.com/en-us/learn/modules/create-windows-virtual-machine-in-azure/">Create a Windows virtual machine in Azure - Learn | Microsoft Docs</a>

<a href="https://docs.microsoft.com/en-us/learn/modules/create-linux-virtual-machine-in-azure/">Create a Linux virtual machine in Azure - Learn | Microsoft Docs</a>
 
This video is focused on troubleshooting VMs: 

<a href="https://www.youtube.com/watch?v=MAJrN-2IPY8">How to troubleshoot common virtual machine issues | Azure Portal Series - YouTube</a>

## Task 2
Please run the model with the default setting in main.py. You do not need to specify any configuration. 

***Model and Dataset***
In this experiment, you will train convolutional neural networks (CNN) for MNIST dataset classification. MNIST dataset contains hand-writing digits 0-9. When you run the code, data will be automatically downloaded and split into training and testing datasets. Training data will be used for CNN training and testing data will be used for evaluation. 

***CNN archietecture***
You are required to learn how to build your own deep learning model. 

## What to be included in your report?
Please record training loss, testing loss, training accuracy, and testing accuracy.

### How to run?
```bash
pip install -r requirements.txt
python main.py
```
