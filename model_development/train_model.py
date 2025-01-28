import os
import torch
import torchvision
import torchviz
import torchsummary

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

import model_utils as mu
import model_architecture as ma

# Utilise any available GPU resources
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Current device type: ", device)

#################################################################################################
# Config
#################################################################################################

# Env Config
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
ARTEFACTS_DIR = os.path.join(BASE_DIR, "model_reporting")
TRAIN_FILE = "mnist_train.csv"
TEST_FILE = "mnist_test.csv"

# Model Config
init_lr = 0.001
batch_size = 100
epochs = 30
img_size = 28
num_classes = 10
class_names = {
    0: "Zero",
    1: "One",
    2: "Two",
    3: "Three",
    4: "Four",
    5: "Five",
    6: "Six",
    7: "Seven",
    8: "Eight",
    9: "Nine",
}

# Training Config
split_size = 0.2


#################################################################################################
# Data Load & Train Test Split & Prep
#################################################################################################

train_set = mu.load_dataset(TRAIN_FILE, DATA_DIR)
test_set = mu.load_dataset(TEST_FILE, DATA_DIR)

# Split the data into train/test splits - split_size defined in config
train_images, test_images, train_labels, test_labels = train_test_split(
    train_set.iloc[:, 1:], train_set.iloc[:, 0], test_size=split_size
)

# Reset the index values
train_images.reset_index(drop=True, inplace=True)
test_images.reset_index(drop=True, inplace=True)
train_labels.reset_index(drop=True, inplace=True)
test_labels.reset_index(drop=True, inplace=True)

# Some pretty cool random augmentations
train_trans = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.RandomCrop(img_size),
        torchvision.transforms.RandomRotation(90),
        torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2),
        torchvision.transforms.RandomAffine(degrees=5, translate=(0.1, 0.1)),
        torchvision.transforms.ToTensor(),
    ]
)

test_trans = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.ToTensor(),
    ]
)

# Perform our transforms and augmentations
train_set = mu.MNISTDataSet(train_images, img_size, train_labels, train_trans)
test_set = mu.MNISTDataSet(test_images, img_size, test_labels, test_trans)

# Create dataset dataloaders
training_dataloader = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, shuffle=True
)
test_dataloader = torch.utils.data.DataLoader(
    test_set, batch_size=batch_size, shuffle=False
)

#################################################################################################
# Neural Net Definition
#################################################################################################

net = ma.Network().to(device)

#################################################################################################
# Neural Net Architecture
#################################################################################################

torchsummary.summary(net, input_size=(1, 28, 28))

# Eval mode for visualising
net.eval()

# Dummy input tensor (e.g., 1 grayscale image of size 28x28)
input_tensor = torch.randn(1, 1, 28, 28)

output = net(input_tensor)
graph = torchviz.make_dot(output, params=dict(net.named_parameters()))
graph.render(os.path.join(ARTEFACTS_DIR, "neural_network_graph"), format="pdf")

# Revert to training mode
net.train()

#################################################################################################
# Optimizer & Scheduler
#################################################################################################

optimizer = torch.optim.Adam(net.parameters(), lr=init_lr)

scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

#################################################################################################
# Training
#################################################################################################

lrs = []
train_loss = []
train_acc = []
test_loss = []
test_acc = []

# Run our epoch cycles
for epoch in range(epochs):
    # Print epoch cycle
    print(f"Epoch Cycle: {epoch+1}")

    # Train the model, and append the current learning rate
    lr, trl, tra = mu.train_model(
        net, device, optimizer, scheduler, training_dataloader, batch_size
    )

    lrs.append(lr)
    train_loss.append(trl / batch_size)
    train_acc.append(tra)

    # Evaluate the model, return the validation loss and validation accuracy
    tel, tea = mu.eval_model(net, device, test_dataloader, len(test_images), batch_size)

    test_loss.append(tel / batch_size)
    test_acc.append(tea)

#################################################################################################
# Evaluation
#################################################################################################

# Create empty tensor for predictions
predictions = torch.LongTensor().to(device)

# Use trained model to generate predictions
for images, _ in test_dataloader:
    preds = net(images.to(device))
    predictions = torch.cat((predictions, preds.argmax(dim=1)), dim=0)

# Compute metrics
accuracy = accuracy_score(test_labels, predictions)
precision = precision_score(test_labels, predictions, average="weighted")
recall = recall_score(test_labels, predictions, average="weighted")
f1 = f1_score(test_labels, predictions, average="weighted")

# Write metrics to file
metrics_file = os.path.join(ARTEFACTS_DIR, "metrics.txt")
with open(metrics_file, "w") as outfile:
    outfile.write(
        f"Accuracy: {round(accuracy, 4)}\nPrecision: {round(precision, 4)}\nRecall: {round(recall, 4)}\nF1 Score: {round(f1, 4)}"
    )

# Save confusion matrix plot
mu.confusion_matrix(
    num_classes,
    test_labels,
    predictions,
    save_path=os.path.join(ARTEFACTS_DIR, "confusion_matrix.png"),
)

epoch_num = list(range(1, epochs + 1))

plt.figure(figsize=(12, 7))
plt.title("Neural Network Learning Rate Curve")
plt.xlabel("Epoch")
plt.ylabel("Learning Rate Value")
plt.yscale("log")
plt.grid()
plt.plot(epoch_num, lrs, "g-")
plt.xticks(epoch_num)
plt.savefig(os.path.join(ARTEFACTS_DIR, "nn_lr_curve.png"))

nl = [x / 1000 for x in train_loss]

plt.figure(figsize=(12, 7))
plt.title("Neural Network Loss Curves")
plt.xlabel("Epoch")
plt.ylabel("Loss Value")
plt.grid()
plt.plot(epoch_num, nl, "b-", label="training loss")
plt.plot(epoch_num, test_loss, "r-", label="validation loss")
plt.legend(loc="upper right")
plt.xticks(epoch_num)
plt.savefig(os.path.join(ARTEFACTS_DIR, "nn_loss_curve.png"))

plt.figure(figsize=(12, 7))
plt.title("Neural Network Accuracy Curves")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.ylim(97, 100)
plt.grid()
plt.plot(epoch_num, train_acc, "b-", label="training accuracy")
plt.plot(epoch_num, test_acc, "r-", label="validation accuracy")
plt.legend(loc="lower right")
plt.xticks(epoch_num)
plt.savefig(os.path.join(ARTEFACTS_DIR, "nn_acc_curve.png"))

#################################################################################################
# Export Model
#################################################################################################

torch.save(net, os.path.join(BASE_DIR, "app", "model", "model.pth"))
