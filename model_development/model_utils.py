import os
import numpy as np
import pandas as pd

import torch

import matplotlib.pyplot as plt
import seaborn as sn


def load_dataset(file_name, data_folder):
    """Read input CSV file for a given file name"""
    # Use os.path.join to construct the full path
    file_path = os.path.join(data_folder, file_name)
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    data = pd.read_csv(file_path)
    return data


def plot_image(label, classnames, image, save_path=None):
    """Plot an example image and class from the dataset, and optionally save it to file"""
    fig1 = plt.figure(figsize=(6, 6))
    fig1.tight_layout()
    plt.title(f"Class: {label}, Name: {classnames[label]}")
    plt.imshow(image.to_numpy().astype(np.uint8).reshape(28, 28), cmap="gray")
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Image saved to {save_path}")


class MNISTDataSet(torch.utils.data.Dataset):
    """Class for handling input CSVs as images using PIL; also handles specified image augmentation"""

    # Get the passed image vector, label vector and transform config
    def __init__(self, images, image_size, labels, transforms=None):
        self.X = images
        self.image_size = image_size
        self.y = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        # Get a single image vector
        data = self.X.iloc[i, :]
        # Reshape the vector into an image of shape 28*28
        data = (
            np.array(data).astype(np.uint8).reshape(self.image_size, self.image_size, 1)
        )

        # Produce any transforms that are provided
        if self.transforms:
            data = self.transforms(data)

        # If the data is a training set, provide the label; otherwise do not
        if self.y is not None:  # train/val
            return (data, self.y[i])
        else:
            return data


def get_num_correct(predictions, labels):
    """Compares model predictions with actual labels, returns the number of matches"""
    return predictions.argmax(dim=1).eq(labels).sum().item()


def eval_model(network, device, validation_data, validation_set_size, batch_size):
    """Evaluation with the validation set"""
    # Ensure the model is in eval mode (this disables dropout/batchnorm etc.)
    network.eval()
    val_loss = 0
    val_correct = 0

    # Set all requires_grad() flags to false
    with torch.no_grad():
        # Loop through our validation data, generate predictions and
        # add to the the loss/accuracy count for each image
        for images, labels in validation_data:
            X, y = images.to(device), labels.to(device)  # to device

            # Get predictions
            preds = network(X)
            # Calculate Loss
            loss = torch.nn.functional.cross_entropy(preds, y)

            val_correct += get_num_correct(preds, y)
            val_loss = loss.item() * batch_size

    # Print the loss and accuracy for the validation set
    print("Validation Loss: ", val_loss / batch_size)
    print("Validation Acc:  ", (val_correct / validation_set_size) * 100)

    # Return loss and accuracy values
    return val_loss, ((val_correct / validation_set_size) * 100)


def confusion_matrix(num_classes, validation_labels, predictions, save_path):
    """Generate and save a confusion matrix for the Neural Network"""

    # Generate the confusion matrix
    cmt = torch.zeros(num_classes, num_classes, dtype=torch.int32)
    for i in range(len(validation_labels)):
        cmt[validation_labels[i], predictions[i]] += 1

    cmt = cmt.cpu().detach().numpy()

    df_cm = pd.DataFrame(
        cmt / np.sum(cmt),
        index=[i for i in range(num_classes)],
        columns=[i for i in range(num_classes)],
    )   

    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=True, fmt=".2f", cmap="Blues")
    plt.title("Neural Network Confusion Matrix", fontsize=20)

    # Use os to ensure the save path is valid
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    print(f"Confusion matrix saved to {save_path}")


def train_model(network, device, optimizer, scheduler, training_data, batch_size):
    """Trains the model using the training data"""

    epoch_loss = 0
    epoch_correct = 0
    network.train()  # training mode

    for images, labels in training_data:
        X, y = images.to(device), labels.to(device)  # put X & y on device
        y_ = network(X)  # get predictions

        # Zeros the gradients
        optimizer.zero_grad()
        # Calculates the loss
        loss = torch.nn.functional.cross_entropy(y_, y)
        # Computes the gradients
        loss.backward()
        # Update weights
        optimizer.step()

        epoch_loss += loss.item() * batch_size
        epoch_correct += get_num_correct(y_, y)

    print("Train Loss: ", epoch_loss / batch_size)
    print("Train Acc:  ", epoch_correct / len(training_data))

    scheduler.step()
    return (
        optimizer.param_groups[0]["lr"],
        epoch_loss,
        epoch_correct / len(training_data),
    )
