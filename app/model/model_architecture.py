import torch


class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        # Image starts as a matrix of size (1, 28, 28)

        # The size of the image after each convolution or pooling layer can be obtained by:
        # output = ((input - kernel_size + (2 * padding)) / stride) + 1

        # Convolutions and batch normalisations
        # Batch norm reduces internal covariate shift
        # Normalises the input feature (subtract batch mean, divide by batch sd)
        # This speeds up neural network training times
        self.conv1 = torch.nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2
        )  # conv1
        self.conv1_bn = torch.nn.BatchNorm2d(num_features=32)
        self.conv2 = torch.nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2
        )  # conv2
        self.conv2_bn = torch.nn.BatchNorm2d(num_features=64)
        self.conv3 = torch.nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
        )  # conv3
        self.conv3_bn = torch.nn.BatchNorm2d(num_features=128)
        self.conv4 = torch.nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=2, stride=1, padding=1
        )  # conv4
        self.conv4_bn = torch.nn.BatchNorm2d(num_features=256)

        # Fully connected linear layers and batch normalisations
        self.fc1 = torch.nn.Linear(
            in_features=256 * 4 * 4, out_features=1024
        )  # linear 1
        self.fc1_bn = torch.nn.BatchNorm1d(num_features=1024)
        self.fc2 = torch.nn.Linear(in_features=1024, out_features=512)  # linear 2
        self.fc2_bn = torch.nn.BatchNorm1d(num_features=512)
        self.fc3 = torch.nn.Linear(in_features=512, out_features=256)  # linear 3
        self.fc3_bn = torch.nn.BatchNorm1d(num_features=256)
        self.fc4 = torch.nn.Linear(in_features=256, out_features=64)  # linear 4
        self.fc4_bn = torch.nn.BatchNorm1d(num_features=64)

        # Final Layer
        self.out = torch.nn.Linear(in_features=64, out_features=10)  # output

        # Dropout
        self.dropout = torch.nn.Dropout(0.4)

    def forward(self, z):
        # Apply Relu then Max Pool function between each convolution layer
        z = torch.nn.functional.relu(self.conv1_bn(self.conv1(z)))  # (1, 28, 28)
        z = torch.nn.functional.max_pool2d(z, kernel_size=2, stride=2)  # (1, 14, 14)

        z = torch.nn.functional.relu(self.conv2_bn(self.conv2(z)))  # (1, 14, 14)
        z = torch.nn.functional.max_pool2d(z, kernel_size=2, stride=2)  # (1, 7, 7)

        z = torch.nn.functional.relu(self.conv3_bn(self.conv3(z)))  # (1, 7, 7)
        z = torch.nn.functional.max_pool2d(z, kernel_size=2, stride=1)  # (1, 6, 6)

        z = torch.nn.functional.relu(self.conv4_bn(self.conv4(z)))  # (1, 7, 7)
        z = torch.nn.functional.max_pool2d(z, kernel_size=4, stride=1)  # (1, 4, 4)

        # Apply Relu function between each fully connected layer
        # print(z.size())
        z = torch.nn.functional.relu(
            self.fc1_bn(self.fc1(z.reshape(-1, 256 * 4 * 4)))
        )  # 256 4 4
        z = self.dropout(z)

        z = torch.nn.functional.relu(self.fc2_bn(self.fc2(z)))
        z = self.dropout(z)

        z = torch.nn.functional.relu(self.fc3_bn(self.fc3(z)))
        z = self.dropout(z)

        z = torch.nn.functional.relu(self.fc4_bn(self.fc4(z)))
        z = self.out(z)

        return z
