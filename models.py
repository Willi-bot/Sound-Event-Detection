import torch


device = torch.device("cuda" if torch.cuda.is_available() else "")

class PrelimModel(torch.nn.Module):
    def __init__(self, input_size, output_size, output_shape):
        super(PrelimModel, self).__init__()
        self.flatten = torch.nn.Flatten(start_dim=1)
        self.relu = torch.nn.ReLU()
        self.fc1 = torch.nn.Linear(input_size, 5096)
        self.fc2 = torch.nn.Linear(5096, 2048)
        self.fc3 = torch.nn.Linear(2048, output_size)
        self.output_shape = output_shape

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x.float()))
        x = self.relu(self.fc2(x.float()))
        x = self.relu(self.fc3(x.float()))
        x = torch.reshape(x, self.output_shape)
        return x

class BasicRCNN(torch.nn.Module):
    def __init__(self, num_classes, frequency_dim, conv_kernels=None, conv_channels=None, maxpool_kernels=None,
                 gru_hidden_channels=32, bidirectional_gru=False, dropout=0.):
        super(BasicRCNN, self).__init__()
        if conv_kernels is None:
            conv_kernels = [3, 3, 3]
        if conv_channels is None:
            conv_channels = [128, 128, 128]
        if maxpool_kernels is None:
            maxpool_kernels = [5, 5, 2]

        self.flatten = torch.nn.Flatten(start_dim=2)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout)
        self.num_classes = num_classes
        self.frequency_dim = frequency_dim

        # stage 1
        conv1, conv2, conv3 = conv_channels
        max1, max2, max3 = maxpool_kernels
        kernel1, kernel2, kernel3 = conv_kernels
        pad1, pad2, pad3 = kernel1 // 2, kernel2 // 2, kernel3 // 2
        self.conv1 = torch.nn.Conv2d(1, conv1, kernel1, padding=pad1)
        self.maxpool1 = torch.nn.MaxPool2d((1, max1), 1)
        self.conv2 = torch.nn.Conv2d(conv1, conv2, kernel2, padding=pad2)
        self.maxpool2 = torch.nn.MaxPool2d((1, max2), 1)
        self.conv3 = torch.nn.Conv2d(conv2, conv3, kernel3, padding=pad3)
        self.maxpool3 = torch.nn.MaxPool2d((max3, max3), max3)

        # stage 2
        self.gru_input_dim = int(((self.frequency_dim - (max1 + max2 - 2)) // max3) * conv3)
        self.gru = torch.nn.GRU(self.gru_input_dim, gru_hidden_channels, batch_first=True, bidirectional=bidirectional_gru)
        if bidirectional_gru:
            fc_input_dim = gru_hidden_channels * 2
        else:
            fc_input_dim = gru_hidden_channels
        self.fc1 = torch.nn.Linear(fc_input_dim, num_classes)

    def forward(self, x):
        shape = x.shape
        x = torch.reshape(x, (shape[0],  1, shape[1], shape[2]))
        x = self.dropout(self.conv1(x.float()))
        x = self.maxpool1(x)
        x = self.relu(x)
        x = self.dropout(self.conv2(x))
        x = self.maxpool2(x)
        x = self.relu(x)
        x = self.dropout(self.conv3(x))
        x = self.maxpool3(x)
        x = self.relu(x)

        # bring time axis to the front
        shape = x.shape
        x = torch.reshape(x, (shape[0], shape[2], shape[1], shape[3]))
        x = self.flatten(x)
        x, _ = self.gru(x)
        x = self.relu(x)

        x = self.fc1(x)
        return x


class AdvancedRCNN(torch.nn.Module):

    def __init__(self, num_classes, dropout=0):
        super(AdvancedRCNN, self).__init__()
        self.dropout = torch.nn.Dropout(dropout)

        self.cnn1 = CNNBlock(1, 128, dropout=dropout)
        self.cnn2 = CNNBlock(128, 128, dropout=dropout)
        self.cnn3 = CNNBlock(128, 128, dropout=dropout)
        self.cnn4 = CNNBlock(128, 128, dropout=dropout)
        self.cnn5 = CNNBlock(128, 128)
        self.cnn6 = CNNBlock(128, 128, maxpool_kernel=(2, 2), stride=2, dropout=dropout)

        # 1408/1152
        self.flatten = torch.nn.Flatten(start_dim=2)
        self.rnn = torch.nn.GRU(1152, 1152, batch_first=True, bidirectional=True)

        self.fc = torch.nn.Linear(1152 * 3, num_classes)

        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim=-1)


    def forward(self, x):
        shape = x.shape
        x = torch.reshape(x, (shape[0], 1, shape[1], shape[2]))

        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        x = self.cnn4(x)
        x = self.cnn5(x)
        x = self.cnn6(x)

        # bring time axis to the front and flatten last two dimensions
        shape = x.shape
        x = torch.reshape(x, (shape[0], shape[2], shape[1], shape[3]))
        x = self.flatten(x)

        residual = x
        x, _ = self.rnn(x)
        x = self.dropout(x)

        x = torch.cat([x, residual], dim=-1)

        x = self.fc(x)

        return x

    def get_weak_logits(self, x):
        sig_x = self.sigmoid(x)
        soft_x = self.softmax(x)

        x = soft_x * sig_x

        x = torch.mean(x, dim=1)

        return x


# from DCASE2019_Yan_54
class CNNBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, conv_channel=128, kernel_size=3, stride=1, maxpool_kernel=(1, 5),
                 maxpool_stride=1, dropout=0):
        super(CNNBlock, self).__init__()
        self.dropout = torch.nn.Dropout(dropout)

        # only works with stride = 1
        padding = kernel_size // 2
        self.bn = torch.nn.BatchNorm2d(in_channels)
        self.conv = torch.nn.Conv2d(in_channels, conv_channel, kernel_size=kernel_size, padding=padding, stride=stride)
        self.shake = ShakeShakeBlock(conv_channel, out_channels)
        self.avgpool = torch.nn.AvgPool2d(maxpool_kernel, maxpool_stride)


    def forward(self, x):
        x = self.bn(x)
        x = self.conv(x)
        x = self.dropout(self.shake(x))
        x = self.avgpool(x)

        return x


# from DCASE2019_Yan_54
class ShakeShakeBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(ShakeShakeBlock, self).__init__()
        self.branch1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.branch2 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        if self.training:
            a = torch.tensor(0.5).detach().to(device)
        else:
            a = torch.rand(1).detach().to(device)

        x1 = self.branch1(x)
        x2 = self.branch2(x)

        # a = a.expand_as(x1)
        out = self.sigmoid(a * x1 + (1. - a) * x2)

        return x * out
