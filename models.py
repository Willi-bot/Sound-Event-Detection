import torch
from transformers import ASTFeatureExtractor, ASTModel


device = torch.device("cuda" if torch.cuda.is_available() else "")

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


# from DCASE2019_Yan_54
class ShakeShakeBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(ShakeShakeBlock, self).__init__()
        self.branch1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.branch2 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.sigmoid = torch.nn.Sigmoid()

        # in case of using a bottleneck or an inverted bottleneck reduce/expand dims of input
        if stride != 1 or in_channels != out_channels:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = torch.nn.Identity()

    def forward(self, x):
        if self.training:
            a = torch.tensor(0.5).detach().to(device)
        else:
            a = torch.rand(1).detach().to(device)

        x1 = self.branch1(x)
        x2 = self.branch2(x)

        # a = a.expand_as(x1)
        out = self.sigmoid(a * x1 + (1. - a) * x2)

        x = self.shortcut(x)
        return x * out


class ShakeRCNN(torch.nn.Module):

    def __init__(self, num_classes, dropout=0):
        super(ShakeRCNN, self).__init__()
        self.dropout = torch.nn.Dropout(dropout)

        self.cnn1 = CNNBlock(1, 8, dropout=dropout, kernel_size=(7, 7), avgpool_kernel=(1, 2), avgpool_stride=(1, 2))
        self.cnn2 = CNNBlock(8, 16, dropout=dropout, kernel_size=(5, 5), avgpool_kernel=(1, 2), avgpool_stride=(1, 2))
        self.cnn3 = CNNBlock(16, 32, dropout=dropout, kernel_size=(5, 5), avgpool_kernel=(2, 2), avgpool_stride=(2, 2))
        self.cnn4 = CNNBlock(32, 64, dropout=dropout, kernel_size=(3, 3), avgpool_kernel=(1, 2), avgpool_stride=(1, 2))
        self.cnn5 = CNNBlock(64, 128, dropout=dropout, kernel_size=(3, 3), avgpool_kernel=(2, 2), avgpool_stride=(2, 2))

        self.flatten = torch.nn.Flatten(start_dim=2)
        self.rnn1 = torch.nn.GRU(3072, 1024, batch_first=True, bidirectional=True)
        self.layernorm1 = torch.nn.LayerNorm(2048)
        self.rnn2 = torch.nn.GRU(2048, 1024, batch_first=True, bidirectional=True)
        self.layernorm2 = torch.nn.LayerNorm(2048)

        self.fc1 = torch.nn.Linear(2048, num_classes)

        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim=-1)
        self.relu = torch.nn.ReLU()


    def forward(self, x):

        x = torch.unsqueeze(x, 1)

        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        x = self.cnn4(x)
        x = self.cnn5(x)

        # bring time axis to the front and flatten last two dimensions
        x = torch.transpose(x, 1, 2)
        x = self.flatten(x)

        x, _ = self.rnn1(x)
        x = self.layernorm1(x)
        x = self.relu(x)
        x, _ = self.rnn2(x)
        x = self.layernorm2(x)

        x = self.fc1(x)

        return x

    def get_weak_logits(self, x):
        sig_x = self.sigmoid(x)
        soft_x = self.softmax(x)

        x = soft_x * sig_x

        x = torch.mean(x, dim=1)

        return x


class CNNBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, avgpool_kernel=(1, 2),
                 avgpool_stride=1, dropout=0):
        super(CNNBlock, self).__init__()
        self.dropout = torch.nn.Dropout(dropout)

        self.conv = torch.nn.Conv2d(in_channels, 2*out_channels, kernel_size=kernel_size, padding='same', stride=stride)
        self.bn1 = torch.nn.BatchNorm2d(2*out_channels)
        self.glu = torch.nn.GLU(dim=1)
        self.shake = ShakeShakeBlock(out_channels, out_channels)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU(inplace=True)
        self.avgpool = torch.nn.AvgPool2d(avgpool_kernel, avgpool_stride)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.glu(x)
        x = self.dropout(x)
        x = self.dropout(self.shake(x))
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.avgpool(x)

        return x


class ASTRCNN(torch.nn.Module):

    def __init__(self, num_classes):
        super(ASTRCNN, self).__init__()

        self.cnn1 = CNNBlock(1, 16, avgpool_kernel=(2, 2), avgpool_stride=2)
        self.cnn2 = CNNBlock(16, 32, avgpool_kernel=(2, 2), avgpool_stride=2)
        self.cnn3 = CNNBlock(64, 64)

        self.flatten = torch.nn.Flatten(start_dim=2)
        self.rnn1 = torch.nn.GRU(6144, 1024, batch_first=True, bidirectional=True)
        self.layernorm1 = torch.nn.LayerNorm(2048)

        self.fc1 = torch.nn.Linear(2048, num_classes)

    def forward(self, x):

        x = torch.unsqueeze(x, 1)
        x = self.cnn1(x)
        x = self.cnn2(x)

        # bring time axis to the front and flatten last two dimensions
        x = torch.transpose(x, 1, 2)
        x = self.flatten(x)
        x, _ = self.rnn1(x)
        x = self.layernorm1(x)

        x = self.fc1(x)

        return x