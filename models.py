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

        self.cnn1 = CNNBlock(1, 16, conv_channel=8, dropout=dropout)
        self.cnn2 = CNNBlock(16, 32, conv_channel=16, dropout=dropout)
        self.cnn3 = CNNBlock(32, 64, conv_channel=32, dropout=dropout)
        self.cnn4 = CNNBlock(64, 128, conv_channel=64, dropout=dropout)
        self.cnn5 = CNNBlock(128, 128, conv_channel=64, dropout=dropout)
        self.cnn6 = CNNBlock(128, 128, conv_channel=64, maxpool_kernel=(2, 2), stride=2, dropout=dropout)

        # 1408/1152
        self.flatten = torch.nn.Flatten(start_dim=2)
        self.rnn = torch.nn.GRU(2688, 1344, batch_first=True, bidirectional=True)

        self.fc = torch.nn.Linear(2688 * 2, num_classes)

        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim=-1)


    def forward(self, x):
        x = torch.unsqueeze(x, 1)

        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        x = self.cnn4(x)
        x = self.cnn5(x)
        x = self.cnn6(x)

        # bring time axis to the front and flatten last two dimensions
        x = torch.transpose(x, 1, 2)
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
        self.conv = torch.nn.Conv2d(in_channels, conv_channel, kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn1 = torch.nn.BatchNorm2d(conv_channel)
        self.relu = torch.nn.ReLU(inplace=True)
        self.shake = ShakeShakeBlock(conv_channel, out_channels)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)
        self.avgpool = torch.nn.AvgPool2d(maxpool_kernel, maxpool_stride)


    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.dropout(self.shake(x))
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.avgpool(x)

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


class AttentionRCNN(torch.nn.Module):

    def __init__(self, num_classes, dropout=0):
        super(AttentionRCNN, self).__init__()
        self.dropout = torch.nn.Dropout(dropout)

        self.cnn1 = ShakeGluCNNBlock(1, 16, dropout=dropout)
        self.cnn2 = ShakeGluCNNBlock(16, 32, dropout=dropout)
        self.cnn3 = ShakeGluCNNBlock(32, 64, dropout=dropout)
        self.cnn4 = ShakeGluCNNBlock(64, 128, dropout=dropout)
        self.cnn5 = ShakeGluCNNBlock(128, 128, dropout=dropout)
        self.cnn6 = ShakeGluCNNBlock(128, 128, maxpool_kernel=(2, 2), stride=2, dropout=dropout)

        self.flatten = torch.nn.Flatten(start_dim=2)
        self.rnn1 = torch.nn.GRU(3712, 1024, batch_first=True, bidirectional=True)
        self.layernorm1 = torch.nn.LayerNorm(2048)
        self.rnn2 = torch.nn.GRU(2048, 1024, batch_first=True, bidirectional=True)
        self.layernorm2 = torch.nn.LayerNorm(2048)

        # attention part
        self.embed = torch.nn.Linear(2048, 256)
        self.layernorm3 = torch.nn.LayerNorm(256)
        self.attention = torch.nn.MultiheadAttention(embed_dim=256, num_heads=4, dropout=dropout)

        self.fc1 = torch.nn.Linear(256, num_classes)

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
        x = self.cnn6(x)

        # bring time axis to the front and flatten last two dimensions
        x = torch.transpose(x, 1, 2)
        x = self.flatten(x)

        x, _ = self.rnn1(x)
        x = self.layernorm1(x)
        x = self.relu(x)
        x, _ = self.rnn2(x)
        x = self.layernorm2(x)

        x = self.embed(x)
        residual = x

        x = self.layernorm3(x)
        x, _ = self.attention(x, x, x)

        x = x + residual

        x = self.fc1(x)

        return x

    def get_weak_logits(self, x):
        sig_x = self.sigmoid(x)
        soft_x = self.softmax(x)

        x = soft_x * sig_x

        x = torch.mean(x, dim=1)

        return x


class ShakeGluCNNBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, maxpool_kernel=(1, 2),
                 maxpool_stride=1, dropout=0):
        super(ShakeGluCNNBlock, self).__init__()
        self.dropout = torch.nn.Dropout(dropout)

        # only works with stride = 1
        padding = kernel_size // 2
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.glu = torch.nn.GLU(dim=1)
        self.shake = ShakeShakeBlock(int(.5*out_channels), out_channels)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU(inplace=True)
        self.avgpool = torch.nn.AvgPool2d(maxpool_kernel, maxpool_stride)

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


class SingleLabelRCNN(torch.nn.Module):

    def __init__(self, num_classes, dropout=0):
        super(SingleLabelRCNN, self).__init__()
        self.dropout = torch.nn.Dropout(dropout)

        self.cnn1 = CNNBlock(1, 16, conv_channel=8, dropout=dropout)
        self.cnn2 = CNNBlock(16, 32, conv_channel=16, dropout=dropout)
        self.cnn3 = CNNBlock(32, 64, conv_channel=32, dropout=dropout)
        self.cnn4 = CNNBlock(64, 128, conv_channel=64, dropout=dropout)
        self.cnn5 = CNNBlock(128, 128, conv_channel=64, dropout=dropout)
        self.cnn6 = CNNBlock(128, 128, conv_channel=64, maxpool_kernel=(2, 2), stride=2, dropout=dropout)

        # 1408/1152
        self.flatten = torch.nn.Flatten(start_dim=2)
        self.rnn = torch.nn.GRU(2688, 1344, batch_first=True, bidirectional=True)

        self.sed_fc1 = torch.nn.Linear(2688 * 2, 100)
        self.sed_fc2 = torch.nn.Linear(100, 1)

        self.classification_fc1 = torch.nn.Linear(2688 * 2, 50)
        self.classification_flatten = torch.nn.Flatten(start_dim=1)
        self.classification_fc2 = torch.nn.Linear(50 * 250, num_classes)

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
        x = self.cnn6(x)

        # bring time axis to the front and flatten last two dimensions
        x = torch.transpose(x, 1, 2)
        x = self.flatten(x)

        residual = x
        x, _ = self.rnn(x)
        x = self.dropout(x)

        x = torch.cat([x, residual], dim=-1)

        sed_x = self.relu(self.sed_fc1(x))
        sed_x = torch.squeeze(self.sed_fc2(sed_x))

        cls_x = self.relu(self.classification_fc1(x))
        cls_x = self.classification_flatten(cls_x)
        cls_x = self.classification_fc2(cls_x)

        return sed_x, cls_x

    def get_weak_logits(self, x):
        sig_x = self.sigmoid(x)
        soft_x = self.softmax(x)

        x = soft_x * sig_x

        x = torch.mean(x, dim=1)

        return x
