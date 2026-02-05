import torch

class YOLO(torch.nn.Module):
    def __init__(self, bbox_count = 2, num_classes = 20, grid_length = 7):
        super(YOLO, self).__init__()
        self.s = grid_length
        # self.batch  = batch
        self.num_classes = num_classes
        self.bbox_count = bbox_count

        self.layer1 = torch.nn.Sequential(
            conv_lrelu(3, 64, 7, 2, 3, bias = False, alpha = 0.1),
            torch.nn.MaxPool2d(kernel_size = 2, stride = 2)
        )
        # 112
        self.layer2 = torch.nn.Sequential(
            conv_lrelu(64, 192, kernel_size = 3, stride = 1, padding = 1),
            torch.nn.MaxPool2d(kernel_size = 2, stride = 2)
        )
        # 56
        self.layer3 = torch.nn.Sequential(
            Module(192, 256),
            Module(256, 512),
            torch.nn.MaxPool2d(kernel_size = 2, stride = 2)
        )

        self.layer4 = torch.nn.Sequential(
            Module(512, 512),
            Module(512, 512),
            Module(512, 512),
            Module(512, 512),
            Module(512, 1024),
            torch.nn.MaxPool2d(kernel_size = 2, stride = 2)
        )

        self.layer5 = torch.nn.Sequential(
            Module(1024, 1024),
            Module(1024, 1024),
            conv_lrelu(1024, 1024, kernel_size = 3, stride = 1, padding = 1, bias = False),
            conv_lrelu(1024, 1024, kernel_size = 3, stride = 2,padding = 1, bias = False)
        )

        self.layer6 = torch.nn.Sequential(
            conv_lrelu(1024, 1024),
            conv_lrelu(1024, 1024)
        )

        self.conn_layer = torch.nn.Sequential(
            torch.nn.Linear(in_features = self.s * self.s * 1024, out_features = 4096),
            torch.nn.LeakyReLU(negative_slope = 0.1),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(4096, out_features = self.s * self.s * (bbox_count * 5 + num_classes))
        )

    def forward(self, x):
        batch = x.shape[0]
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = torch.flatten(x, 1)
        x = self.conn_layer(x)
        x = torch.reshape(x, (batch, self.bbox_count * 5 + self.num_classes, self.s, self.s))
        x = torch.permute(x, (0, 2, 3, 1))
        return x

class conv_lrelu(torch.nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size = 3, stride = 1, padding = 1, bias = False, alpha = 0.1):
        super(conv_lrelu, self).__init__()
        self.conv = torch.nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding,
            bias = bias
        )
        # self.bn = torch.nn.BatchNorm2d(out_ch)
        self.lrelu = torch.nn.LeakyReLU(negative_slope = alpha)

    def forward(self, x):
        x = self.conv(x)
        # x = self.bn(x)
        x = self.lrelu(x)
        return x

class Module(torch.nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Module, self).__init__()
        self.conv = torch.nn.Sequential(
            conv_lrelu(in_ch, out_ch // 2, kernel_size = 1, stride = 1, padding = 0, bias = False),
            conv_lrelu(out_ch // 2, out_ch, kernel_size = 3, stride = 1, padding = 1, bias = False)
        )

    def forward(self, x):
        return self.conv(x)
