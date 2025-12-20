from tinygrad import Tensor, nn


class ResidualDenseBlock:
    def __init__(self, num_feat=64, num_grow_ch=32):
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)

    def __call__(self, x):
        x1 = self.conv1(x).leaky_relu(0.2)
        x2 = self.conv2(x.cat(x1, dim=1)).leaky_relu(0.2)
        x3 = self.conv3(x.cat(x1, x2, dim=1)).leaky_relu(0.2)
        x4 = self.conv4(x.cat(x1, x2, x3, dim=1)).leaky_relu(0.2)
        x5 = self.conv5(x.cat(x1, x2, x3, x4, dim=1))
        return x5 * 0.2 + x


class RRDB:
    def __init__(self, num_feat, num_grow_ch=32):
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def __call__(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet:
    def __init__(self, num_in_ch, num_out_ch, num_feat=64, num_block=23, num_grow_ch=32):
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)

        # self.body = make_layer(RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch)
        self.body = []
        for _ in range(num_block):
            self.body.append(RRDB(num_feat=num_feat, num_grow_ch=num_grow_ch))
            
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

    def __call__(self, x):
        feat = self.conv_first(x)

        # body_feat = self.conv_body(self.body(feat))
        x = feat
        for layer in self.body:
            x = layer(x)

        body_feat = self.conv_body(x)
        feat = feat + body_feat

        # feat = self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')).leaky_relu(0.2)
        # feat = self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')).leaky_relu(0.2)

        _, _, h, w = feat.shape
        feat = self.conv_up1(feat.interpolate(size=(h * 2, w * 2), mode='nearest')).leaky_relu(0.2)

        _, _, h, w = feat.shape
        feat = self.conv_up2(feat.interpolate(size=(h * 2, w * 2), mode='nearest')).leaky_relu(0.2)

        out = self.conv_last(self.conv_hr(feat).leaky_relu(0.2))
        return out
