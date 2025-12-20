from tinygrad import nn, Tensor


def pixel_shuffle(x: Tensor, upscale_factor: int) -> Tensor:
    b, c, h, w = x.shape
    
    # Ensure channels are divisible by upscale_factor squared
    if c % (upscale_factor * upscale_factor) != 0:
        raise ValueError(f"Input channels {c} must be divisible by {upscale_factor**2}")
    
    c_out = c // (upscale_factor * upscale_factor)
    
    # 1. Reshape to split channels into (c_out, r, r)
    x = x.reshape(b, c_out, upscale_factor, upscale_factor, h, w)
    
    # 2. Permute to move the 'r' dimensions next to spatial dimensions (h, w)
    #    From: (B, C_out, r_h, r_w, H, W) 
    #    To:   (B, C_out, H, r_h, W, r_w)
    x = x.permute(0, 1, 4, 2, 5, 3)
    
    # 3. Reshape to combine (H, r_h) -> H_new and (W, r_w) -> W_new
    return x.reshape(b, c_out, h * upscale_factor, w * upscale_factor)


class PReLU:
    def __init__(self, num_parameters):
        self.weight = Tensor.zeros(num_parameters)

    def __call__(self, x):
        return (x > 0).where(x, x * self.weight.reshape(1, -1, 1, 1))


class SRVGGNetCompact:
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu'):
        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        self.num_feat = num_feat
        self.num_conv = num_conv
        self.upscale = upscale
        self.act_type = act_type

        # self.body = nn.ModuleList()
        self.body = []
        # the first conv
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        # the first activation
        self.body.append(PReLU(num_parameters=num_feat))

        # the body structure
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            self.body.append(PReLU(num_parameters=num_feat))

        # the last conv
        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        # upsample
        # self.upsampler = nn.PixelShuffle(upscale)

    def __call__(self, x):
        out = x
        for i in range(0, len(self.body)):
            out = self.body[i](out)

        # out = self.upsampler(out)
        out = pixel_shuffle(out, self.upscale)
        # # add the nearest upsampled image, so that the network learns the residual
        # base = F.interpolate(x, scale_factor=self.upscale, mode='nearest')
        _, _, h, w = x.shape
        base = x.interpolate(size=(h * self.upscale, w * self.upscale), mode='nearest')
        out += base
        return out


# m = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
# print(m(Tensor.ones(1, 3, 32, 32)).numpy())
