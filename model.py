import torch
import torch.nn as nn

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class SimplifiedChannelAttention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(c, c, 1, 1, 0, bias=True)

    def forward(self, x):
        return x * self.conv(self.avg_pool(x))

class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2):
        super().__init__()
        dw_channel = c * DW_Expand
        
        self.conv1 = nn.Conv2d(c, dw_channel, 1, 1, 0, bias=True)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, 3, 1, 1, groups=dw_channel, bias=True)
        self.sg = SimpleGate()
        self.sca = SimplifiedChannelAttention(dw_channel // 2)
        self.conv3 = nn.Conv2d(dw_channel // 2, c, 1, 1, 0, bias=True)
        
        self.norm1 = nn.LayerNorm(c, eps=1e-5)
        self.norm2 = nn.LayerNorm(c, eps=1e-5)
        
        ffn_channel = c * FFN_Expand
        self.conv4 = nn.Conv2d(c, ffn_channel, 1, 1, 0, bias=True)
        self.sg2 = SimpleGate()
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, 1, 1, 0, bias=True)

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, x):
        B, C, H, W = x.shape
        # LayerNorm expects channel last
        xn = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        x1 = self.conv1(xn)
        x1 = self.conv2(x1)
        x1 = self.sg(x1)
        x1 = self.sca(x1)
        x1 = self.conv3(x1)
        x = x + x1 * self.beta
        
        xn2 = self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x2 = self.conv4(xn2)
        x2 = self.sg2(x2)
        x2 = self.conv5(x2)
        x = x + x2 * self.gamma
        return x

class NAFNet(nn.Module):
    def __init__(self, in_channels=1, width=32, enc_blk_nums=[1, 1, 1, 1], dec_blk_nums=[1, 1, 1, 1]):
        super().__init__()
        self.intro = nn.Conv2d(in_channels, width, 3, padding=1, bias=True)
        
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        
        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, chan * 2, 2, stride=2))
            chan *= 2
            
        self.middle_blks = nn.Sequential(*[NAFBlock(chan) for _ in range(1)])
        
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        
        for num in dec_blk_nums:
            self.ups.append(nn.Sequential(
                nn.Conv2d(chan, chan * 2, 1, bias=False),
                nn.PixelShuffle(2)
            ))
            chan //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            
        self.ending = nn.Conv2d(width, width, 3, padding=1, bias=True)
        
    def forward(self, x):
        x = self.intro(x)
        encs = []
        
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)
            
        x = self.middle_blks(x)
        
        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)
            
        x = self.ending(x)
        return x

class NAFNetSR(nn.Module):
    """
    Wraps NAFNet to output a 2x super-resolved image.
    Uses NAFNet for deep feature extraction at the input resolution,
    followed by a PixelShuffle layer for 2x upsampling.
    """
    def __init__(self, channels=1, width=32, scale=2,
                 enc_blk_nums=None, dec_blk_nums=None, hr_blocks=2):
        super().__init__()
        if enc_blk_nums is None:
            enc_blk_nums = [1, 1, 1, 1]
        if dec_blk_nums is None:
            dec_blk_nums = [1, 1, 1, 1]
        self.scale = scale
        # Base NAFNet feature extractor (maintains input resolution)
        self.feature_extractor = NAFNet(
            in_channels=channels,
            width=width,
            enc_blk_nums=enc_blk_nums,
            dec_blk_nums=dec_blk_nums,
        )
        
        # Upsampling module for Super Resolution
        self.upconv1 = nn.Conv2d(width, width * (scale ** 2), 3, 1, 1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(scale)
        # HR-resolution processing. Previously a single 3x3 conv had to
        # synthesize ALL high-frequency detail, which is why this model
        # scored below bicubic on clean input.
        self.hr_blocks = nn.Sequential(
            *[NAFBlock(width) for _ in range(hr_blocks)]) if hr_blocks > 0 \
            else nn.Identity()
        self.upconv2 = nn.Conv2d(width, channels, 3, 1, 1, bias=True)
        
        # Global bicubic skip connection (to ease learning)
        self.upsample = nn.Upsample(scale_factor=scale, mode='bicubic', align_corners=False)

    def forward(self, x):
        # Base image upscaled
        base = self.upsample(x)
        
        # Extract features
        feat = self.feature_extractor(x)
        
        # Upsample features
        residual = self.upconv1(feat)
        residual = self.pixel_shuffle(residual)
        residual = self.hr_blocks(residual)
        residual = self.upconv2(residual)
        
        return base + residual

# quick smoke test: run `python model.py`
if __name__ == "__main__":
    x = torch.randn(2, 1, 128, 128) # Batch 2, Grayscale, 128x128
    model = NAFNetSR(channels=1, width=32, scale=2)
    y = model(x)
    print("Input shape:", x.shape)
    print("Output shape:", y.shape) # Should be [2, 1, 256, 256]
    assert y.shape == (2, 1, 256, 256), "Output shape does not match 2x SR expectation"
