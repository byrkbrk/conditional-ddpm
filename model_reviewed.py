import torch
from torch import nn


class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_res=False):
        super(ResidualConvBlock, self).__init__()
        self.is_res = is_res
        self.same_channels = in_channels == out_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        
        self.handle_channel = None
        if self.is_res and not self.same_channels:
            self.handle_channel = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        
    def forward(self, x):
        x_out = self.conv2(self.conv1(x))
        if not self.is_res:
            return x_out
        else:
            if self.handle_channel:
                x_out = x_out + self.handle_channel(x)
            else:
                x_out = x_out + x
            return x_out / 1.414 # normalize


class UNetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetDown, self).__init__()
        layers = [ResidualConvBlock(in_channels, out_channels), # no skip-connection
                  ResidualConvBlock(out_channels, out_channels),
                  nn.MaxPool2d(2)]
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2), # double h & w sizes
            ResidualConvBlock(out_channels, out_channels), # no residual-connection
            ResidualConvBlock(out_channels, out_channels)
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x, x_skip):
        return self.model(torch.cat([x, x_skip], dim=1))


class EmbedFC(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(EmbedFC, self).__init__()
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.reshape(-1, self.input_dim)
        return self.model(x)[:, :, None, None]


class ContextUnet(nn.Module):
    def __init__(self, in_channels, height, width, n_feat, n_cfeat):
        super(ContextUnet, self).__init__()
        self.init_conv = ResidualConvBlock(in_channels, n_feat, True)
        self.down1 = UNetDown(n_feat, n_feat)
        self.down2 = UNetDown(n_feat, 2*n_feat)
        self.to_vec = nn.Sequential(
            nn.AvgPool2d((height//2**2, width//2**2)), 
            nn.GELU())
        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2*n_feat, 2*n_feat, (height//2**2, width//2**2)),
            nn.GroupNorm(8, 2*n_feat),
            nn.GELU()
        )
        self.up1 = UNetUp(4*n_feat, n_feat)
        self.up2 = UNetUp(2*n_feat, n_feat)
        self.out = nn.Sequential(
            nn.Conv2d(2*n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.GELU(),
            nn.Conv2d(n_feat, in_channels, 1, 1)
        )
        
        self.timeemb1 = EmbedFC(1, 2*n_feat)
        self.timeemb2 = EmbedFC(1, n_feat)
        self.contextemb1 = EmbedFC(n_cfeat, 2*n_feat)
        self.contextemb2 = EmbedFC(n_cfeat, n_feat)

    def forward(self, x, t, c):
        x = self.init_conv(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        hidden_vec = self.to_vec(down2)
        up0 = self.up0(hidden_vec)
        up1 = self.up1(up0*self.contextemb1(c) + self.timeemb1(t), down2)
        up2 = self.up2(up1*self.contextemb2(c) + self.timeemb2(t), down1)
        return self.out(torch.cat([up2, x], axis=1))



if __name__ == "__main__":
    x = torch.randn(5, 4, 8, 8)
    res_conv_block = ResidualConvBlock(4, 8, False)
    print(x.shape)
    print(res_conv_block(x).shape)

    unet_down = UNetDown(4, 36)
    print(x.shape)
    print(unet_down(x).shape)

    x_skip = torch.randn(5, 4, 8, 8)
    unet_up = UNetUp(8, 2)
    print(x.shape)
    print(unet_up(x, x_skip).shape)

    x_emb = torch.randn(5, 16)
    embed_fc = EmbedFC(16, 32)
    print(x_emb.shape)
    print(embed_fc(x_emb).shape)

    in_channels = 1
    height = width = 64
    n_feature = 64
    n_cfeature = 10
    context_unet = ContextUnet(in_channels, height, width, n_feature, n_cfeature)
    
    timesteps = 100
    n_samples = 7
    x = torch.randn(n_samples, in_channels, height, width)
    c = torch.rand(n_samples, n_cfeature)
    t = torch.randint(1, timesteps + 1, (n_samples, ))
    print(x.shape)
    print(context_unet(x, t/timesteps, c).shape)
            



