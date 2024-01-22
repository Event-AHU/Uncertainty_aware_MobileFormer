import torch
from torch import nn, einsum
from einops import rearrange
import pdb

# inputs: x(b c h w) z(b m d)
# output: z(b m d)
class Mobile2Former(nn.Module):
    def __init__(self, dim, heads, channel, dropout=0.):
        super(Mobile2Former, self).__init__()
        inner_dim = heads * channel
        self.heads = heads
        self.ln = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim)
        self.attend = nn.Softmax(dim=-1)
        self.scale = channel ** -0.5
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.LayerNorm(dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, z):
        b, m, d = z.shape
        b,c,t,h,w = x.shape
        # x [12, 96, 1, 14, 14]
        # z [12,6,192]
        z = self.ln(z)
        x =  x.reshape(b, c, t*h*w).transpose(1,2).unsqueeze(1)
        q = self.to_q(z).view(b, self.heads, m, c) # 12 2 6 16
        dots = q @ x.transpose(2, 3) * self.scale
        attn = self.attend(dots)
        out = attn @ x
        out = rearrange(out, 'b h m c -> b m (h c)')
        return z + self.to_out(out)


# inputs: x(b c h w) z(b m d)
# output: x(b c h w)
class Former2Mobile(nn.Module):
    def __init__(self, dim, heads, channel, dropout=0.):
        super(Former2Mobile, self).__init__()
        inner_dim = heads * channel # 2x24
        self.heads = heads
        self.to_k = nn.Linear(dim, inner_dim)
        self.to_v = nn.Linear(dim, inner_dim)
        self.attend = nn.Softmax(dim=-1)
        self.scale = channel ** -0.5

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, channel),
            nn.LayerNorm(channel),
            nn.Dropout(dropout)
        )

    def forward(self, x, z):
        b, m, d = z.shape
        b,c,t,h, w = x.shape
        q =  x.reshape(b, c, t*h*w).transpose(1,2).unsqueeze(1) #[8, 1, 6272, 24]
        k = self.to_k(z).view(b, self.heads, m, c) # [8, 2, 6, 24]
        v = self.to_v(z).view(b, self.heads, m, c)
        dots = q @ k.transpose(2, 3) * self.scale
        attn = self.attend(dots)
        out = attn @ v
        out = rearrange(out, 'b h l c -> b l (h c)') #[8, 6272, 48]
        out = self.to_out(out) #[8, 6272, 24]
        out = out.view(b, c, t, h, w)
        return x + out
