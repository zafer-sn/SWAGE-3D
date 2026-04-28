import torch
import torchvision.models as models
import torchvision
import utils
from torch.nn.utils import spectral_norm  # Spectral normalizasyon için import ekledim

# Attention modülü tanımlıyoruz
class SelfAttention(torch.nn.Module):
    def __init__(self, in_channels, use_spectral_norm=False):
        super(SelfAttention, self).__init__()
        self.in_channels = in_channels
        self.query_conv = torch.nn.Conv3d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = torch.nn.Conv3d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = torch.nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.gamma = torch.nn.Parameter(torch.zeros(1))  # Attention ağırlığı, eğitim ile öğrenilecek
        
        # Spectral normalizasyon opsiyonel olarak uygulanabilir
        if use_spectral_norm:
            self.query_conv = spectral_norm(self.query_conv)
            self.key_conv = spectral_norm(self.key_conv)
            self.value_conv = spectral_norm(self.value_conv)
            
    def forward(self, x):
        batch_size, C, D, H, W = x.size()
        
        # Projeksiyon işlemleri
        proj_query = self.query_conv(x).view(batch_size, -1, D * H * W).permute(0, 2, 1)  # B x (D*H*W) x C'
        proj_key = self.key_conv(x).view(batch_size, -1, D * H * W)  # B x C' x (D*H*W)
        
        # Attention map hesaplama
        energy = torch.bmm(proj_query, proj_key)  # Batch matrix multiplication, B x (D*H*W) x (D*H*W)
        attention = torch.nn.functional.softmax(energy, dim=2)
        
        # Value projeksiyon
        proj_value = self.value_conv(x).view(batch_size, -1, D * H * W)  # B x C x (D*H*W)
        
        # Output hesaplama
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, D, H, W)
        
        # Residual bağlantı ve gamma parametresi ile ağırlıklandırma
        out = self.gamma * out + x
        
        return out

class _G(torch.nn.Module):
    def __init__(self, args):
        super(_G, self).__init__()
        self.args = args
        self.cube_len = args.cube_len
        self.use_attention = args.use_attention if hasattr(args, 'use_attention') else True
        self.attention_after = args.attention_after if hasattr(args, 'attention_after') else 3  # Hangi katmandan sonra attention uygulanacak

        padd = (0, 0, 0)
        if self.cube_len == 32:
            padd = (1,1,1)

        self.layer1 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(self.args.z_size, self.cube_len*8, kernel_size=4, stride=2, bias=args.bias, padding=padd),
            torch.nn.BatchNorm3d(self.cube_len*8),
            torch.nn.ReLU()
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(self.cube_len*8, self.cube_len*4, kernel_size=4, stride=2, bias=args.bias, padding=(1, 1, 1)),
            torch.nn.BatchNorm3d(self.cube_len*4),
            torch.nn.ReLU()
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(self.cube_len*4, self.cube_len*2, kernel_size=4, stride=2, bias=args.bias, padding=(1, 1, 1)),
            torch.nn.BatchNorm3d(self.cube_len*2),
            torch.nn.ReLU()
        )
        
        # Attention katmanı ekleme
        if self.use_attention:
            self.attention = SelfAttention(self.cube_len*2, use_spectral_norm=args.use_spectral_norm if hasattr(args, 'use_spectral_norm') else False)
        
        self.layer4 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(self.cube_len*2, self.cube_len, kernel_size=4, stride=2, bias=args.bias, padding=(1, 1, 1)),
            torch.nn.BatchNorm3d(self.cube_len),
            torch.nn.ReLU()
        )
        self.layer5 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(self.cube_len, 1, kernel_size=4, stride=2, bias=args.bias, padding=(1, 1, 1)),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        out = x.view(-1, self.args.z_size, 1, 1, 1)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        
        # Attention uygulaması
        if self.use_attention:
            out = self.attention(out)
            
        out = self.layer4(out)
        out = self.layer5(out)

        return out


class _D(torch.nn.Module):
    def __init__(self, args):
        super(_D, self).__init__()
        self.args = args
        self.cube_len = args.cube_len
        self.is_wasserstein = args.wasserstein
        self.use_spectral_norm = args.use_spectral_norm  # Yeni parametre

        padd = (0,0,0)
        if self.cube_len == 32:
            padd = (1,1,1)

        # Spectral Normalization kullanılıyorsa
        if self.use_spectral_norm:
            # BatchNorm kullanmadan, spectral norm ile
            self.layer1 = torch.nn.Sequential(
                spectral_norm(torch.nn.Conv3d(1, self.cube_len, kernel_size=4, stride=2, bias=args.bias, padding=(1, 1, 1))),
                torch.nn.LeakyReLU(self.args.leak_value)
            )
            self.layer2 = torch.nn.Sequential(
                spectral_norm(torch.nn.Conv3d(self.cube_len, self.cube_len*2, kernel_size=4, stride=2, bias=args.bias, padding=(1, 1, 1))),
                torch.nn.LeakyReLU(self.args.leak_value)
            )
            self.layer3 = torch.nn.Sequential(
                spectral_norm(torch.nn.Conv3d(self.cube_len*2, self.cube_len*4, kernel_size=4, stride=2, bias=args.bias, padding=(1, 1, 1))),
                torch.nn.LeakyReLU(self.args.leak_value)
            )
            self.layer4 = torch.nn.Sequential(
                spectral_norm(torch.nn.Conv3d(self.cube_len*4, self.cube_len*8, kernel_size=4, stride=2, bias=args.bias, padding=(1, 1, 1))),
                torch.nn.LeakyReLU(self.args.leak_value)
            )
            # Son katmana da spectral normalizasyon uygula
            self.layer5 = spectral_norm(torch.nn.Conv3d(self.cube_len*8, 1, kernel_size=4, stride=2, bias=args.bias, padding=padd))
        else:
            # Orijinal model BatchNorm ile
            self.layer1 = torch.nn.Sequential(
                torch.nn.Conv3d(1, self.cube_len, kernel_size=4, stride=2, bias=args.bias, padding=(1, 1, 1)),
                torch.nn.BatchNorm3d(self.cube_len),
                torch.nn.LeakyReLU(self.args.leak_value)
            )
            self.layer2 = torch.nn.Sequential(
                torch.nn.Conv3d(self.cube_len, self.cube_len*2, kernel_size=4, stride=2, bias=args.bias, padding=(1, 1, 1)),
                torch.nn.BatchNorm3d(self.cube_len*2),
                torch.nn.LeakyReLU(self.args.leak_value)
            )
            self.layer3 = torch.nn.Sequential(
                torch.nn.Conv3d(self.cube_len*2, self.cube_len*4, kernel_size=4, stride=2, bias=args.bias, padding=(1, 1, 1)),
                torch.nn.BatchNorm3d(self.cube_len*4),
                torch.nn.LeakyReLU(self.args.leak_value)
            )
            self.layer4 = torch.nn.Sequential(
                torch.nn.Conv3d(self.cube_len*4, self.cube_len*8, kernel_size=4, stride=2, bias=args.bias, padding=(1, 1, 1)),
                torch.nn.BatchNorm3d(self.cube_len*8),
                torch.nn.LeakyReLU(self.args.leak_value)
            )
            # WGAN için son katmanda sigmoid kullanmıyoruz
            self.layer5 = torch.nn.Conv3d(self.cube_len*8, 1, kernel_size=4, stride=2, bias=args.bias, padding=padd)

    def forward(self, x):
        out = x.view(-1, 1, self.args.cube_len, self.args.cube_len, self.args.cube_len)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        
        return out


class _E(torch.nn.Module):
    def __init__(self, args):
        super(_E, self).__init__()
        self.args = args
        self.z_size = args.z_size
        self.use_depth = args.use_depth if hasattr(args, 'use_depth') else True
        
        # ResNet18 modelini yükle
        resnet = models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        
        # Giriş kanal sayısını belirle
        in_channels = 4 if self.use_depth else 3
        
        # İlk katmanı giriş kanal sayısına uygun hale getir (RGB + depth veya sadece RGB)
        self.conv1 = torch.nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Conv1 katmanının ağırlıklarını başlatma
        with torch.no_grad():
            self.conv1.weight[:, :3] = resnet.conv1.weight
            # 4. kanalı (depth) varsa, RGB kanallarının ortalamasını kullanarak başlat
            if self.use_depth:
                self.conv1.weight[:, 3] = resnet.conv1.weight.mean(dim=1, keepdim=True).squeeze(1)
        
        # ResNet18'in diğer katmanlarını kullan
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        
        # ResNet18'in son feature boyutu 512
        self.fc_mu = torch.nn.Linear(512, self.z_size)
        self.fc_var = torch.nn.Linear(512, self.z_size)
    
    def forward(self, x):
        # Beklenen kanal sayısını belirle
        expected_channels = 4 if self.use_depth else 3
        
        # Giriş boyutunu kontrol et
        if x.dim() != 4 or x.size(1) != expected_channels:
            raise ValueError(f"Beklenmeyen giriş boyutu: {x.shape}. Beklenen: [batch_size, {expected_channels}, H, W]")
        
        # Batch boyutunu dinamik olarak al
        batch_size = x.size(0)
        
        if batch_size == 0:
            # Boş batch durumunu ele al
            return torch.empty((0, self.z_size), device=x.device), torch.empty((0, self.z_size), device=x.device)
        
        # ResNet18 özellik çıkarımı
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        
        # Açık batch boyutu kullanarak düzleştir
        x = x.view(batch_size, -1)
        
        # VAE için mu ve log_var hesaplama
        z_mu = self.fc_mu(x)
        z_log_var = self.fc_var(x)
        
        return z_mu, z_log_var
    
    def reparameterize(self, mu, var):
        # Giriş boyutlarını kontrol et
        if mu.dim() != 2 or var.dim() != 2:
            raise ValueError(f"Beklenmeyen mu/var boyutu. mu: {mu.shape}, var: {var.shape}. Beklenen: [batch_size, z_size]")
        
        # Boş batch durumunu kontrol et
        batch_size = mu.size(0)
        if batch_size == 0:
            return torch.empty((0, self.z_size), device=mu.device)
        
        if self.training:
            std = var.mul(0.5).exp_()
            # Aynı boyutta random normal tensör 
            eps = utils.var_or_cuda(torch.randn_like(std))
            # Reparameterization trick
            z = eps.mul(std).add_(mu)
            return z
        else:
            return mu

