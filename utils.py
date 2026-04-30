import scipy.ndimage as nd
import scipy.io as io
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import skimage.measure as sk
from mpl_toolkits import mplot3d
import matplotlib.gridspec as gridspec
import numpy as np
from torch.utils import data
from torch.autograd import Variable
import torch
import os
import pickle
from skimage.io import imread
import trimesh
from PIL import Image
from torchvision import transforms
import binvox_rw
import glob

def getVoxelFromMat(path, cube_len=64):
    """Mat 데이터로 부터 Voxel 을 가져오는 함수"""
    voxels = io.loadmat(path)['instance']
    voxels = np.pad(voxels, (1, 1), 'constant', constant_values=(0, 0))
    if cube_len != 32 and cube_len == 64:
        voxels = nd.zoom(voxels, (2, 2, 2), mode='constant', order=0)
    return voxels

def getVolumeFromBinvox(path):
    with open(path, 'rb') as file:
        data = np.int32(binvox_rw.read_as_3d_array(file).data)
    return data

def getVolumeFromOFF(path, sideLen=32):
    mesh = trimesh.load(path)
    volume = trimesh.voxel.VoxelMesh(mesh, 0.5).matrix
    (x, y, z) = map(float, volume.shape)
    volume = nd.zoom(volume.astype(float),
                     (sideLen/x, sideLen/y, sideLen/z),
                     order=1,
                     mode='nearest')
    volume[np.nonzero(volume)] = 1.0
    return volume.astype(np.bool)


def getVFByMarchingCubes(voxels, threshold=0.5):
    """Voxel 로 부터 Vertices, faces 리턴 하는 함수"""
    v, f = sk.marching_cubes_classic(voxels, level=threshold)
    return v, f


def plotVoxelVisdom(voxels, visdom, title):
    v, f = getVFByMarchingCubes(voxels)
    visdom.mesh(X=v, Y=f, opts=dict(opacity=0.5, title=title))


def plotFromVoxels(voxels):
    z, x, y = voxels.nonzero()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, -z, zdir='z', c='red')
    plt.savefig('test')
    plt.show()


def SavePloat_Voxels(voxels, path, iteration, titles=None):
    """
    Voxel modellerini görselleştirir ve kaydeder.
    
    Args:
        voxels: Görselleştirilecek voxel array'i [batch, x, y, z]
        path: Kaydedilecek dizin
        iteration: İterasyon numarası (dosya adı için)
        titles: Her bir voxel modeli için başlıklar (opsiyonel)
    """
    voxels = voxels.__ge__(0.5)
    fig = plt.figure(figsize=(32, 16))
    
    batch_size = voxels.shape[0]
    cols = min(4, batch_size)
    rows = (batch_size + cols - 1) // cols
    
    gs = gridspec.GridSpec(rows, cols)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(voxels):
        if i >= rows * cols:
            break
            
        ax = plt.subplot(gs[i], projection='3d')
        ax.set_aspect('equal')
        
        x, y, z = sample.nonzero()
        ax.scatter(x, y, z, zdir='z', c='red', s=20)
        
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.set_xlim(0, voxels.shape[1])
        ax.set_ylim(0, voxels.shape[2])
        ax.set_zlim(0, voxels.shape[3])
        
        # Başlık ekle (varsa)
        if titles is not None and i < len(titles):
            ax.set_title(titles[i], fontsize=18, y=1.05)
    
    # Dizin yoksa oluştur
    if not os.path.exists(path):
        os.makedirs(path)
    
    # Kaydet
    plt.savefig(path + '/{}.png'.format(str(iteration).zfill(3)), bbox_inches='tight')
    plt.close()

    # Ayrıca voxel verilerini pickle olarak kaydet
    with open(path + '/{}.pkl'.format(str(iteration).zfill(3)), "wb") as f:
        pickle.dump(voxels, f, protocol=pickle.HIGHEST_PROTOCOL)


def make_hyparam_string(hyparam_dict):
    str_result = ""
    for i in hyparam_dict.keys():
        str_result = str_result + str(i) + "=" + str(hyparam_dict[i]) + "_"
    return str_result[:-1]

class ShapeNetDataset(data.Dataset):
    """Custom Dataset compatible with torch.utils.data.DataLoader"""

    def __init__(self, root, args):
        """Set the path for Data.

        Args:
            root: image directory.
            transform: Tensor transformer.
        """
        self.root = root
        self.listdir = os.listdir(self.root)
        self.args = args

    def __getitem__(self, index):
        #with open(self.root + self.listdir[index], "rb") as f:
            #volume = np.asarray(getVoxelFromMat(f, self.args.cube_len), dtype=np.float32)
            #plotFromVoxels(volume)
        model_3d_file = [name for name in self.listdir if name.endswith('.' + "binvox")][index]
        volume = np.array(getVolumeFromBinvox(self.root + model_3d_file), dtype=np.float32)
        return torch.FloatTensor(volume)

    def __len__(self):
        return len([name for name in self.listdir if name.endswith('.' + "binvox")])

class ShapeNetPlusImageDataset(data.Dataset):
    """Custom Dataset compatible with torch.utils.data.DataLoader"""

    def __init__(self, root, args):
        """Set the path for Data.

        Args:
            root: image directory.
            transform: Tensor transformer.
        """
        self.root = root
        self.listdir = [name for name in os.listdir(self.root) if os.path.isfile(os.path.join(self.root, name))]
        self.args = args
        self.img_size = args.image_size
        
        # RGB ve depth için normalizasyon değerlerini tanımlayalım
        # RGB için ImageNet'in standart değerleri
        self.rgb_mean = [0.485, 0.456, 0.406]
        self.rgb_std = [0.229, 0.224, 0.225]
        
        # Depth için aynı değerleri kullanalım (alternatif olarak kendi veri setinize özgü değerler hesaplayabilirsiniz)
        self.depth_mean = [0.330]  # Derinlik haritaları için ortalama değer
        self.depth_std = [0.236]   # Derinlik haritaları için standart sapma
        
        # Transform işlemlerini optimize edelim
        self.rgb_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.rgb_mean, std=self.rgb_std)
        ])
        
        self.depth_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.depth_mean, std=self.depth_std)
        ])
        
        # Veri setini önceden indexleyelim (daha hızlı erişim için)
        self.model_3d_files = [name for name in self.listdir if name.endswith('.' + "binvox")]
        
        # İndeksleme ve eşleştirme yaparak arama süresini azaltalım
        self.model_2d_files = {}
        self.model_2d_depth_files = {}
        
        for model_3d_file in self.model_3d_files:
            base_name = model_3d_file[:-7]
            self.model_2d_files[model_3d_file] = base_name + "_002.png"
            self.model_2d_depth_files[model_3d_file] = base_name + "_002" + "_depth.png"

    def __getitem__(self, index):
        model_3d_file = self.model_3d_files[index]
        model_2d_file = self.model_2d_files[model_3d_file]
        model_2d_file_depth = self.model_2d_depth_files[model_3d_file]

        # Voxel verisini yükleyelim
        volume = np.array(getVolumeFromBinvox(self.root + model_3d_file), dtype=np.float32).copy()
        
        try:
            # RGB ve depth görüntüleri ayrı ayrı normalize edelim
            rgb_image = self.rgb_transform(Image.open(self.root + model_2d_file))
            depth_image = self.depth_transform(Image.open(self.root + model_2d_file_depth))
            
            # Normalize edilmiş RGB ve depth görüntülerini birleştirelim
            combined_image = torch.cat((rgb_image, depth_image), dim=0)
            
            return (combined_image, torch.from_numpy(volume).float())
            
        except Exception as e:
            print(f"Error loading image {model_2d_file}: {str(e)}")
            # Hata durumunda boş veriler döndür
            return (torch.zeros(4, self.img_size, self.img_size), torch.zeros_like(torch.from_numpy(volume).float()))

    def __len__(self):
        return len(self.model_3d_files)

class ShapeNetMultiviewDataset(data.Dataset):
    """Custom Dataset compatible with torch.utils.data.DataLoader"""

    def __init__(self, root, args):
        """Set the path for Data.

        Args:
            root: image directory.
            transform: Tensor transformer.
        """
        self.root = root
        self.listdir = os.listdir(self.root)
        self.args = args
        self.img_size = args.image_size
        self.p = transforms.Compose([transforms.Resize((self.img_size, self.img_size))])

    def __getitem__(self, index):

        model_3d_file = [name for name in self.listdir if name.endswith('.' + "binvox")][index]

        model_2d_files = [name for name in self.listdir if name.startswith(model_3d_file[:-7]) and name.endswith(".png")][:3]
        #with open(self.root + model_3d_file, "rb") as f:
        volume = np.array(getVolumeFromBinvox(self.root + model_3d_file), dtype=np.float32)
        #print(volume.shape)
        #plotFromVoxels(volume)
        #with open(self.root + model_2d_file, "rb") as g:
        #image = np.array(imread(self.root + model_2d_file))
        images = [torch.FloatTensor(np.array(self.p(Image.open(self.root +x )))) for x in model_2d_files]
        return (images, torch.FloatTensor(volume) )

    def __len__(self):
        return len( [name for name in self.listdir if name.endswith('.' + "binvox")])


def var_or_cuda(x):
    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)  # Non-blocking transfer daha hızlı CPU-GPU transferi sağlar
    return Variable(x)

def generateZ(args):

    if args.z_dis == "norm":
        Z = var_or_cuda(torch.Tensor(args.batch_size, args.z_size).normal_(0, 0.33))
    elif args.z_dis == "uni":
        Z = var_or_cuda(torch.randn(args.batch_size, args.z_size))
    else:
        print("z_dist is not normal or uniform")

    return Z

########################## Pickle helper ###############################


def read_pickle(path, G, G_solver, D_, D_solver,E_=None,E_solver = None ):
    try:

        files = os.listdir(path)
        file_list = [int(file.split('_')[-1].split('.')[0]) for file in files]
        file_list.sort()
        recent_iter = str(file_list[-1])
        print(recent_iter, path)

        with open(path + "/G_" + recent_iter + ".pkl", "rb") as f:
            G.load_state_dict(torch.load(f))
        with open(path + "/G_optim_" + recent_iter + ".pkl", "rb") as f:
            G_solver.load_state_dict(torch.load(f))
        with open(path + "/D_" + recent_iter + ".pkl", "rb") as f:
            D_.load_state_dict(torch.load(f))
        with open(path + "/D_optim_" + recent_iter + ".pkl", "rb") as f:
            D_solver.load_state_dict(torch.load(f))
        if E_ is not None:
            with open(path + "/E_" + recent_iter + ".pkl", "rb") as f:
                E_.load_state_dict(torch.load(f))
            with open(path + "/E_optim_" + recent_iter + ".pkl", "rb") as f:
                E_solver.load_state_dict(torch.load(f))


    except Exception as e:

        print("fail try read_pickle", e)



def save_new_pickle(path, iteration, G, G_solver, D_, D_solver, E_=None,E_solver = None):
    if not os.path.exists(path):
        os.makedirs(path)

    with open(path + "/G_" + str(iteration) + ".pkl", "wb") as f:
        torch.save(G.state_dict(), f)
    with open(path + "/G_optim_" + str(iteration) + ".pkl", "wb") as f:
        torch.save(G_solver.state_dict(), f)
    with open(path + "/D_" + str(iteration) + ".pkl", "wb") as f:
        torch.save(D_.state_dict(), f)
    with open(path + "/D_optim_" + str(iteration) + ".pkl", "wb") as f:
        torch.save(D_solver.state_dict(), f)
    if E_ is not None:
        with open(path + "/E_" + str(iteration) + ".pkl", "wb") as f:
            torch.save(E_.state_dict(), f)
        with open(path + "/E_optim_" + str(iteration) + ".pkl", "wb") as f:
            torch.save(E_solver.state_dict(), f)

def calculate_iou(pred_voxel, gt_voxel, threshold=0.5):
    """
    3D Voxel tensörler için IoU (Intersection over Union) hesaplar.
    
    Args:
        pred_voxel: Tahmin edilen voxel tensor
        gt_voxel: Ground truth voxel tensor
        threshold: Tahmin edilen voxel'i binary'ye çevirmek için eşik değeri
        
    Returns:
        float: IoU değeri
    """
    # Tensörlerin uygun şekilde biçimlendirilmiş olduğundan emin olalım
    if pred_voxel.dim() != gt_voxel.dim():
        print(f"Uyarı: Tensör boyutları eşleşmiyor. pred_voxel: {pred_voxel.shape}, gt_voxel: {gt_voxel.shape}")
        if pred_voxel.dim() == 5 and gt_voxel.dim() == 4:  # Yaygın bir durum
            gt_voxel = gt_voxel.view(-1, 1, gt_voxel.size(1), gt_voxel.size(2), gt_voxel.size(3))
        elif pred_voxel.dim() == 4 and gt_voxel.dim() == 5:
            pred_voxel = pred_voxel.view(-1, 1, pred_voxel.size(1), pred_voxel.size(2), pred_voxel.size(3))
    
    # Tahmin edilen voxel'i binary'ye çevirme (eşik değerinden büyükse 1, değilse 0)
    pred_binary = (pred_voxel > threshold).float()
    gt_binary = (gt_voxel > threshold).float()  # GT'nin de binary olduğundan emin olalım
    
    # Kesişim ve birleşim hesaplama
    intersection = torch.sum(pred_binary * gt_binary).float()
    union = torch.sum(torch.clamp(pred_binary + gt_binary, 0, 1)).float()
    
    # Sıfıra bölme hatasını önlemek için kontrol
    if union.item() < 1e-6:
        return 0.0
        
    return (intersection / union).item()

def calculate_metrics(y_pred, y_true, threshold=0.5):
    """
    Discriminator için precision, recall ve F1 score hesaplar.
    
    Args:
        y_pred: Tahmin edilen değerler (discriminator çıktısı)
        y_true: Gerçek etiketler
        threshold: İkili sınıflandırma için eşik değeri
        
    Returns:
        tuple: (precision, recall, f1_score) değerleri
    """
    # Tahminleri binary'ye çevirme
    y_pred_binary = (y_pred >= threshold).float()
    
    # True Positive (TP), False Positive (FP), False Negative (FN) hesaplama
    tp = torch.sum(y_pred_binary * y_true).item()
    fp = torch.sum(y_pred_binary * (1 - y_true)).item()
    fn = torch.sum((1 - y_pred_binary) * y_true).item()
    
    # Sıfıra bölme hatasını önlemek için kontrol
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # F1 Score hesaplama
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1_score

def get_mixup_params(batch_size, alpha, device):
    """
    MixUp işlemi için bir lambda oranı ve karıştırma indeksleri üretir.
    """
    if alpha > 0:
        lam = torch.distributions.beta.Beta(alpha, alpha).sample().to(device)
    else:
        lam = torch.tensor(1.0, device=device)
    indices = torch.randperm(batch_size, device=device)
    return lam, indices

def apply_mixup_with_params(x, lam, indices):
    """
    Belirli parametrelerle (lambda ve indices) herhangi bir boyuttaki tensöre MixUp uygular.
    """
    if x is None:
        return None
        
    # Lambda değerini tensör boyutuna göre yeniden şekillendir
    if x.dim() == 5:  # 3D voxel (B, C, D, H, W)
        lam_reshaped = lam.view(1, 1, 1, 1, 1)
    elif x.dim() == 4:  # 2D görüntü (B, C, H, W)
        lam_reshaped = lam.view(1, 1, 1, 1)
    elif x.dim() == 2:  # Latent vektör (B, Z)
        lam_reshaped = lam.view(1, 1)
    else:
        lam_reshaped = lam
        
    mixed_x = lam_reshaped * x + (1 - lam_reshaped) * x[indices]
    return mixed_x

def calculate_wasserstein_loss_d(d_real, d_fake):
    """
    Wasserstein loss for discriminator/critic
    
    Args:
        d_real: Discriminator çıktısı (gerçek veri)
        d_fake: Discriminator çıktısı (sahte veri)
        
    Returns:
        Discriminator için Wasserstein loss
    """
    # Gerçek örnekler için ortalama çıktı
    d_real_mean = torch.mean(d_real)
    
    # Sahte örnekler için ortalama çıktı
    d_fake_mean = torch.mean(d_fake)
    
    # Critic loss: gerçek örneklerde yüksek değer, sahte örneklerde düşük değer
    # Negatif işaret, minimizasyon optimizer'ı kullandığımız için
    d_loss = -d_real_mean + d_fake_mean
    
    return d_loss

def calculate_wasserstein_loss_g(d_fake):
    """
    Wasserstein loss for generator
    
    Args:
        d_fake: Discriminator çıktısı (sahte veri)
        
    Returns:
        Generator için Wasserstein loss
    """
    # Generator, discriminator'ın sahte örnekler için yüksek değer üretmesini ister
    # Negatif işaret, minimizasyon optimizer'ı kullandığımız için
    g_loss = -torch.mean(d_fake)
    
    return g_loss

def compute_gradient_penalty(discriminator, real_samples, fake_samples, device="cuda", lambda_gp=10.0):
    """
    WGAN-GP için gradient penalty hesaplama
    
    Args:
        discriminator: Discriminator modeli
        real_samples: Gerçek örnekler
        fake_samples: Sahte örnekler
        device: İşlemin yapılacağı cihaz
        lambda_gp: Gradient penalty katsayısı
        
    Returns:
        Gradient penalty değeri
    """
    # Batch boyutu
    batch_size = real_samples.size(0)
    
    # Gerçek ve sahte örnekler arasında rastgele interpolasyon
    alpha = torch.rand(batch_size, 1, 1, 1, 1, device=device)
    # alpha'nın real_samples ile aynı boyutta olması için genişletme
    alpha = alpha.expand_as(real_samples)
    
    # Interpolasyon örnekleri
    interpolates = alpha * real_samples + (1 - alpha) * fake_samples
    interpolates = interpolates.requires_grad_(True)
    
    # Interpolasyon örneklerine discriminator uygulama
    d_interpolates = discriminator(interpolates)
    
    # Sabit değerli gradyan hedefi
    fake = torch.ones(d_interpolates.size(), device=device, requires_grad=False)
    
    # Çıktıya göre gradyanları hesaplama
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    
    # Gradyanları düzleştir
    gradients = gradients.view(batch_size, -1)
    
    # Gradient norm hesaplama (L2 norm)
    gradient_norm = gradients.norm(2, dim=1)
    
    # Gradient penalty hesaplama
    gradient_penalty = lambda_gp * ((gradient_norm - 1) ** 2).mean()
    
    return gradient_penalty

# Wasserstein ve geleneksel GAN ayrımı için doğruluk hesaplama
def calculate_accuracy_for_wasserstein(d_real, d_fake, is_wasserstein=False):
    """
    Wasserstein veya geleneksel GAN için doğruluk hesaplama
    
    Args:
        d_real: Discriminator çıktısı (gerçek veri)
        d_fake: Discriminator çıktısı (sahte veri)
        is_wasserstein: Wasserstein modu aktif mi?
        
    Returns:
        Doğruluk değeri
    """
    if is_wasserstein:
        # WGAN için skorlar doğrudan karşılaştırılır - sigmoid kullanılmaz
        d_real_acu = torch.gt(d_real, 0).float()
        d_fake_acu = torch.lt(d_fake, 0).float()
    else:
        # Geleneksel GAN için sigmoid uygulayarak 0.5 eşiğine göre karar verilir
        d_real_acu = torch.ge(torch.sigmoid(d_real.squeeze()), 0.5).float()
        d_fake_acu = torch.le(torch.sigmoid(d_fake.squeeze()), 0.5).float()
    
    # Tüm tahminlerin ortalamasını al
    d_total_acu = torch.mean(torch.cat((d_real_acu, d_fake_acu), 0))
    return d_total_acu

def visualize_comparative_results(input_image, generated_voxel, gt_voxel, save_path=None, title="Comparative Visualization"):
    """
    2D girdi resmi, Generator tarafından üretilen 3D model ve ground truth 3D modeli yan yana gösterir.
    
    Args:
        input_image: 2D girdi resmi (4 kanal - RGB + Depth)
        generated_voxel: Generator tarafından üretilen voxel verisi
        gt_voxel: Ground truth voxel verisi
        save_path: Görselin kaydedileceği dosya yolu (None ise gösterilir, kaydedilmez)
        title: Görsel başlığı
    """
    # Görsel boyutu ve düzeni
    fig = plt.figure(figsize=(18, 6))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])
    
    # 2D girdi resmini göster
    ax1 = plt.subplot(gs[0])
    # RGB kanallarını ayır (ilk 3 kanal)
    rgb_img = input_image[:3].permute(1, 2, 0)
    
    # De-normalizasyon (ImageNet normalizasyonunu geri al)
    rgb_mean = torch.tensor([0.485, 0.456, 0.406])
    rgb_std = torch.tensor([0.229, 0.224, 0.225])
    rgb_img = rgb_img * rgb_std + rgb_mean
    
    # Değerleri [0, 1] aralığına sınırla
    rgb_img = torch.clamp(rgb_img, 0, 1)
    
    # 2D RGB görüntüyü göster
    ax1.imshow(rgb_img)
    ax1.set_title("2D Input (RGB)")
    ax1.axis('off')
    
    # Generator çıktısını göster (3D voxel)
    ax2 = plt.subplot(gs[1], projection='3d')
    # 0.5'ten büyük değerlere sahip voxelleri göster
    x_gen, y_gen, z_gen = np.where(generated_voxel > 0.5)
    ax2.scatter(x_gen, y_gen, z_gen, c='blue', marker='o', alpha=0.7, s=10)
    ax2.set_title("Generated 3D Model")
    
    # 3D görünümü ayarla
    ax2.set_xlim(0, generated_voxel.shape[0])
    ax2.set_ylim(0, generated_voxel.shape[1])
    ax2.set_zlim(0, generated_voxel.shape[2])
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    ax2.set_zticklabels([])
    ax2.view_init(elev=30, azim=45)  # Görünüm açısını ayarla
    
    # Ground truth voxel göster
    ax3 = plt.subplot(gs[2], projection='3d')
    x_gt, y_gt, z_gt = np.where(gt_voxel > 0.5)
    ax3.scatter(x_gt, y_gt, z_gt, c='red', marker='o', alpha=0.7, s=10)
    ax3.set_title("Ground Truth 3D Model")
    
    # 3D görünümü ayarla (aynı ölçekleri ve açıları kullan)
    ax3.set_xlim(0, gt_voxel.shape[0])
    ax3.set_ylim(0, gt_voxel.shape[1])
    ax3.set_zlim(0, gt_voxel.shape[2])
    ax3.set_xticklabels([])
    ax3.set_yticklabels([])
    ax3.set_zticklabels([])
    ax3.view_init(elev=30, azim=45)  # GT için de aynı açıyı kullan
    
    # Genel başlık
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    # Kaydet veya göster
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
    else:
        plt.show()