import torch
from torch import optim
from collections import OrderedDict
from utils import make_hyparam_string, SavePloat_Voxels, calculate_iou
import os
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch.amp import autocast
import matplotlib.gridspec as gridspec
from matplotlib.colors import LightSource
from matplotlib import cm
import matplotlib as mpl
from scipy.ndimage import rotate
from PIL import Image
import torchvision.transforms as transforms

from utils import ShapeNetPlusImageDataset, var_or_cuda, visualize_comparative_results, calculate_iou
from model import _G, _D, _E

# Set seed for better reproducibility
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Set a nice modern style for plots
plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']

# Define a modern color palette
COLOR_PALETTE = {
    'generated': '#3498db',  # Blue
    'ground_truth': '#e74c3c',  # Red
    'input_rgb': '#2ecc71',  # Green
    'depth': '#9b59b6',  # Purple
    'background': '#ffffff',
    'text': '#2c3e50',
    'accent': '#f39c12'  # Orange
}

def save_input_images(input_tensor, save_rgb_path, save_depth_path):
    """
    Giriş tensor'ını RGB ve depth görüntüleri olarak kaydeder.
    
    Args:
        input_tensor: 4 kanallı giriş tensörü (RGB + Depth)
        save_rgb_path: RGB görüntüsünün kaydedileceği dosya yolu
        save_depth_path: Depth görüntüsünün kaydedileceği dosya yolu
        
    Returns:
        bool: Başarılı ise True
    """
    try:
        # RGB kanallarını ayır (ilk 3 kanal)
        rgb_img = input_tensor[:3].permute(1, 2, 0)
        depth_img = input_tensor[3].unsqueeze(0)
        
        # ImageNet normalizasyonunu geri al
        rgb_mean = torch.tensor([0.485, 0.456, 0.406])
        rgb_std = torch.tensor([0.229, 0.224, 0.225])
        rgb_img = rgb_img * rgb_std + rgb_mean
        
        # Depth normalizasyonunu geri al
        depth_mean = torch.tensor([0.330])
        depth_std = torch.tensor([0.236])
        depth_img = depth_img * depth_std + depth_mean
        
        # [0, 1] aralığına sıkıştır
        rgb_img = torch.clamp(rgb_img, 0, 1)
        depth_img = torch.clamp(depth_img, 0, 1)
        
        # RGB görüntüyü kaydet
        rgb_np = (rgb_img.numpy() * 255).astype(np.uint8)
        rgb_pil = Image.fromarray(rgb_np)
        rgb_pil.save(save_rgb_path)
        
        # Depth görüntüyü kaydet - Önce normalize edip 0-255 aralığına getir
        depth_np = (depth_img.numpy()[0] * 255).astype(np.uint8)
        
        # Depth'i jet colormap ile görselleştir
        colored_depth = cm.plasma(depth_np.astype(float) / 255.0)
        colored_depth = (colored_depth[:, :, :3] * 255).astype(np.uint8)
        depth_pil = Image.fromarray(colored_depth)
        depth_pil.save(save_depth_path)
        
        print(f"RGB ve Depth görüntüleri kaydedildi: {save_rgb_path}, {save_depth_path}")
        return True
        
    except Exception as e:
        print(f"Görüntüler kaydedilirken hata: {e}")
        return False

def voxel_to_obj(voxel_data, threshold=0.5, filename="output.obj"):
    """
    Voxel verisini .obj formatına dönüştürür.
    
    Args:
        voxel_data: 3D numpy array formatında voxel verisi
        threshold: Voxel'in var kabul edileceği eşik değeri
        filename: Çıktı dosyasının tam yolu
        
    Returns:
        bool: Başarılı ise True
    """
    try:
        # Voxel eşikleme
        voxel_data = voxel_data > threshold
        
        # OBJ dosyasını açma
        with open(filename, 'w') as f:
            # OBJ dosyası başlığı
            f.write("# Voxel to OBJ conversion - SWAGE-3D\n")
            f.write(f"# Created: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Vertex listesini oluşturacağız
            vertex_count = 0
            vertices = []
            faces = []
            
            # Voxel'in boyutlarını alma
            depth, height, width = voxel_data.shape
            
            # Voxel merkez pozisyonunu hesaplama (merkezi 0,0,0 yapmak için)
            center_x = width / 2
            center_y = height / 2
            center_z = depth / 2
            
            # Dolu voxelleri bul
            for z in range(depth):
                for y in range(height):
                    for x in range(width):
                        if voxel_data[z, y, x]:
                            # Küp köşe noktalarının koordinatları (normalize edilmiş)
                            # Voxel büyüklüğünü 1 birim olarak alıyoruz
                            v = [
                                (x - center_x, y - center_y, z - center_z),           # 0: sol alt arka
                                (x+1 - center_x, y - center_y, z - center_z),          # 1: sağ alt arka
                                (x+1 - center_x, y+1 - center_y, z - center_z),        # 2: sağ üst arka
                                (x - center_x, y+1 - center_y, z - center_z),          # 3: sol üst arka
                                (x - center_x, y - center_y, z+1 - center_z),          # 4: sol alt ön
                                (x+1 - center_x, y - center_y, z+1 - center_z),        # 5: sağ alt ön
                                (x+1 - center_x, y+1 - center_y, z+1 - center_z),      # 6: sağ üst ön
                                (x - center_x, y+1 - center_y, z+1 - center_z)         # 7: sol üst ön
                            ]
                            
                            # Komşu voxelleri kontrol et ve sadece görünür yüzeyleri ekle
                            # Arka yüz
                            if z == 0 or not voxel_data[z-1, y, x]:
                                vertices.extend([v[0], v[1], v[2], v[3]])
                                faces.append((vertex_count, vertex_count+1, vertex_count+2, vertex_count+3))
                                vertex_count += 4
                            
                            # Ön yüz
                            if z == depth-1 or not voxel_data[z+1, y, x]:
                                vertices.extend([v[4], v[5], v[6], v[7]])
                                faces.append((vertex_count, vertex_count+1, vertex_count+2, vertex_count+3))
                                vertex_count += 4
                            
                            # Alt yüz
                            if y == 0 or not voxel_data[z, y-1, x]:
                                vertices.extend([v[0], v[1], v[5], v[4]])
                                faces.append((vertex_count, vertex_count+1, vertex_count+2, vertex_count+3))
                                vertex_count += 4
                            
                            # Üst yüz
                            if y == height-1 or not voxel_data[z, y+1, x]:
                                vertices.extend([v[3], v[2], v[6], v[7]])
                                faces.append((vertex_count, vertex_count+1, vertex_count+2, vertex_count+3))
                                vertex_count += 4
                            
                            # Sol yüz
                            if x == 0 or not voxel_data[z, y, x-1]:
                                vertices.extend([v[0], v[3], v[7], v[4]])
                                faces.append((vertex_count, vertex_count+1, vertex_count+2, vertex_count+3))
                                vertex_count += 4
                            
                            # Sağ yüz
                            if x == width-1 or not voxel_data[z, y, x+1]:
                                vertices.extend([v[1], v[2], v[6], v[5]])
                                faces.append((vertex_count, vertex_count+1, vertex_count+2, vertex_count+3))
                                vertex_count += 4
            
            # Vertex'leri yaz
            for vertex in vertices:
                f.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
            
            # Yüzleri yaz
            for face in faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1} {face[3]+1}\n")
            
            # OBJ dosyası kapanış bilgileri
            f.write("\n# End of OBJ file\n")
            
        print(f"OBJ dosyası başarıyla oluşturuldu: {filename}")
        return True
    
    except Exception as e:
        print(f"OBJ dosyası oluşturulurken hata: {e}")
        return False

def rotate_voxel(voxel, rot_x=0, rot_y=0, rot_z=0):
    """
    3D voxel verisine rotasyon uygular. Her eksende ayrı ayrı rotasyon yapılabilir.
    
    Args:
        voxel: Numpy array formatında voxel verisi
        rot_x: X ekseni etrafında rotasyon açısı (derece) - Pitch (yukarı/aşağı)
        rot_y: Y ekseni etrafında rotasyon açısı (derece) - Yaw (sağa/sola)
        rot_z: Z ekseni etrafında rotasyon açısı (derece) - Roll (saat yönü/tersi)
    
    Returns:
        Rotasyon uygulanmış voxel verisi
    """
    # Veri tipini float32 olarak değiştir ve kopyala
    rotated = np.copy(voxel).astype(np.float32)
    
    # X ekseni etrafında rotasyon (pitch - yukarı/aşağı)
    if rot_x != 0:
        rotated = rotate(rotated, angle=rot_x, axes=(1, 2), reshape=False, order=0, mode='constant', cval=0.0)
    
    # Y ekseni etrafında rotasyon (yaw - sağa/sola)
    if rot_y != 0:
        rotated = rotate(rotated, angle=rot_y, axes=(0, 2), reshape=False, order=0, mode='constant', cval=0.0)
    
    # Z ekseni etrafında rotasyon (roll - yukarıdan bakışta saat yönü/tersi)
    if rot_z != 0:
        rotated = rotate(rotated, angle=rot_z, axes=(0, 1), reshape=False, order=0, mode='constant', cval=0.0)
    
    return rotated

def create_high_quality_visualization(input_image, generated_voxel, gt_voxel, iou_score, sample_idx, save_path, 
                                      resolution=600, dpi=300, category_name="Unknown", args=None):
    """
    Creates a high-quality visualization suitable for academic publications.
    
    Args:
        input_image: 2D input image (4 channels - RGB + Depth)
        generated_voxel: Generated voxel data from the model
        gt_voxel: Ground truth voxel data
        iou_score: IoU score
        sample_idx: Sample index
        save_path: Path to save the visualization
        resolution: Resolution (pixels)
        dpi: Dots per inch
        category_name: Object category name
        args: Argümanlar (rotasyon parametreleri için)
    """
    # Set up figure
    plt.figure(figsize=(15, 7), dpi=dpi, facecolor=COLOR_PALETTE['background'])
    
    gs = gridspec.GridSpec(2, 3, height_ratios=[1, 3], width_ratios=[1, 1, 1])
    
    # Title
    """plt.suptitle(f"SWAGE-3D: 3D Shape Reconstruction from a Single Image\nCategory: {category_name} - IoU: {iou_score:.4f}", 
                 fontsize=18, y=0.98, color=COLOR_PALETTE['text'], fontweight='bold')"""
    
    # Top row - RGB input and depth map
    ax_rgb = plt.subplot(gs[0, 0:2])
    ax_depth = plt.subplot(gs[0, 2])
    
    # Extract RGB channels (first 3 channels)
    rgb_img = input_image[:3].permute(1, 2, 0)
    depth_img = input_image[3].unsqueeze(0).permute(1, 2, 0)
    
    # De-normalization (revert ImageNet normalization)
    rgb_mean = torch.tensor([0.485, 0.456, 0.406])
    rgb_std = torch.tensor([0.229, 0.224, 0.225])
    rgb_img = rgb_img * rgb_std + rgb_mean
    
    # De-normalize depth map
    depth_mean = torch.tensor([0.330])
    depth_std = torch.tensor([0.236])
    depth_img = depth_img * depth_std + depth_mean
    
    # Clamp values to [0, 1] range
    rgb_img = torch.clamp(rgb_img, 0, 1)
    depth_img = torch.clamp(depth_img, 0, 1)
    
    # Display images
    ax_rgb.imshow(rgb_img)
    ax_rgb.set_title("Input: RGB Image", fontsize=14, color=COLOR_PALETTE['text'], fontweight='bold')
    ax_rgb.axis('off')
    
    # Display depth map as a heatmap with a more appealing colormap
    depth_cmap = plt.cm.plasma
    depth_plot = ax_depth.imshow(depth_img.squeeze(), cmap=depth_cmap)
    ax_depth.set_title("Input: Depth Map", fontsize=14, color=COLOR_PALETTE['text'], fontweight='bold')
    ax_depth.axis('off')
    
    # Add color bar for depth map
    cbar = plt.colorbar(depth_plot, ax=ax_depth, orientation='vertical', shrink=0.8)
    cbar.ax.set_ylabel('Depth', rotation=270, labelpad=15)
    
    # Bottom row - 3D Models
    ax_generated = plt.subplot(gs[1, 0], projection='3d', facecolor=COLOR_PALETTE['background'])
    ax_gt = plt.subplot(gs[1, 1], projection='3d', facecolor=COLOR_PALETTE['background'])
    ax_overlay = plt.subplot(gs[1, 2], projection='3d', facecolor=COLOR_PALETTE['background'])
    
    # Voxel visualization function
    def visualize_voxels(ax, voxels, title, color='blue', alpha=0.7, rot_x=30, rot_y=0, rot_z=45):
        # Tüm rotasyonları bir arada uygula
        # Veri tipini float32 olarak değiştir
        rotated_voxels = voxels.copy().astype(np.float32)
        
        # Rotasyonları uygula
        if rot_x != 0 or rot_y != 0 or rot_z != 0:
            try:
                rotated_voxels = rotate_voxel(rotated_voxels, 
                                            rot_x=rot_x, 
                                            rot_y=rot_y, 
                                            rot_z=rot_z)
            except Exception as e:
                print(f"Rotasyon sırasında hata: {e}")
                # Hata alınırsa rotasyon uygulama, orijinal veriyi kullan
                pass
        
        # Eşikleme
        threshold = 0.3  # Daha düşük threshold daha çok voxel gösterir
        filled = rotated_voxels > threshold
        
        # Voxel görselleştirme
        ax.voxels(filled, facecolors=color, edgecolor='k', linewidth=0.2, alpha=alpha)
        
        # View ayarları
        ax.set_xlim(0, voxels.shape[0])
        ax.set_ylim(0, voxels.shape[1])
        ax.set_zlim(0, voxels.shape[2])
        
        # Eksenleri temizle
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        
        # Sabit bir bakış açısı kullan
        ax.view_init(elev=30, azim=45)  # Sabit bakış açısı
        
        # Başlık
        ax.set_title(title, fontsize=14, color=COLOR_PALETTE['text'], fontweight='bold', y=1.05)
    
    # Varsayılan rotasyon değerleri 
    gen_rot_x = 30
    gen_rot_y = 0
    gen_rot_z = 45
    gt_rot_x = 30
    gt_rot_y = 0
    gt_rot_z = 45
    
    # Eğer args parametresi verilmişse rotasyon değerlerini al
    if args is not None:
        if hasattr(args, 'gen_rot_x'):
            gen_rot_x = args.gen_rot_x
        if hasattr(args, 'gen_rot_y'):
            gen_rot_y = args.gen_rot_y
        if hasattr(args, 'gen_rot_z'):
            gen_rot_z = args.gen_rot_z
        if hasattr(args, 'gt_rot_x'):
            gt_rot_x = args.gt_rot_x
        if hasattr(args, 'gt_rot_y'):
            gt_rot_y = args.gt_rot_y
        if hasattr(args, 'gt_rot_z'):
            gt_rot_z = args.gt_rot_z
    
    # Visualize 3D models - generated model
    visualize_voxels(ax_generated, generated_voxel, "Generated 3D Model", 
                     color=COLOR_PALETTE['generated'], alpha=0.7, 
                     rot_x=gen_rot_x, rot_y=gen_rot_y, rot_z=gen_rot_z)
    
    # Visualize 3D models - ground truth model
    visualize_voxels(ax_gt, gt_voxel, "Ground Truth 3D Model", 
                     color=COLOR_PALETTE['ground_truth'], alpha=0.7, 
                     rot_x=gt_rot_x, rot_y=gt_rot_y, rot_z=gt_rot_z)
    
    # Visualize 3D models - comparison
    visualize_voxels(ax_overlay, generated_voxel, "Comparison (Generated vs. Ground Truth)", 
                     color=COLOR_PALETTE['generated'], alpha=0.6, 
                     rot_x=gen_rot_x, rot_y=gen_rot_y, rot_z=gen_rot_z)
                     
    visualize_voxels(ax_overlay, gt_voxel, "", 
                     color=COLOR_PALETTE['ground_truth'], alpha=0.3, 
                     rot_x=gt_rot_x, rot_y=gt_rot_y, rot_z=gt_rot_z)
    
    # Footer
    """plt.figtext(0.5, 0.01, "SWAGE-3D: Single-View 3D Shape Generation and Reconstruction", 
                ha="center", fontsize=12, fontweight='bold',
                bbox={"facecolor":COLOR_PALETTE['accent'], "alpha":0.2, "pad":5, "boxstyle":"round,pad=0.5"})"""
    
    # Tighten layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save
    plt.savefig(save_path, bbox_inches='tight', dpi=dpi, facecolor=COLOR_PALETTE['background'])
    plt.close()
    
    print(f"High-quality visualization saved: {save_path}")

def test_best_samples_3DVAEGAN(args):
    """
    Creates a specialized test and visualization for the best results.
    Selects examples with the highest IoU scores for use in the paper.
    """
    print("SWAGE-3D: Testing and visualizing best samples for publication")
    
    # Hyperparameter list
    hyparam_list = [("model", args.model_name),
                   ("cube", args.cube_len),
                   ("bs", args.batch_size),
                   ("g_lr", args.g_lr),
                   ("d_lr", args.d_lr),
                   ("z", args.z_dis),
                   ("bias", args.bias),
                   ("sl", args.soft_label),
                   ("wgan", args.wasserstein),
                   ("sn", args.use_spectral_norm),
                   ("atn", args.use_attention),
                   ("sch", args.use_scheduler)]

    hyparam_dict = OrderedDict(((arg, value) for arg, value in hyparam_list))
    log_param = make_hyparam_string(hyparam_dict)
    print(f"Test parameters: {log_param}")
    
    # Load test dataset
    dsets_path = args.input_dir + args.data_dir + "test/"
    print(f"Test dataset path: {dsets_path}")
    
    # Get category name (from args.data_dir)
    category_name = args.data_dir.strip("/").capitalize()
    print(f"Category: {category_name}")
    
    # Optimized dataloader settings
    dsets = ShapeNetPlusImageDataset(dsets_path, args)
    
    # Use larger batch size for testing, improves speed
    test_batch_size = min(32, args.batch_size * 2)
    print(f"Test batch size: {test_batch_size}")
    
    dset_loaders = torch.utils.data.DataLoader(
        dsets, 
        batch_size=test_batch_size,
        shuffle=False,  # No shuffling for test
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=(args.num_workers > 0),
        drop_last=False
    )
    
    # Load model and set to evaluation mode
    print("Loading model...")
    
    # Create model instances
    E = _E(args)
    G = _G(args)
    D = _D(args)
    
    # Create optimizer instances (required for loading weights)
    G_solver = optim.Adam(G.parameters(), lr=args.g_lr, betas=args.beta)
    E_solver = optim.Adam(E.parameters(), lr=args.e_lr, betas=args.beta)
    D_solver = optim.Adam(D.parameters(), lr=args.d_lr, betas=args.beta)
    
    # Move to GPU
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        G.cuda()
        E.cuda()
        D.cuda()
    
    # Load model weights
    pickle_path = args.output_dir + args.pickle_dir + log_param
    print(f"Loading weights from: {pickle_path}")
    
    try:
        # Find latest model weights
        files = os.listdir(pickle_path)
        generator_files = [f for f in files if f.startswith("G_") and f.endswith(".pkl")]
        generator_epochs = [int(f.split('_')[-1].split('.')[0]) for f in generator_files]
        
        if not generator_epochs:
            raise FileNotFoundError("Model weights not found")
            
        # Get highest epoch value
        latest_epoch = 139
        
        # Load weights
        print(f"Loading epoch {latest_epoch}...")
        
        from utils import read_pickle
        read_pickle(pickle_path, G, G_solver, D, D_solver, E, E_solver)
        
        print(f"Successfully loaded epoch {latest_epoch}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Set models to evaluation mode
    G.eval()
    E.eval()
    D.eval()
    
    # Lists for storing best samples
    all_samples = []
    
    # Process test dataset
    test_start_time = time.time()
    
    print("Processing all samples and calculating IoU scores...")
    
    with torch.no_grad():
        for i, (image, model_3d) in enumerate(dset_loaders):
            # Move data to GPU
            if torch.cuda.is_available():
                X = model_3d.to(device='cuda', non_blocking=True)
                image = image.to(device='cuda', non_blocking=True)
            else:
                X = var_or_cuda(model_3d)
                image = var_or_cuda(image)
            
            # Fix voxel shape
            X_reshaped = X.view(-1, 1, args.cube_len, args.cube_len, args.cube_len)
            
            # Reconstruction with Encoder and Generator
            with autocast(device_type='cuda', enabled=args.use_amp):
                z_mu, z_var = E(image)
                Z_vae = E.reparameterize(z_mu, z_var)
                G_vae = G(Z_vae)
            
            # Calculate IoU for each sample in the batch
            batch_size = X.size(0)
            
            for j in range(batch_size):
                sample_input = image[j].cpu()
                sample_generated = G_vae[j].cpu().squeeze().numpy()
                sample_gt = X_reshaped[j].cpu().squeeze().numpy()
                
                # Calculate IoU
                sample_iou = calculate_iou(
                    torch.tensor(sample_generated).unsqueeze(0).unsqueeze(0), 
                    torch.tensor(sample_gt).unsqueeze(0).unsqueeze(0)
                )
                
                # Store sample
                all_samples.append({
                    'input': sample_input,
                    'generated': sample_generated,
                    'ground_truth': sample_gt,
                    'iou': sample_iou,
                    'batch_idx': i,
                    'sample_idx': j
                })
            
            # Print progress
            if (i+1) % 5 == 0 or i == 0:
                print(f"Processed batch: {i+1}/{len(dset_loaders)}")
    
    # Processing time
    test_time = time.time() - test_start_time
    print(f"All samples processed in {test_time:.2f} seconds")
    
    # Sort by IoU value (highest to lowest)
    all_samples.sort(key=lambda x: x['iou'], reverse=True)
    
    # Select top N samples
    num_best_samples = args.num_best_samples if hasattr(args, 'num_best_samples') else 3
    best_samples = all_samples[:num_best_samples]
    
    print(f"\nSelected {num_best_samples} samples with highest IoU scores:")
    for i, sample in enumerate(best_samples):
        print(f"Sample {i+1}: IoU = {sample['iou']:.4f}, Batch {sample['batch_idx']}, Sample {sample['sample_idx']}")
    
    # Directory for results
    results_dir = os.path.join(args.output_dir, "best_samples_results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # OBJ dosyaları için klasör oluştur
    obj_dir = os.path.join(results_dir, "obj_files")
    if not os.path.exists(obj_dir):
        os.makedirs(obj_dir)
    
    # RGB ve Depth görüntüleri için klasör oluştur
    rgb_dir = os.path.join(obj_dir, "rgb_images")
    depth_dir = os.path.join(obj_dir, "depth_images")
    if not os.path.exists(rgb_dir):
        os.makedirs(rgb_dir)
    if not os.path.exists(depth_dir):
        os.makedirs(depth_dir)
    
    # Timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # En iyi örnekler için OBJ dosyaları oluştur
    print("\nEn iyi voxel örneklerini OBJ formatına dönüştürüyorum...")
    for i, sample in enumerate(best_samples):
        # Generated model için OBJ dosyası
        gen_obj_path = os.path.join(obj_dir, f"generated_sample_{i+1}_iou_{sample['iou']:.4f}_{timestamp}.obj")
        gen_threshold = 0.3  # Daha düşük eşik, daha fazla detay gösterir
        voxel_to_obj(sample['generated'], threshold=gen_threshold, filename=gen_obj_path)
        
        # Ground truth model için OBJ dosyası
        gt_obj_path = os.path.join(obj_dir, f"groundtruth_sample_{i+1}_iou_{sample['iou']:.4f}_{timestamp}.obj")
        gt_threshold = 0.3
        voxel_to_obj(sample['ground_truth'], threshold=gt_threshold, filename=gt_obj_path)
        
        # RGB ve Depth görüntülerini kaydet
        rgb_path = os.path.join(rgb_dir, f"rgb_sample_{i+1}_iou_{sample['iou']:.4f}_{timestamp}.png")
        depth_path = os.path.join(depth_dir, f"depth_sample_{i+1}_iou_{sample['iou']:.4f}_{timestamp}.png")
        save_input_images(sample['input'], rgb_path, depth_path)
    
    print(f"Toplam {len(best_samples)*2} OBJ dosyası başarıyla oluşturuldu ve kaydedildi: {obj_dir}")
    print(f"Toplam {len(best_samples)*2} RGB ve Depth görüntüsü kaydedildi: {rgb_dir}, {depth_dir}")
    
    # High quality visualizations
    high_quality_dir = os.path.join(results_dir, "high_quality_samples")
    if not os.path.exists(high_quality_dir):
        os.makedirs(high_quality_dir)
    
    print("\nCreating high-quality visualizations...")
    
    # Visualize best samples - high quality
    for i, sample in enumerate(best_samples):
        # Create high-quality visualization
        high_quality_path = os.path.join(high_quality_dir, f"best_sample_{i+1}_iou_{sample['iou']:.4f}_{timestamp}.png")
        
        create_high_quality_visualization(
            input_image=sample['input'],
            generated_voxel=sample['generated'],
            gt_voxel=sample['ground_truth'],
            iou_score=sample['iou'],
            sample_idx=i+1,
            save_path=high_quality_path,
            category_name=category_name,
            args=args  # Rotasyon parametreleri için args geçiyoruz
        )
    
    # Create a composite image showing all top samples together
    print("\nCreating composite visualization...")
    
    fig = plt.figure(figsize=(18, 6 * num_best_samples), dpi=200, facecolor=COLOR_PALETTE['background'])
    
    for i, sample in enumerate(best_samples):
        row_idx = i
        
        # RGB Image
        ax_rgb = plt.subplot2grid((num_best_samples, 3), (row_idx, 0))
        
        # Extract and de-normalize RGB channels
        rgb_img = sample['input'][:3].permute(1, 2, 0)
        rgb_mean = torch.tensor([0.485, 0.456, 0.406])
        rgb_std = torch.tensor([0.229, 0.224, 0.225])
        rgb_img = rgb_img * rgb_std + rgb_mean
        rgb_img = torch.clamp(rgb_img, 0, 1)
        
        # Display RGB image
        ax_rgb.imshow(rgb_img)
        ax_rgb.set_title(f"Sample {i+1}: Input RGB+Depth Image", 
                        fontsize=14, color=COLOR_PALETTE['text'], fontweight='bold')
        ax_rgb.axis('off')
        
        # Generated 3D Model - Rotasyon parametrelerini kullan
        ax_gen = plt.subplot2grid((num_best_samples, 3), (row_idx, 1), projection='3d', facecolor=COLOR_PALETTE['background'])
        
        try:
            # Tüm rotasyonları bir arada uygula
            rotated_gen = sample['generated'].copy().astype(np.float32)
            
            if hasattr(args, 'gen_rot_x') or hasattr(args, 'gen_rot_y') or hasattr(args, 'gen_rot_z'):
                rot_x = args.gen_rot_x if hasattr(args, 'gen_rot_x') else 0
                rot_y = args.gen_rot_y if hasattr(args, 'gen_rot_y') else 0
                rot_z = args.gen_rot_z if hasattr(args, 'gen_rot_z') else 0
                
                try:
                    rotated_gen = rotate_voxel(rotated_gen, rot_x=rot_x, rot_y=rot_y, rot_z=rot_z)
                except Exception as e:
                    print(f"Generated model rotasyon hatası: {e}")
                    # Hata olursa orijinal veriyi kullan
                    rotated_gen = sample['generated'].copy().astype(np.float32)
            
            # Threshold değerini düşürerek daha fazla voxel gösterilmesini sağla
            threshold = 0.5  # 0.5 yerine daha düşük
            gen_mask = rotated_gen > threshold
            
            ax_gen.voxels(gen_mask, facecolors=COLOR_PALETTE['generated'], edgecolor='k', linewidth=0.2, alpha=0.8)
            ax_gen.set_title(f"Generated 3D Model (IoU: {sample['iou']:.4f})", 
                             fontsize=14, color=COLOR_PALETTE['text'], fontweight='bold')
            ax_gen.axis('off')
            ax_gen.view_init(elev=30, azim=45)
        except Exception as e:
            print(f"Generated model visualization error: {e}")
        
        # Ground Truth 3D Model - Rotasyon parametrelerini kullan
        ax_gt = plt.subplot2grid((num_best_samples, 3), (row_idx, 2), projection='3d', facecolor=COLOR_PALETTE['background'])
        
        try:
            # Tüm rotasyonları bir arada uygula
            rotated_gt = sample['ground_truth'].copy().astype(np.float32)
            
            if hasattr(args, 'gt_rot_x') or hasattr(args, 'gt_rot_y') or hasattr(args, 'gt_rot_z'):
                rot_x = args.gt_rot_x if hasattr(args, 'gt_rot_x') else 0 
                rot_y = args.gt_rot_y if hasattr(args, 'gt_rot_y') else 0
                rot_z = args.gt_rot_z if hasattr(args, 'gt_rot_z') else 0
                
                try:
                    rotated_gt = rotate_voxel(rotated_gt, rot_x=rot_x, rot_y=rot_y, rot_z=rot_z)
                except Exception as e:
                    print(f"Ground truth model rotasyon hatası: {e}")
                    # Hata olursa orijinal veriyi kullan
                    rotated_gt = sample['ground_truth'].copy().astype(np.float32)
            
            # Threshold değerini düşürerek daha fazla voxel gösterilmesini sağla
            threshold = 0.5  # 0.5 yerine daha düşük
            gt_mask = rotated_gt > threshold
            
            ax_gt.voxels(gt_mask, facecolors=COLOR_PALETTE['ground_truth'], edgecolor='k', linewidth=0.2, alpha=0.8)
            ax_gt.set_title("Ground Truth 3D Model", 
                           fontsize=14, color=COLOR_PALETTE['text'], fontweight='bold')
            ax_gt.axis('off')
            ax_gt.view_init(elev=30, azim=45)
        except Exception as e:
            print(f"Ground truth model visualization error: {e}")
    
    plt.suptitle("SWAGE-3D: Best Performing Samples", fontsize=20, color=COLOR_PALETTE['text'], fontweight='bold', y=0.99)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    composite_path = os.path.join(results_dir, f"best_{num_best_samples}_samples_composite_{timestamp}.png")
    plt.savefig(composite_path, bbox_inches='tight', facecolor=COLOR_PALETTE['background'])
    plt.close()
    
    print(f"Composite visualization saved: {composite_path}")
    
    # Result summary
    print("\n" + "=" * 60)
    print(f"Completed Visualization of Top {num_best_samples} Samples")
    print("=" * 60)
    print(f"Total samples processed: {len(all_samples)}")
    print(f"Highest IoU score: {best_samples[0]['iou']:.4f}")
    print(f"Average IoU score (all test dataset): {sum(s['iou'] for s in all_samples)/len(all_samples):.4f}")
    print(f"Results saved to directory: {results_dir}")
    print(f"OBJ files saved to directory: {obj_dir}")
    print(f"RGB images saved to directory: {rgb_dir}")
    print(f"Depth images saved to directory: {depth_dir}")
    print("=" * 60)
    
    # Visualize IoU distribution
    plt.figure(figsize=(10, 6), facecolor=COLOR_PALETTE['background'])
    iou_vals = [s['iou'] for s in all_samples]
    n, bins, patches = plt.hist(iou_vals, bins=20, alpha=0.7, color=COLOR_PALETTE['generated'])
    
    # Color the histogram bars based on their values
    cm = plt.cm.get_cmap('viridis')
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    norm = plt.Normalize(min(bin_centers), max(bin_centers))
    for c, p in zip(bin_centers, patches):
        plt.setp(p, 'facecolor', cm(norm(c)))
    
    plt.axvline(best_samples[-1]['iou'], color=COLOR_PALETTE['ground_truth'], linestyle='--', linewidth=2,
                label=f'Top {num_best_samples} threshold: {best_samples[-1]["iou"]:.4f}')
    plt.xlabel('IoU Score', fontsize=12, fontweight='bold', color=COLOR_PALETTE['text'])
    plt.ylabel('Number of Samples', fontsize=12, fontweight='bold', color=COLOR_PALETTE['text'])
    plt.title('Test Dataset IoU Distribution - SWAGE-3D', 
             fontsize=14, fontweight='bold', color=COLOR_PALETTE['text'])
    plt.grid(alpha=0.3)
    plt.legend(frameon=True, fancybox=True, facecolor='white', framealpha=0.9, 
              fontsize=10)
    
    # Add stats as text annotations
    avg_iou = sum(iou_vals)/len(iou_vals)
    median_iou = np.median(iou_vals)
    std_iou = np.std(iou_vals)
    
    stats_text = f"Statistics:\nMean: {avg_iou:.4f}\nMedian: {median_iou:.4f}\nStd Dev: {std_iou:.4f}"
    plt.annotate(stats_text, xy=(0.02, 0.95), xycoords='axes fraction', 
                 fontsize=10, fontweight='bold',
                 bbox=dict(facecolor='white', alpha=0.8, boxstyle="round,pad=0.5"))
    
    # Save IoU histogram
    hist_path = os.path.join(results_dir, f"iou_distribution_{timestamp}.png")
    plt.savefig(hist_path, bbox_inches='tight', facecolor=COLOR_PALETTE['background'])
    plt.close()
    
    print(f"IoU distribution histogram saved: {hist_path}")
    
    return {
        'best_samples': best_samples,
        'all_samples': len(all_samples),
        'best_iou': best_samples[0]['iou'],
        'average_iou': sum(s['iou'] for s in all_samples)/len(all_samples),
        'obj_dir': obj_dir,  # OBJ dosyalarının yolunu da döndür
        'rgb_dir': rgb_dir,  # RGB görüntülerinin yolunu döndür
        'depth_dir': depth_dir  # Depth görüntülerinin yolunu döndür
    }
