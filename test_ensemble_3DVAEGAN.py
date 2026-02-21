import torch
from torch import optim
from torch import nn
from collections import OrderedDict
from utils import make_hyparam_string, save_new_pickle, read_pickle, SavePloat_Voxels, generateZ
from utils import calculate_iou, calculate_metrics, calculate_accuracy_for_wasserstein
from utils import calculate_reconstruction_fscore, calculate_chamfer_distance_voxels
from utils import calculate_wasserstein_loss_d, calculate_wasserstein_loss_g
import os
import time
import datetime
from torch.amp import autocast
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch.nn import functional as F

from utils import ShapeNetPlusImageDataset, var_or_cuda, visualize_comparative_results
from model import _G, _D, _E

np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def test_ensemble_3DVAEGAN(args):
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
                    ("sch", args.use_scheduler)]  # Scheduler parametresi eklendi

    hyparam_dict = OrderedDict(((arg, value) for arg, value in hyparam_list))
    log_param = make_hyparam_string(hyparam_dict)
    print(f"Test parametreleri: {log_param}")
    
    # Ensemble için kullanılacak checkpoint epokları
    try:
        ensemble_epochs = [int(e) for e in args.ensemble_epochs.split(',')]
    except:
        ensemble_epochs = [149]
    print(f"Ensemble için kullanılacak epoklar: {ensemble_epochs}")
    
    # Test dataset yükleme
    dsets_path = args.input_dir + args.data_dir + "test/"
    print(f"Test veri kümesi yolu: {dsets_path}")
    
    dsets = ShapeNetPlusImageDataset(dsets_path, args)
    dset_loaders = torch.utils.data.DataLoader(
        dsets, 
        batch_size=args.batch_size, 
        shuffle=False,  # Test için karıştırma kapalı (tutarlı sonuçlar için)
        num_workers=args.num_workers,
        pin_memory=args.pin_memory, 
        prefetch_factor=args.prefetch_factor,
        persistent_workers=(args.num_workers > 0),
        drop_last=False
    )
    
    print(f"Test Dataloader: {args.num_workers} işçi ile optimize edilmiş veri yükleme aktif")

    # Ensemble için modelleri yükleme
    ensemble_models = []
    pickle_base_path = args.output_dir + args.pickle_dir + log_param
    
    # Sonuçlar için klasör
    results_dir = f"{args.output_dir}/ensemble_results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Test zamanını al
    test_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_file = f"{results_dir}/ensemble_results_{test_date}{'_wgan' if args.wasserstein else ''}.txt"

    with open(results_file, 'w') as f:
        model_type = "Wasserstein 3D-VAE-GAN" if args.wasserstein else "3D-VAE-GAN"
        f.write(f"{model_type} Ensemble Test Sonuçları - {test_date}\n")
        f.write("=" * 60 + "\n")
        f.write(f"Ensemble için kullanılan epoklar: {ensemble_epochs}\n")
        f.write("=" * 60 + "\n\n")
    
    # Her epok için model oluştur ve yükle
    for epoch in ensemble_epochs:
        print(f"Epok {epoch} için model yükleniyor...")
        
        # Model instance'larını oluştur
        G = _G(args)
        D = _D(args)
        E = _E(args)
        
        # Optimizerları oluştur
        G_solver = optim.Adam(G.parameters(), lr=args.g_lr, betas=args.beta)
        D_solver = optim.Adam(D.parameters(), lr=args.d_lr, betas=args.beta)
        E_solver = optim.Adam(E.parameters(), lr=args.e_lr, betas=args.beta)
        
        # GPU'ya taşı
        if torch.cuda.is_available():
            G.cuda()
            D.cuda()
            E.cuda()
        
        # Model ağırlıklarını yükle
        try:
            with open(f"{pickle_base_path}/G_{epoch}.pkl", "rb") as f:
                G.load_state_dict(torch.load(f))
            with open(f"{pickle_base_path}/D_{epoch}.pkl", "rb") as f:
                D.load_state_dict(torch.load(f))
            with open(f"{pickle_base_path}/E_{epoch}.pkl", "rb") as f:
                E.load_state_dict(torch.load(f))
                
            # Eval moduna al
            G.eval()
            D.eval()
            E.eval()
            
            # Ensemble listesine ekle
            ensemble_models.append({"epoch": epoch, "G": G, "D": D, "E": E})
            print(f"Epok {epoch} için model başarıyla yüklendi")
        except Exception as e:
            print(f"Epok {epoch} için model yükleme hatası: {e}")
    
    if len(ensemble_models) == 0:
        print("Hiçbir model yüklenemedi! Test sonlandırılıyor.")
        return
    
    print(f"{len(ensemble_models)} adet model ensemble için hazır.")
    
    # Metrikleri takip etmek için değişkenler
    total_recon_loss = 0
    total_ensemble_iou = 0
    total_weighted_iou = 0
    total_ensemble_f_score = 0
    total_weighted_f_score = 0
    total_ensemble_cd = 0
    total_weighted_cd = 0
    total_best_iou = 0
    
    individual_ious = {model["epoch"]: 0 for model in ensemble_models}
    individual_f_scores = {model["epoch"]: 0 for model in ensemble_models}
    individual_counts = {model["epoch"]: 0 for model in ensemble_models}
    
    # Test verisi üzerinde değerlendirme
    batch_count = 0
    test_start_time = time.time()
    
    # Örnek görselleştirmeler için klasör
    visualization_dir = f"{results_dir}/visualizations"
    if not os.path.exists(visualization_dir):
        os.makedirs(visualization_dir)
    
    criterion = nn.BCEWithLogitsLoss()  # Normal GAN için
    
    print("Ensemble test başlatılıyor...")
    
    with torch.no_grad():
        for i, (image, model_3d) in enumerate(dset_loaders):
            batch_start_time = time.time()
            
            # GPU'ya veri transferi
            if args.use_amp:
                X = model_3d.to(device='cuda', non_blocking=True)
                image = image.to(device='cuda', non_blocking=True)
            else:
                X = var_or_cuda(model_3d)
                image = var_or_cuda(image)
            
            # X'i doğru şekilde yeniden şekillendir
            X_reshaped = X.view(-1, 1, args.cube_len, args.cube_len, args.cube_len)
            
            # Her model için ayrı tahminler yap
            all_reconstructions = []
            individual_batch_ious = {}
            best_batch_iou = 0
            best_model_idx = 0
            
            for idx, model_data in enumerate(ensemble_models):
                epoch = model_data["epoch"]
                G = model_data["G"]
                E = model_data["E"]
                
                with autocast(device_type='cuda', enabled=args.use_amp):
                    # Encoder ve Generator ile rekonstrüksiyon
                    z_mu, z_var = E(image)
                    Z_vae = E.reparameterize(z_mu, z_var)
                    G_vae = G(Z_vae)
                    
                    # IoU ve F-Score hesapla
                    batch_iou = calculate_iou(G_vae, X_reshaped, threshold=args.voxel_threshold)
                    batch_f_score = calculate_reconstruction_fscore(G_vae, X_reshaped, threshold=args.voxel_threshold)
                    
                    individual_batch_ious[epoch] = batch_iou
                    
                    # En iyi modeli izle
                    if batch_iou > best_batch_iou:
                        best_batch_iou = batch_iou
                        best_model_idx = idx
                    
                    # İstatistikleri güncelle
                    individual_ious[epoch] += batch_iou
                    individual_f_scores[epoch] += batch_f_score
                    individual_counts[epoch] += 1
                
                # Rekonstrüksiyonu kaydet
                all_reconstructions.append(G_vae)
            
            # Ensemble yaklaşımları
            # 1. Ortalama birleştirme (Simple Average Ensemble) - Bilimsel Standart
            ensemble_reconstruction = torch.mean(torch.stack(all_reconstructions), dim=0)
            
            # 2. Epok Ağırlıklı Birleştirme (Epoch-Weighted Ensemble)
            # Daha geç eğitilmiş modeller genellikle daha iyidir, bu yüzden onlara sabit daha fazla ağırlık verilir.
            # Bu yöntem GT (Ground Truth) gerektirmediği için bilimsel olarak geçerlidir.
            weights = torch.linspace(0.8, 1.2, len(ensemble_models)).to(device='cuda' if torch.cuda.is_available() else 'cpu')
            weights = weights / weights.sum()
            weighted_reconstruction = torch.sum(torch.stack([w * rec for w, rec in zip(weights, all_reconstructions)]), dim=0)
            
            # Metrikleri Hesapla (Thresholding sonrası)
            ensemble_iou = calculate_iou(ensemble_reconstruction, X_reshaped, threshold=args.voxel_threshold)
            ensemble_f_score = calculate_reconstruction_fscore(ensemble_reconstruction, X_reshaped, threshold=args.voxel_threshold)
            ensemble_cd = calculate_chamfer_distance_voxels(ensemble_reconstruction, X_reshaped, threshold=args.voxel_threshold)
            
            weighted_iou = calculate_iou(weighted_reconstruction, X_reshaped, threshold=args.voxel_threshold)
            weighted_f_score = calculate_reconstruction_fscore(weighted_reconstruction, X_reshaped, threshold=args.voxel_threshold)
            weighted_cd = calculate_chamfer_distance_voxels(weighted_reconstruction, X_reshaped, threshold=args.voxel_threshold)
            
            # Metrikleri güncelle
            total_ensemble_iou += ensemble_iou
            total_weighted_iou += weighted_iou
            total_ensemble_f_score += ensemble_f_score
            total_weighted_f_score += weighted_f_score
            total_ensemble_cd += ensemble_cd
            total_weighted_cd += weighted_cd
            total_best_iou += best_batch_iou
            batch_count += 1
            
            # Batch sonuçlarını yazdır
            print(f"Batch {i+1} - Average Ens. IoU: {ensemble_iou:.4f}, CD: {ensemble_cd:.4f}")
            
            # Sonuçları dosyaya yaz (UTF-8)
            with open(results_file, 'a', encoding='utf-8') as f:
                f.write(f"Batch {i+1}: Avg.IoU: {ensemble_iou:.4f}, Weighted.IoU: {weighted_iou:.4f}, CD: {ensemble_cd:.4f}\n")
    
    # Ortalama metrikleri hesapla
    if batch_count > 0:
        avg_ensemble_iou = total_ensemble_iou / batch_count
        avg_weighted_iou = total_weighted_iou / batch_count
        avg_ensemble_f_score = total_ensemble_f_score / batch_count
        avg_weighted_f_score = total_weighted_f_score / batch_count
        avg_ensemble_cd = total_ensemble_cd / batch_count
        avg_weighted_cd = total_weighted_cd / batch_count
        avg_best_iou = total_best_iou / batch_count
        
        # Her epok için ortalama metrikler
        avg_individual_ious = {epoch: total/individual_counts[epoch] for epoch, total in individual_ious.items()}
        avg_individual_f_scores = {epoch: total/individual_counts[epoch] for epoch, total in individual_f_scores.items()}
    
    # Genel sonuçları yazdır
    print("\n" + "=" * 60)
    print("Ensemble Final Sonuçları")
    print("-" * 60)
    print(f"Avg Ensemble  - IoU: {avg_ensemble_iou:.4f}, F1: {avg_ensemble_f_score:.4f}, CD: {avg_ensemble_cd:.4f}")
    print(f"Weighted Ens. - IoU: {avg_weighted_iou:.4f}, F1: {avg_weighted_f_score:.4f}, CD: {avg_weighted_cd:.4f}")
    print("=" * 60)
    
    # Sonuçları dosyaya kaydet
    with open(results_file, 'a') as f:
        f.write("\n" + "=" * 60 + "\n")
        f.write("Ensemble Final Sonuçları\n")
        f.write(f"Avg Ensemble  - IoU: {avg_ensemble_iou:.4f}, F1: {avg_ensemble_f_score:.4f}, CD: {avg_ensemble_cd:.4f}\n")
        f.write(f"Weighted Ens. - IoU: {avg_weighted_iou:.4f}, F1: {avg_weighted_f_score:.4f}, CD: {avg_weighted_cd:.4f}\n")
        f.write("=" * 60 + "\n")
    
    return {
        'ensemble_iou': avg_ensemble_iou,
        'weighted_iou': avg_weighted_iou,
        'individual_ious': avg_individual_ious
    }

def visualize_ensemble_results(individual_ious, ensemble_iou, weighted_iou, results_dir, test_date):
    """Ensemble test sonuçlarını görselleştirir"""
    
    # Grafik boyutunu ayarla
    plt.figure(figsize=(12, 6))
    
    # Epochs ve IoU değerlerini ayır
    epochs = list(individual_ious.keys())
    iou_values = list(individual_ious.values())
    
    # Bireysel model IoU değerlerini çiz
    plt.plot(epochs, iou_values, 'o-', label='Individual Models')
    
    # Ensemble IoU değerlerini çiz
    ensemble_x = [min(epochs), max(epochs)]
    plt.plot(ensemble_x, [ensemble_iou, ensemble_iou], 'r--', label='Average Ensemble')
    plt.plot(ensemble_x, [weighted_iou, weighted_iou], 'g--', label='Weighted Ensemble')
    
    # En yüksek IoU değerine sahip noktayı vurgula
    best_epoch = max(individual_ious, key=individual_ious.get)
    best_iou = individual_ious[best_epoch]
    plt.plot(best_epoch, best_iou, 'ro', markersize=10, label='Best Individual Model')
    
    # Grafik etiketleri ve başlığı
    plt.xlabel('Epoch')
    plt.ylabel('IoU Score')
    plt.title('Ensemble Test Results - IoU Comparison')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # x-ekseni için tam sayı tikleri
    plt.xticks(epochs)
    
    # Dosyaya kaydet
    plt.savefig(f"{results_dir}/ensemble_comparison_{test_date}.png", dpi=300, bbox_inches='tight')
    plt.close()
