import torch
from torch import optim
from torch import nn
from collections import OrderedDict
from utils import make_hyparam_string, save_new_pickle, read_pickle, SavePloat_Voxels, generateZ
from utils import calculate_iou, calculate_metrics, calculate_accuracy_for_wasserstein
from utils import calculate_wasserstein_loss_d, calculate_wasserstein_loss_g
import os
import time
import datetime
from torch.amp import autocast
import numpy as np
import matplotlib.pyplot as plt
import pickle
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
    ensemble_epochs = [129, 139, 149]  # 30-60-90 milestonelardan sonraki epoklar
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
    
    # Validation metriklerini yükle ve sabit ağırlıkları hesapla
    val_metrics_path = pickle_base_path + "/val_metrics.pkl"
    val_weights = []
    best_val_epoch = -1
    best_val_iou = -1
    
    if os.path.exists(val_metrics_path):
        with open(val_metrics_path, "rb") as f:
            val_metrics = pickle.load(f)
        
        # Ensemble için kullanılan epokların validation IoU değerlerini al
        ensemble_val_ious = []
        for epoch in ensemble_epochs:
            iou = val_metrics.get(epoch, 0)
            ensemble_val_ious.append(iou)
            if iou > best_val_iou:
                best_val_iou = iou
                best_val_epoch = epoch
        
        # Ağırlıkları hesapla
        total_val_iou = sum(ensemble_val_ious)
        if total_val_iou > 0:
            val_weights = [iou / total_val_iou for iou in ensemble_val_ious]
        else:
            val_weights = [1.0 / len(ensemble_epochs)] * len(ensemble_epochs)
            
        print(f"Validation tabanlı sabit ağırlıklar belirlendi: {dict(zip(ensemble_epochs, val_weights))}")
        print(f"Validation setine göre en iyi model: Epok {best_val_epoch} (IoU: {best_val_iou:.4f})")
    else:
        print("Uyarı: val_metrics.pkl bulunamadı! Eşit ağırlıklar kullanılacak.")
        val_weights = [1.0 / len(ensemble_epochs)] * len(ensemble_epochs)
        best_val_epoch = ensemble_epochs[-1]

    # En iyi validation modelinin indeksini bul
    best_val_model_idx = -1
    for idx, model_data in enumerate(ensemble_models):
        if model_data["epoch"] == best_val_epoch:
            best_val_model_idx = idx
            break

    # Metrikleri takip etmek için değişkenler
    total_recon_loss = 0
    total_ensemble_iou = 0
    total_weighted_iou = 0
    total_best_val_model_iou = 0
    individual_ious = {model["epoch"]: 0 for model in ensemble_models}
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
            
            for idx, model_data in enumerate(ensemble_models):
                epoch = model_data["epoch"]
                G = model_data["G"]
                E = model_data["E"]
                
                with autocast(device_type='cuda', enabled=args.use_amp):
                    # Encoder ve Generator ile rekonstrüksiyon
                    z_mu, z_var = E(image)
                    Z_vae = E.reparameterize(z_mu, z_var)
                    G_vae = G(Z_vae)
                    
                    # IoU hesapla
                    batch_iou = calculate_iou(G_vae, X_reshaped, threshold=0.5)
                    individual_batch_ious[epoch] = batch_iou
                    
                    # İstatistikleri güncelle
                    individual_ious[epoch] += batch_iou
                    individual_counts[epoch] += 1
                
                # Rekonstrüksiyonu kaydet
                all_reconstructions.append(G_vae)
            
            # Ensemble yaklaşımları
            # 1. Ortalama birleştirme (Average Ensemble)
            ensemble_reconstruction = torch.mean(torch.stack(all_reconstructions), dim=0)
            ensemble_iou = calculate_iou(ensemble_reconstruction, X_reshaped, threshold=0.5)
            
            # 2. Ağırlıklı ortalama - Validation IoU ile sabit ağırlıklandırılmış (Weighted Ensemble)
            weighted_reconstruction = torch.sum(torch.stack([w * rec for w, rec in zip(val_weights, all_reconstructions)]), dim=0)
            weighted_iou = calculate_iou(weighted_reconstruction, X_reshaped, threshold=0.5)
            
            # 3. Validation setine göre en iyi modelin performansı
            best_val_model_reconstruction = all_reconstructions[best_val_model_idx]
            best_val_model_batch_iou = individual_batch_ious[best_val_epoch]
            
            # Metrikleri güncelle
            total_ensemble_iou += ensemble_iou
            total_weighted_iou += weighted_iou
            total_best_val_model_iou += best_val_model_batch_iou
            batch_count += 1
            
            # Batch sonuçlarını yazdır
            batch_time = time.time() - batch_start_time
            print(f"Batch {i+1} - Ensemble IoU: {ensemble_iou:.4f}, Weighted IoU: {weighted_iou:.4f}, "
                  f"Best Val Model IoU: {best_val_model_batch_iou:.4f} (Epok {best_val_epoch}), "
                  f"Time: {batch_time:.2f}s")
            
            # Sonuçları dosyaya yaz
            with open(results_file, 'a') as f:
                f.write(f"Batch {i+1}: Ensemble IoU: {ensemble_iou:.4f}, Weighted IoU: {weighted_iou:.4f}, "
                        f"Best Val Model IoU: {best_val_model_batch_iou:.4f} (Epok {best_val_epoch})\n")
                for epoch, iou in individual_batch_ious.items():
                    f.write(f"  - Epok {epoch}: IoU = {iou:.4f}\n")
                f.write("\n")
            
            # İlk 5 batch için görselleştirme kaydet
            if i < 5:
                # Görselleştirme için örnek seç (batch'in ilk örneği)
                sample_idx = 0
                
                # Her model için ayrı görselleştirme
                for idx, model_data in enumerate(ensemble_models):
                    epoch = model_data["epoch"]
                    sample_reconstruction = all_reconstructions[idx][sample_idx].cpu().squeeze().numpy()
                    SavePloat_Voxels(
                        np.array([sample_reconstruction]), 
                        f"{visualization_dir}/batch_{i+1}_epoch_{epoch}", 
                        0
                    )
                
                # Ensemble sonuçları
                ensemble_sample = ensemble_reconstruction[sample_idx].cpu().squeeze().numpy()
                weighted_sample = weighted_reconstruction[sample_idx].cpu().squeeze().numpy()
                ground_truth = X_reshaped[sample_idx].cpu().squeeze().numpy()
                
                SavePloat_Voxels(
                    np.array([ground_truth, ensemble_sample, weighted_sample]), 
                    f"{visualization_dir}/batch_{i+1}_ensemble", 
                    0,
                    titles=["Ground Truth", "Average Ensemble", "Weighted Ensemble"]
                )
                
                # Görselleştirme klasörü
                comparative_vis_dir = os.path.join(visualization_dir, "comparative")
                if not os.path.exists(comparative_vis_dir):
                    os.makedirs(comparative_vis_dir)
                
                # Batch'in ilk 2 örneğini görselleştir (daha fazla bilgi gösteriliyor, bu nedenle daha az örnek)
                max_samples = min(2, X.size(0))
                
                for j in range(max_samples):
                    # Örnek veri hazırlama
                    input_image = image[j].cpu()  # 2D input görüntü
                    ensemble_voxel = ensemble_reconstruction[j].cpu().squeeze().numpy()  # Ensemble output
                    ground_truth_voxel = X_reshaped[j].cpu().squeeze().numpy()  # Ground truth
                    
                    # Görsel oluştur ve kaydet
                    fig_path = os.path.join(comparative_vis_dir, f"comparison_batch{i+1}_sample{j+1}.png")
                    
                    # Önce standart görselleştirme
                    visualize_comparative_results(
                        input_image, 
                        ensemble_voxel, 
                        ground_truth_voxel,
                        save_path=fig_path,
                        title=f"Ensemble - Batch {i+1}, Sample {j+1} - IoU: {ensemble_iou:.4f}"
                    )
                    
                    # Her model için ayrı görselleştirme yap
                    for idx, model_data in enumerate(ensemble_models):
                        epoch = model_data["epoch"]
                        model_output = all_reconstructions[idx][j].cpu().squeeze().numpy()
                        model_iou = individual_batch_ious[epoch]
                        
                        model_fig_path = os.path.join(comparative_vis_dir, f"comparison_batch{i+1}_sample{j+1}_model{epoch}.png")
                        visualize_comparative_results(
                            input_image,
                            model_output,
                            ground_truth_voxel,
                            save_path=model_fig_path,
                            title=f"Model Epoch {epoch} - Batch {i+1}, Sample {j+1} - IoU: {model_iou:.4f}"
                        )
    
    # Toplam test süresini hesapla
    total_test_time = time.time() - test_start_time
    
    # Ortalama metrikleri hesapla
    if batch_count > 0:
        avg_ensemble_iou = total_ensemble_iou / batch_count
        avg_weighted_iou = total_weighted_iou / batch_count
        avg_best_val_model_iou = total_best_val_model_iou / batch_count
        
        # Her epok için ortalama IoU
        avg_individual_ious = {epoch: total/individual_counts[epoch] if individual_counts[epoch] > 0 else 0 
                               for epoch, total in individual_ious.items()}
    else:
        avg_ensemble_iou = 0
        avg_weighted_iou = 0
        avg_best_val_model_iou = 0
        avg_individual_ious = {epoch: 0 for epoch in individual_ious.keys()}
    
    # Sonuçları yazdır
    print("\n" + "=" * 60)
    print("Ensemble Test Sonuçları (Sızıntısız/Leak-free)")
    print("=" * 60)
    print(f"Ensemble Modelleri: Epok {ensemble_epochs}")
    print(f"Ortalama Ortalama (Simple Average) Ensemble IoU: {avg_ensemble_iou:.4f}")
    print(f"Ortalama Validation-Weighted Ensemble IoU: {avg_weighted_iou:.4f}")
    print(f"En İyi Validation Modeli Performansı (Epok {best_val_epoch}): {avg_best_val_model_iou:.4f}")
    print("\nBireysel Model Sonuçları (Test Seti):")
    
    # En iyi testi modelini belirle (sadece bilgi amaçlı)
    best_test_epoch = max(avg_individual_ious, key=avg_individual_ious.get)
    best_single_test_iou = avg_individual_ious[best_test_epoch]
    
    for epoch, avg_iou in avg_individual_ious.items():
        print(f"  Epok {epoch}: Ortalama IoU = {avg_iou:.4f}" + 
              (" (Validation'a göre seçilen en iyi model)" if epoch == best_val_epoch else ""))
    
    print(f"\nToplam Test Süresi: {total_test_time:.2f} saniye")
    print(f"Test Edilen Batch Sayısı: {batch_count}")
    print("=" * 60)
    
    # Sonuçları dosyaya kaydet
    with open(results_file, 'a') as f:
        f.write("\n" + "=" * 60 + "\n")
        f.write("Ensemble Test Sonuçları (Sızıntısız)\n")
        f.write("=" * 60 + "\n")
        f.write(f"Ortalama Ortalama Ensemble IoU: {avg_ensemble_iou:.4f}\n")
        f.write(f"Ortalama Validation-Weighted Ensemble IoU: {avg_weighted_iou:.4f}\n")
        f.write(f"En İyi Validation Modeli Performansı (Epok {best_val_epoch}): {avg_best_val_model_iou:.4f}\n\n")
        f.write("Bireysel Model Sonuçları:\n")
        
        for epoch, avg_iou in avg_individual_ious.items():
            f.write(f"  Epok {epoch}: Ortalama IoU = {avg_iou:.4f}" + 
                  (" (En İyi Validation Modeli)" if epoch == best_val_epoch else "") + "\n")
        
        # Ensemble karşılaştırması
        f.write("\nEnsemble Karşılaştırması:\n")
        best_overall_iou = max(avg_ensemble_iou, avg_weighted_iou, avg_best_val_model_iou)
        if best_overall_iou == avg_weighted_iou:
            f.write("  En iyi sonuç: Validation-Weighted Ensemble\n")
        elif best_overall_iou == avg_ensemble_iou:
            f.write("  En iyi sonuç: Simple Average Ensemble\n")
        else:
            f.write(f"  En iyi sonuç: Tek model (Validation'da seçilen Epok {best_val_epoch})\n")
        
        f.write(f"\nToplam Test Süresi: {total_test_time:.2f} saniye\n")
        f.write(f"Test Edilen Batch Sayısı: {batch_count}\n")
        f.write(f"Test Tarihi: {test_date}\n")
        f.write("=" * 60 + "\n")
    
    print(f"\nSonuçlar {results_file} dosyasına kaydedildi.")
    print(f"Görselleştirmeler {visualization_dir} dizinine kaydedildi.")
    
    # Sonuçları görselleştir
    visualize_ensemble_results(avg_individual_ious, avg_ensemble_iou, avg_weighted_iou, results_dir, test_date)
    
    return {
        'ensemble_iou': avg_ensemble_iou,
        'weighted_iou': avg_weighted_iou,
        'best_val_model_iou': avg_best_val_model_iou,
        'individual_ious': avg_individual_ious,
        'best_val_epoch': best_val_epoch
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
