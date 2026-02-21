import torch
from torch import optim
from torch import nn
from collections import OrderedDict
from utils import make_hyparam_string, save_new_pickle, read_pickle, SavePloat_Voxels, generateZ
from utils import calculate_iou, calculate_metrics, calculate_accuracy_for_wasserstein
from utils import calculate_reconstruction_fscore, calculate_chamfer_distance_voxels
from utils import calculate_wasserstein_loss_d, calculate_wasserstein_loss_g  # WGAN loss fonksiyonları
import os
import time
import datetime
from torch.amp import autocast
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from utils import ShapeNetPlusImageDataset, var_or_cuda, visualize_comparative_results
from model import _G, _D, _E
from lr_scheduler import get_scheduler  # Yeni scheduler modülü

import numpy as np
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def test_3DVAEGAN(args):
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
    # datset define
    dsets_path = args.input_dir + args.data_dir + "test/"
    print(dsets_path)
    
    # Optimize edilmiş dataloader ayarları
    dsets = ShapeNetPlusImageDataset(dsets_path, args)
    dset_loaders = torch.utils.data.DataLoader(
        dsets, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=args.pin_memory, 
        prefetch_factor=args.prefetch_factor,
        persistent_workers=(args.num_workers > 0),
        drop_last=False  # Test için son batch'i koruyoruz
    )
    
    print(f"Test Dataloader: {args.num_workers} işçi ile optimize edilmiş veri yükleme aktif")

    # model define
    E = _E(args)
    G = _G(args)
    D =_D(args)
    G_solver = optim.Adam(G.parameters(), lr=args.g_lr, betas=args.beta)
    E_solver = optim.Adam(E.parameters(), lr=args.e_lr, betas=args.beta)
    D_solver= optim.Adam(D.parameters(), lr=args.d_lr, betas=args.beta)
    
    # Test sırasında scheduler kullanmıyoruz, yalnızca bilgi amaçlı ekliyoruz
    if args.use_scheduler:
        print(f"Not: Test modunda scheduler kullanılmıyor, {args.scheduler_type} scheduler eğitim için yapılandırıldı")
    
    criterion = nn.BCEWithLogitsLoss()  # Normal GAN için
    
    if torch.cuda.is_available():
        print("using cuda")
        G.cuda()
        E.cuda()
        D.cuda()

    pickle_path = args.output_dir + args.pickle_dir + log_param
    read_pickle(pickle_path, G, G_solver, D, D_solver, E, E_solver)
    
    # ÖNEMLİ: Modelleri eval() moduna al
    G.eval()
    E.eval()
    D.eval()
    
    # Metrik hesaplama için değişkenler
    total_recon_loss = 0
    total_iou = 0
    total_f_score = 0
    total_cd = 0
    total_d_loss = 0
    total_d_accuracy = 0
    batch_count = 0
    
    # Normal GAN metrikleri
    if not args.wasserstein:
        total_precision = 0
        total_recall = 0
        total_f1 = 0
    
    # Test zamanını al
    test_start_time = time.time()
    test_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Sonuçlar için .txt dosyası oluştur
    results_dir = f"{args.output_dir}/test_results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    results_file = f"{results_dir}/{args.model_name}_test_results_{test_date}.txt"
    
    # UTF-8 ile dosyayı aç
    with open(results_file, 'w', encoding='utf-8') as f:
        model_type = "Wasserstein 3D-VAE-GAN" if args.wasserstein else "3D-VAE-GAN"
        f.write(f"{model_type} Test Sonuçları - {test_date}\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("Batch | Recon Loss | IoU | F-Score | CD | D Loss | D Accuracy\n")
        f.write("-" * 80 + "\n")
    
    with torch.no_grad():  # Torch no_grad ile gradyanların hesaplanmasını önlüyoruz
        for i, (image, model_3d) in enumerate(dset_loaders):
            batch_start_time = time.time()
            
            # Non-blocking transferler ile GPU'ya veri aktarma
            if args.use_amp:
                X = model_3d.to(device='cuda', non_blocking=True)
                image = image.to(device='cuda', non_blocking=True)
            else:
                X = var_or_cuda(model_3d)
                image = var_or_cuda(image)
            
            # X'i doğru şekilde yeniden şekillendir [B, 1, D, H, W]
            if X.dim() == 4:
                X_reshaped = X.unsqueeze(1)
            else:
                X_reshaped = X.view(-1, 1, args.cube_len, args.cube_len, args.cube_len)

            # AMP kullanarak hesaplama
            with autocast(device_type='cuda', enabled=args.use_amp):
                # Gerçek model için discriminator sonuçları
                d_real = D(X_reshaped)
                
                # Encoder ve Generator ile yeniden oluşturma
                z_mu, z_var = E(image)
                Z_vae = E.reparameterize(z_mu, z_var)
                G_vae = G(Z_vae)
                
                # Yeniden oluşturulan model için discriminator sonuçları
                d_fake = D(G_vae)
                
                # Recon loss - MSE ile hesapla
                recon_loss = torch.mean(torch.sum(torch.pow((G_vae - X_reshaped), 2), dim=(1, 2, 3, 4)))
                
                # Discriminator loss hesaplama - model tipine göre
                if args.wasserstein:
                    d_loss = calculate_wasserstein_loss_d(d_real, d_fake)
                else:
                    real_labels = var_or_cuda(torch.ones(X.size(0)))
                    fake_labels = var_or_cuda(torch.zeros(X.size(0)))
                    d_real_loss = criterion(d_real.squeeze(), real_labels)
                    d_fake_loss = criterion(d_fake.squeeze(), fake_labels)
                    d_loss = d_real_loss + d_fake_loss
            
            # Rekonstrüksiyon metriklerini hesapla
            batch_iou = calculate_iou(G_vae, X_reshaped, threshold=args.voxel_threshold)
            batch_f_score = calculate_reconstruction_fscore(G_vae, X_reshaped, threshold=args.voxel_threshold)
            batch_cd = calculate_chamfer_distance_voxels(G_vae, X_reshaped, threshold=args.voxel_threshold)
            
            # Discriminator doğruluğunu hesapla
            d_accuracy = calculate_accuracy_for_wasserstein(d_real, d_fake, args.wasserstein)
            
            # Normal GAN için metrikler
            if not args.wasserstein:
                d_real_sigmoid = torch.sigmoid(d_real.squeeze())
                d_fake_sigmoid = torch.sigmoid(d_fake.squeeze())
                
                all_preds = torch.cat([d_real_sigmoid, d_fake_sigmoid])
                all_labels = torch.cat([real_labels, fake_labels])
                precision, recall, f1 = calculate_metrics(all_preds, all_labels)
            
            # Toplam değerlere ekle
            total_recon_loss += recon_loss.item()
            total_iou += batch_iou
            total_f_score += batch_f_score
            total_cd += batch_cd
            total_d_loss += d_loss.item()
            total_d_accuracy += d_accuracy.item()
            
            if not args.wasserstein:
                total_precision += precision
                total_recall += recall
                total_f1 += f1
                
            batch_count += 1
            
            batch_time = time.time() - batch_start_time
            
            # Batch sonuçlarını yazdır
            print(f"Batch {i+1} - IoU: {batch_iou:.4f}, F-Score: {batch_f_score:.4f}, CD: {batch_cd:.4f}, Time: {batch_time:.2f}s")
            
            with open(results_file, 'a', encoding='utf-8') as f:
                f.write(f"{i+1} | {recon_loss.item():.4f} | {batch_iou:.4f} | {batch_f_score:.4f} | {batch_cd:.4f} | {d_loss.item():.4f} | {d_accuracy:.4f}\n")
            
            # Görselleştirmek için ilk 5 batch'in görüntülerini kaydet
            if i < 5:
                # Görselleştirme klasörü
                vis_dir = os.path.join(args.output_dir, "test_visualizations")
                if not os.path.exists(vis_dir):
                    os.makedirs(vis_dir)
                
                # Batch'in ilk 4 örneğini görselleştir
                max_samples = min(4, X.size(0))
                
                for j in range(max_samples):
                    # Örnek veri hazırlama
                    input_image = image[j].cpu()  # 2D input görüntü (RGB+Depth)
                    reconstructed_voxel = G_vae[j].cpu().squeeze().numpy()  # Generator çıktısı
                    ground_truth_voxel = X_reshaped[j].cpu().squeeze().numpy()  # Ground truth
                    
                    # Görsel oluştur ve kaydet
                    fig_path = os.path.join(vis_dir, f"comparison_batch{i+1}_sample{j+1}.png")
                    visualize_comparative_results(
                        input_image, 
                        reconstructed_voxel, 
                        ground_truth_voxel,
                        save_path=fig_path,
                        title=f"Batch {i+1}, Sample {j+1} - IoU: {batch_iou:.4f}"
                    )
                
                # Sadece 3D voxel örneklerini kaydet (eskisi gibi)
                samples = G_vae.cpu().data[:8].squeeze().numpy()
                image_path = args.output_dir + args.image_dir + '3DVAEGAN_test'
                if not os.path.exists(image_path):
                    os.makedirs(image_path)
                SavePloat_Voxels(samples, image_path, i)
    
    # Ortalama metrikleri hesapla
    if batch_count > 0:
        avg_recon_loss = total_recon_loss / batch_count
        avg_iou = total_iou / batch_count
        avg_f_score = total_f_score / batch_count
        avg_cd = total_cd / batch_count
        avg_d_loss = total_d_loss / batch_count
        avg_d_accuracy = total_d_accuracy / batch_count
        
        if not args.wasserstein:
            avg_precision = total_precision / batch_count
            avg_recall = total_recall / batch_count
            avg_f1 = total_f1 / batch_count
    else:
        avg_recon_loss = 0
        avg_iou = 0
        avg_f_score = 0
        avg_cd = 0
        avg_d_loss = 0
        avg_d_accuracy = 0
        
        if not args.wasserstein:
            avg_precision = 0
            avg_recall = 0
            avg_f1 = 0
    
    # Toplam test süresini hesapla
    total_test_time = time.time() - test_start_time
    
    # Genel sonuçları yazdır
    print("\n" + "=" * 50)
    print("Test Sonuçları")
    print("=" * 50)
    print(f"Ortalama Recon Loss: {avg_recon_loss:.4f}")
    print(f"Ortalama IoU: {avg_iou:.4f}")
    print(f"Ortalama F-Score: {avg_f_score:.4f}")
    print(f"Ortalama Chamfer Distance: {avg_cd:.4f}")
    print(f"Ortalama D Loss: {avg_d_loss:.4f}")
    print(f"Ortalama D Accuracy: {avg_d_accuracy:.4f}")
    
    if not args.wasserstein:
        print(f"Ortalama Precision: {avg_precision:.4f}")
        print(f"Ortalama Recall: {avg_recall:.4f}")
        print(f"Ortalama F1 Score: {avg_f1:.4f}")
        
    print(f"Toplam Test Süresi: {total_test_time:.2f} saniye")
    print("=" * 50)
    
    # Genel sonuçları dosyaya yaz
    with open(results_file, 'a') as f:
        f.write("\n" + "=" * 50 + "\n")
        f.write("Test Sonuçları\n")
        f.write("=" * 50 + "\n")
        f.write(f"Ortalama Recon Loss: {avg_recon_loss:.4f}\n")
        f.write(f"Ortalama IoU: {avg_iou:.4f}\n")
        f.write(f"Ortalama F-Score: {avg_f_score:.4f}\n")
        f.write(f"Ortalama Chamfer Distance: {avg_cd:.4f}\n")
        f.write(f"Ortalama D Loss: {avg_d_loss:.4f}\n")
        f.write(f"Ortalama D Accuracy: {avg_d_accuracy:.4f}\n")
        
        if not args.wasserstein:
            f.write(f"Ortalama Precision: {avg_precision:.4f}\n")
            f.write(f"Ortalama Recall: {avg_recall:.4f}\n")
            f.write(f"Ortalama F1 Score: {avg_f1:.4f}\n")
            
        f.write(f"Test Edilen Batch Sayısı: {batch_count}\n")
        f.write(f"Toplam Test Süresi: {total_test_time:.2f} saniye\n")
        f.write(f"Test Tarihi: {test_date}\n")
        f.write("=" * 50 + "\n")
    
    print(f"\nSonuçlar {results_file} dosyasına kaydedildi.")
