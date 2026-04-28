import torch
from torch import optim
from torch import nn
import torch.multiprocessing as mp
from collections import OrderedDict
from utils import make_hyparam_string, save_new_pickle, read_pickle, SavePloat_Voxels, generateZ, calculate_iou, calculate_metrics
from utils import calculate_wasserstein_loss_d, calculate_wasserstein_loss_g, compute_gradient_penalty, calculate_accuracy_for_wasserstein
from utils import mixup_data, get_mixup_params
import utils
import os
import time
import matplotlib.gridspec as gridspec
import numpy as np
import matplotlib.pyplot as plt
from utils import ShapeNetPlusImageDataset, var_or_cuda
from model import _G, _D, _E
from lr_scheduler import get_scheduler  # Yeni eklenen scheduler modülü
from torch.amp import autocast, GradScaler
plt.switch_backend("TkAgg")

def train_vae(args):
    # PyTorch çoklu CPU işleme optimizasyonu
    if args.num_workers > 0:
        mp.set_start_method('spawn', force=True)  # Windows'da gerekli olabilir

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
    print(log_param)

    # AMP için GradScaler oluşturma
    scaler = GradScaler(enabled=args.use_amp)

    # for using tensorboard
    if args.use_tensorboard:
        import tensorflow as tf

        summary_writer = tf.summary.create_file_writer(args.output_dir + args.log_dir + log_param)

        def inject_summary(summary_writer, tag, value, step):
            with summary_writer.as_default():
                tf.summary.scalar(tag, value, step=step)

        inject_summary = inject_summary

    # Dataloader optimizasyonu
    dsets_path = args.input_dir + args.data_dir + "train/"
    print(dsets_path)
    dsets = ShapeNetPlusImageDataset(dsets_path, args)
    
    # Optimize edilmiş dataloader ayarları
    dset_loaders = torch.utils.data.DataLoader(
        dsets, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=(args.num_workers > 0),
        drop_last=True
    )

    # Veri yükleyici uyarısı
    print(f"Dataloader: {args.num_workers} işçi ile optimize edilmiş veri yükleme aktif")

    # model define
    D = _D(args)
    G = _G(args)
    E = _E(args)
    
    # Generator attention bilgisini göster
    if args.use_attention:
        print(f"Generator attention mekanizması etkin - Katman {args.attention_after} sonrasında uygulanıyor")

    # WGAN için optimizer ayarları
    if args.wasserstein:
        # WGAN önerilen parametreler: RMSProp veya Adam (düşük beta değerleri ile)
        D_solver = optim.Adam(D.parameters(), lr=args.d_lr, betas=(0.5, 0.9))
        G_solver = optim.Adam(G.parameters(), lr=args.g_lr, betas=(0.5, 0.9))
        E_solver = optim.Adam(E.parameters(), lr=args.e_lr, betas=(0.5, 0.9))
        print("Wasserstein GAN modunda çalışılıyor - optimizer ayarları güncellendi")
    else:
        # Geleneksel GAN optimizer ayarları
        D_solver = optim.Adam(D.parameters(), lr=args.d_lr, betas=args.beta)
        G_solver = optim.Adam(G.parameters(), lr=args.g_lr, betas=args.beta)
        E_solver = optim.Adam(E.parameters(), lr=args.e_lr, betas=args.beta)

    # Learning rate schedulers for all models
    if args.use_scheduler:
        # Milestone'ları string'den listeye çevir
        milestones = [int(m) for m in args.scheduler_milestones.split(',')]
        
        if args.d_scheduler:
            D_scheduler = get_scheduler(
                optimizer=D_solver,
                scheduler_type=args.scheduler_type,
                milestones=milestones,
                gamma=args.scheduler_gamma,
                patience=args.scheduler_patience,
                min_lr=args.scheduler_min_lr,
                warmup_epochs=args.scheduler_warmup
            )
            print(f"Discriminator için {args.scheduler_type} scheduler aktif")
        
        if args.g_scheduler:
            G_scheduler = get_scheduler(
                optimizer=G_solver,
                scheduler_type=args.scheduler_type,
                milestones=milestones,
                gamma=args.scheduler_gamma,
                patience=args.scheduler_patience,
                min_lr=args.scheduler_min_lr,
                warmup_epochs=args.scheduler_warmup
            )
            print(f"Generator için {args.scheduler_type} scheduler aktif")
        
        if args.e_scheduler:
            E_scheduler = get_scheduler(
                optimizer=E_solver,
                scheduler_type=args.scheduler_type,
                milestones=milestones,
                gamma=args.scheduler_gamma,
                patience=args.scheduler_patience,
                min_lr=args.scheduler_min_lr,
                warmup_epochs=args.scheduler_warmup
            )
            print(f"Encoder için {args.scheduler_type} scheduler aktif")
    else:
        print("Learning rate scheduler devre dışı bırakıldı")

    # Modelleri ve optimizerleri GPU'ya atama
    if torch.cuda.is_available():
        print(f"using cuda: {torch.cuda.get_device_name(0)}")
        
        # Bellek kullanımını optimize etme
        if args.optimize_memory:
            torch.cuda.empty_cache()
        
        D.cuda()
        G.cuda()
        E.cuda()

    # Loss fonksiyonları
    criterion = nn.BCEWithLogitsLoss()  # WGAN modunda kullanılmaz, sadece normal GAN için

    pickle_path = "." + args.pickle_dir + log_param
    read_pickle(pickle_path, G, G_solver, D, D_solver)

    # Plateau scheduler için en iyi metrikleri izleme
    if args.use_scheduler and args.scheduler_type == 'plateau':
        best_d_loss = float('inf')
        best_g_loss = float('inf')
        best_recon_loss = float('inf')

    for epoch in range(args.n_epochs):
        # GPU belleğini optimize et
        if args.optimize_memory and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Epok başlangıç zamanını kaydet
        epoch_start_time = time.time()
        
        # Epoch ortalamaları için değerleri sıfırlama
        epoch_d_real_loss = 0
        epoch_d_fake_loss = 0
        epoch_d_loss = 0
        epoch_g_loss = 0
        epoch_d_total_acu = 0
        epoch_recon_loss = 0
        epoch_kl_loss = 0
        epoch_iou = 0
        # Yeni metrikler için değişkenler
        epoch_precision = 0
        epoch_recall = 0
        epoch_f1 = 0
        # WGAN için yeni metrikler
        epoch_gp = 0  # gradient penalty
        batch_count = 0
        
        # WGAN için critic iteration sayısı
        n_critic = 5 if args.wasserstein else 1
        
        for i, (image, model_3d) in enumerate(dset_loaders):
            # Her bir float tensörünü half precision'a çevirme
            if args.use_amp:
                model_3d = model_3d.to(device='cuda', non_blocking=True)
                image = image.to(device='cuda', non_blocking=True)
            else:
                model_3d = var_or_cuda(model_3d)
                image = var_or_cuda(image)

            if model_3d.size()[0] != int(args.batch_size):
                continue

            # ============= Train the critic/discriminator =============#
            # Her n_critic iterasyonda bir, critic'i eğitiyoruz
            # Ama önce scaler'ı bir kez güncellemeliyiz
            if args.wasserstein:
                for critic_iter in range(n_critic):
                    with autocast(device_type='cuda', enabled=args.use_amp):
                        Z = generateZ(args)
                        
                        # MixUp parametrelerini bir kez üret
                        lam, indices = None, None
                        if (args.use_2d_mixup or args.use_3d_mixup or args.use_latent_mixup) and np.random.random() < args.mixup_prob:
                            lam, indices = utils.get_mixup_params(args.batch_size, args.mixup_alpha, device='cuda')
                        
                        # MixUp uygulamaları (aktif ise ve parametreler üretilmişse)
                        if args.use_2d_mixup and lam is not None:
                            image_mixed = utils.mixup_data(image, indices, lam, device='cuda')
                        else:
                            image_mixed = image
                        
                        model_3d_view = model_3d.view(-1, 1, args.cube_len, args.cube_len, args.cube_len)
                        if args.use_3d_mixup and lam is not None:
                            model_3d_mixed = utils.mixup_data(model_3d_view, indices, lam, device='cuda')
                        else:
                            model_3d_mixed = model_3d_view
                        
                        # Encoder çıktıları
                        z_mu, z_var = E(image_mixed)
                        
                        # Latent MixUp (aktif ise)
                        if args.use_latent_mixup and lam is not None:
                            z_mu_mixed = utils.mixup_data(z_mu, indices, lam, device='cuda')
                            z_var_mixed = utils.mixup_data(z_var, indices, lam, device='cuda')
                        else:
                            z_mu_mixed, z_var_mixed = z_mu, z_var
                        
                        Z_vae = E.reparameterize(z_mu_mixed, z_var_mixed)
                        G_vae = G(Z_vae)
                        
                        # Discriminator çıktıları
                        d_real = D(model_3d_mixed)
                        fake_prior = G(Z)
                        d_fake_p = D(fake_prior.detach())
                        d_fake_v = D(G_vae.detach())
                        
                        # Wasserstein loss hesaplama
                        d_loss_p = calculate_wasserstein_loss_d(d_real, d_fake_p)
                        d_loss_v = calculate_wasserstein_loss_d(d_real, d_fake_v)
                        d_loss = (d_loss_p + d_loss_v) / 2
                        
                        # Gradient penalty (WGAN-GP) ekle
                        if args.gradient_penalty:
                            # Spectral normalizasyon kullanılıyorsa daha düşük GP değeri kullan
                            effective_lambda = args.lambda_gp
                            if args.use_spectral_norm:
                                effective_lambda = 1.0  # SN ile GP ağırlığını önemli ölçüde azalt
                                
                            gp_p = compute_gradient_penalty(D, model_3d_mixed, fake_prior.detach(), device='cuda', lambda_gp=effective_lambda)
                            gp_v = compute_gradient_penalty(D, model_3d_mixed, G_vae.detach(), device='cuda', lambda_gp=effective_lambda)
                            d_loss = d_loss + (gp_p + gp_v) / 2
                            if critic_iter == 0:
                                epoch_gp += (gp_p.item() + gp_v.item()) / 2
                    
                    # ...
                    
                    # İlk iterasyonda metrikleri kaydet
                    if critic_iter == 0:
                        d_real_acu = torch.gt(d_real, 0).float()
                        d_fake_p_acu = torch.lt(d_fake_p, 0).float()
                        d_fake_v_acu = torch.lt(d_fake_v, 0).float()
                        d_total_acu = torch.mean(torch.cat((d_real_acu, d_fake_p_acu, d_fake_v_acu), 0))
                        epoch_d_loss += d_loss.item()
                        epoch_d_total_acu += d_total_acu.item()
            else:
                # Normal GAN için tek iterasyon eğitim
                with autocast(device_type='cuda', enabled=args.use_amp):
                    # ...
                    z_mu, z_var = E(image)
                    # ...
                    Z_vae = E.reparameterize(z_mu, z_var)
                    G_vae = G(Z_vae)

                    # Geleneksel GAN loss hesaplama
                    real_labels = var_or_cuda(torch.ones(args.batch_size))
                    fake_labels = var_or_cuda(torch.zeros(args.batch_size))

                    if args.soft_label:
                        real_labels = var_or_cuda(torch.Tensor(args.batch_size).uniform_(0.7, 1.0))
                        fake_labels = var_or_cuda(torch.Tensor(args.batch_size).uniform_(0, 0.3))

                    d_real = D(model_3d)
                    d_real_loss = criterion(d_real.squeeze(), real_labels)

                    fake_prior = G(Z)
                    d_fake_p = D(fake_prior.detach())
                    d_fake_p_loss = criterion(d_fake_p.squeeze(), fake_labels)

                    d_fake_v = D(G_vae.detach())
                    d_fake_v_loss = criterion(d_fake_v.squeeze(), fake_labels)

                    d_loss = d_real_loss + (d_fake_p_loss + d_fake_v_loss) / 2
                    
                    # Discriminator doğruluğunu hesaplama
                    d_real_acu = torch.ge(torch.sigmoid(d_real.squeeze()), 0.5).float()
                    d_fake_p_acu = torch.le(torch.sigmoid(d_fake_p.squeeze()), 0.5).float()
                    d_fake_v_acu = torch.le(torch.sigmoid(d_fake_v.squeeze()), 0.5).float()
                    d_total_acu = torch.mean(torch.cat((d_real_acu, d_fake_p_acu, d_fake_v_acu), 0))
                    
                    # Normal GAN için metrikleri kaydet
                    epoch_d_real_loss += d_real_loss.item()
                    epoch_d_fake_loss += (d_fake_p_loss.item() + d_fake_v_loss.item()) / 2
                    epoch_d_loss += d_loss.item()
                    epoch_d_total_acu += d_total_acu.item()

                # Normal GAN için discriminator eğitimi
                if d_total_acu <= args.d_thresh:
                    D.zero_grad()
                    scaler.scale(d_loss).backward()
                    scaler.step(D_solver)
                    scaler.update()

            # ============= Train the Encoder =============#
            with autocast(device_type='cuda', enabled=args.use_amp):
                # Recon loss: Batch başına ortalama MSE - model_3d_mixed kullan
                recon_loss = torch.mean(torch.sum(torch.pow((G_vae - model_3d_mixed), 2), dim=(1, 2, 3, 4)))
                
                # KL kaybını batch ortalaması alınarak hesaplama
                KLLoss = -0.5 * torch.mean(torch.sum(1 + z_var_mixed - torch.pow(z_mu_mixed, 2) - torch.exp(z_var_mixed), dim=1))
                
                # VAE β-parametresi ile iki kaybı dengeleme
                beta = 1.0  # Varsayılan beta değeri
                E_loss = recon_loss + beta * KLLoss

            # IoU değerini batch için hesaplama
            batch_iou = calculate_iou(G_vae, model_3d_mixed)
            
            E.zero_grad()
            # AMP ile gradient hesaplama ve optimizasyon
            scaler.scale(E_loss).backward()
            scaler.step(E_solver)
            scaler.update()
            
            # =============== Train the generator ===============#
            with autocast(device_type='cuda', enabled=args.use_amp):
                # Yeni örnekler oluşturulup, discriminator üzerinden geçiyor
                Z_new = generateZ(args)
                fake_prior = G(Z_new)
                d_fake_p = D(fake_prior)
                
                Z_vae_detached = Z_vae.detach()
                G_vae_new = G(Z_vae_detached)
                d_fake_v = D(G_vae_new)

                if args.wasserstein:
                    # Wasserstein loss for generator
                    gan_p_loss = calculate_wasserstein_loss_g(d_fake_p)
                    gan_v_loss = calculate_wasserstein_loss_g(d_fake_v)
                else:
                    # Normal GAN loss for generator
                    real_labels = var_or_cuda(torch.ones(args.batch_size))
                    gan_p_loss = criterion(d_fake_p.squeeze(), real_labels)
                    gan_v_loss = criterion(d_fake_v.squeeze(), real_labels)
                
                gan_loss = (gan_p_loss + gan_v_loss) / 2
                
                # VAE branch'ı ile yeniden oluşturma hatasını hesaplama - model_3d_mixed kullan
                recon_loss_new = torch.mean(torch.sum(torch.pow((G_vae_new - model_3d_mixed), 2), dim=(1, 2, 3, 4)))
                
                lambda_recon = 1.0  # İhtiyaca göre ayarlanabilir
                g_loss = gan_loss + lambda_recon * recon_loss_new

            # Generator eğitimi
            G.zero_grad()
            # AMP ile gradient hesaplama ve optimizasyon
            scaler.scale(g_loss).backward()
            scaler.step(G_solver)
            scaler.update()

            # Metrics hesaplama ve kaydetme
            with torch.no_grad():
                # Epoch metriklerini güncelle
                epoch_g_loss += g_loss.item()
                epoch_recon_loss += recon_loss.item()
                epoch_kl_loss += KLLoss.item()
                epoch_iou += batch_iou
                
                # Normal GAN için precision, recall, f1 hesapla
                if not args.wasserstein:
                    d_real_sigmoid = torch.sigmoid(d_real)
                    d_fake = (d_fake_p + d_fake_v) / 2
                    d_fake_sigmoid = torch.sigmoid(d_fake)
                    
                    all_preds = torch.cat([d_real_sigmoid.squeeze(), d_fake_sigmoid.squeeze()])
                    all_labels = torch.cat([real_labels, fake_labels])
                    precision, recall, f1 = calculate_metrics(all_preds, all_labels)
                    
                    epoch_precision += precision
                    epoch_recall += recall
                    epoch_f1 += f1
                
                batch_count += 1

            # Her 10 yinelemede bir bellek kullanımını raporla
            if args.optimize_memory and i % 10 == 0 and torch.cuda.is_available():
                print(f"GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB / {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

        # Epoch ortalaması hesaplama
        if batch_count > 0:
            epoch_d_loss /= batch_count
            epoch_g_loss /= batch_count
            epoch_d_total_acu /= batch_count
            epoch_recon_loss /= batch_count
            epoch_kl_loss /= batch_count
            epoch_iou /= batch_count
            
            if not args.wasserstein:
                epoch_d_real_loss /= batch_count
                epoch_d_fake_loss /= batch_count
                epoch_precision /= batch_count
                epoch_recall /= batch_count
                epoch_f1 /= batch_count
            
            if args.wasserstein and args.gradient_penalty:
                epoch_gp /= batch_count

        # Epok süresini hesapla
        epoch_duration = time.time() - epoch_start_time

        # =============== Step the schedulers ===============#
        if args.use_scheduler:
            # Plateau scheduler için metrikler
            if args.scheduler_type == 'plateau':
                if args.d_scheduler:
                    D_scheduler.step(epoch_d_loss)
                    # En iyi değeri güncelle
                    best_d_loss = min(best_d_loss, epoch_d_loss)
                
                if args.g_scheduler:
                    G_scheduler.step(epoch_g_loss)
                    # En iyi değeri güncelle
                    best_g_loss = min(best_g_loss, epoch_g_loss)
                
                if args.e_scheduler:
                    E_scheduler.step(epoch_recon_loss)
                    # En iyi değeri güncelle
                    best_recon_loss = min(best_recon_loss, epoch_recon_loss)
            else:
                # Diğer scheduler tipleri için normal adım
                if args.d_scheduler:
                    D_scheduler.step()
                
                if args.g_scheduler:
                    G_scheduler.step()
                
                if args.e_scheduler:
                    E_scheduler.step()
            
            # Güncel öğrenme oranlarını yazma
            current_lr_d = D_solver.param_groups[0]['lr']
            current_lr_g = G_solver.param_groups[0]['lr'] 
            current_lr_e = E_solver.param_groups[0]['lr']
            print(f"Güncel LR - D: {current_lr_d:.6f}, G: {current_lr_g:.6f}, E: {current_lr_e:.6f}")
            
            # Scheduler bilgilerini tensorboard'a kaydetme
            if args.use_tensorboard:
                info_lr = {
                    'lr/discriminator_lr': current_lr_d,
                    'lr/generator_lr': current_lr_g,
                    'lr/encoder_lr': current_lr_e
                }
                for tag, value in info_lr.items():
                    inject_summary(summary_writer, tag, value, epoch)

        # =============== logging each iteration ===============#
        iteration = int(G_solver.state_dict()['state'][G_solver.state_dict()['param_groups'][0]['params'][0]]['step'].item())
        if args.use_tensorboard:
            log_save_path = args.output_dir + args.log_dir + log_param
            if not os.path.exists(log_save_path):
                os.makedirs(log_save_path)

            info = {
                'loss/loss_D': epoch_d_loss,
                'loss/loss_G': epoch_g_loss,
                'loss/acc_D' : epoch_d_total_acu,
                'loss/loss_recon' : epoch_recon_loss,
                'loss/loss_kl' : epoch_kl_loss,
                'metrics/iou' : epoch_iou,
                'performance/epoch_duration': epoch_duration
            }
            
            # Normal GAN ise ek metrikler ekle
            if not args.wasserstein:
                info.update({
                    'loss/loss_D_R': epoch_d_real_loss,
                    'loss/loss_D_F': epoch_d_fake_loss,
                    'metrics/precision': epoch_precision,
                    'metrics/recall': epoch_recall,
                    'metrics/f1_score': epoch_f1,
                })
                
            # WGAN-GP ise gradient penalty metriğini ekle
            if args.wasserstein and args.gradient_penalty:
                info['loss/gradient_penalty'] = epoch_gp

            for tag, value in info.items():
                inject_summary(summary_writer, tag, value, epoch)

            summary_writer.flush()

        # =============== each epoch save model or save image ===============#
        if args.wasserstein:
            print('Epoch-{}, Iter-{}; Duration: {:.2f}s, IoU: {:.4f}, Recon_loss: {:.4f}, KLLoss: {:.4f}, D_loss: {:.4f}, G_loss: {:.4f}, D_acu: {:.4f}'.format(
                epoch,
                iteration,
                epoch_duration,
                epoch_iou,
                epoch_recon_loss,
                epoch_kl_loss,
                epoch_d_loss, 
                epoch_g_loss, 
                epoch_d_total_acu
            ))
            
            if args.gradient_penalty:
                print('Gradient Penalty: {:.4f}'.format(epoch_gp))
        else:
            print('Epoch-{}, Iter-{}; Duration: {:.2f}s, IoU: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}, Recon_loss: {:.4}, KLLoss: {:.4}, D_loss: {:.4}, G_loss: {:.4}, D_acu: {:.4}, D_lr: {:.4}'.format(
                epoch,
                iteration,
                epoch_duration,
                epoch_iou,
                epoch_precision,
                epoch_recall,
                epoch_f1,
                epoch_recon_loss,
                epoch_kl_loss,
                epoch_d_loss, 
                epoch_g_loss, 
                epoch_d_total_acu, 
                D_solver.state_dict()['param_groups'][0]["lr"]
            ))

        if (epoch + 1) % args.image_save_step == 0:
            samples = G_vae.cpu().data[:8].squeeze().numpy()

            image_path = args.output_dir + args.image_dir + log_param
            if not os.path.exists(image_path):
                os.makedirs(image_path)

            SavePloat_Voxels(samples, image_path, epoch)

        if (epoch + 1) % args.pickle_step == 0:
            pickle_save_path = args.output_dir + args.pickle_dir + log_param
            save_new_pickle(pickle_save_path, epoch, G, G_solver, D, D_solver, E, E_solver)

        if args.lrsh and not args.wasserstein:  # WGAN için genellikle sabit lr kullanırız
            try:
                D_scheduler.step()
            except Exception as e:
                print("fail lr scheduling", e)
