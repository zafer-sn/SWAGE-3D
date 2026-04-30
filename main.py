import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import argparse
from train_vae import train_vae
from test_3DVAEGAN import test_3DVAEGAN
from test_ensemble_3DVAEGAN import test_ensemble_3DVAEGAN
from test_best_samples_3DVAEGAN import test_best_samples_3DVAEGAN  # Newly added best samples test
import torch

def main(args):
    # Configure cuDNN optimizations
    if torch.cuda.is_available():
        # In training mode, use benchmark=True to select the fastest algorithms
        if not args.test:
            torch.backends.cudnn.benchmark = True  # Choose best algorithms for repeating input sizes
            torch.backends.cudnn.deterministic = False  # Turn off determinism for faster operation
            print("cuDNN benchmark mode enabled - Training will be accelerated")
        else:
            # In test mode, use deterministic=True for consistent results
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            print("cuDNN deterministic mode enabled - Testing will be consistent")
        
        # Optimize GPU memory settings
        torch.cuda.empty_cache()
        if args.optimize_memory:
            # Free up idle GPU memory
            torch.cuda.set_per_process_memory_fraction(0.9)  # Use 90% of GPU memory (helps prevent OOM errors)

    if args.test == False:        
        if args.alg_type == '3DVAEGAN':
            train_vae(args)
        
    else:
        if args.alg_type == '3DVAEGAN':
            if args.test_best_samples:
                print("TESTING BEST SAMPLES AND CREATING VISUALIZATIONS FOR PUBLICATION")
                test_best_samples_3DVAEGAN(args)
            elif args.use_ensemble:
                print("TESTING SWAGE-3D WITH ENSEMBLE APPROACH")
                test_ensemble_3DVAEGAN(args)
            else:
                print("TESTING SWAGE-3D")
                test_3DVAEGAN(args)

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Model Parmeters
    parser.add_argument('--n_epochs', type=float, default=150,
                        help='max epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='each batch size')
    parser.add_argument('--g_lr', type=float, default=0.0025,
                        help='generator learning rate')
    parser.add_argument('--e_lr', type=float, default=1e-4,
                        help='encoder learning rate')
    parser.add_argument('--d_lr', type=float, default=0.001,
                        help='discriminator learning rate')
    parser.add_argument('--beta', type=tuple, default=(0.5, 0.5),
                        help='beta for adam')
    parser.add_argument('--d_thresh', type=float, default=0.8,
                        help='for balance dsicriminator and generator')
    parser.add_argument('--z_size', type=float, default=200,
                        help='latent space size')
    parser.add_argument('--z_dis', type=str, default="norm", choices=["norm", "uni"],
                        help='uniform: uni, normal: norm')
    parser.add_argument('--bias', type=str2bool, default=False,
                        help='using cnn bias')
    parser.add_argument('--leak_value', type=float, default=0.2,
                        help='leakeay relu')
    parser.add_argument('--cube_len', type=float, default=32,
                        help='cube length')
    parser.add_argument('--image_size', type=float, default=224,
                        help='cube length')
    parser.add_argument('--obj', type=str, default="watercraft",
                        help='tranining dataset object category')
    parser.add_argument('--soft_label', type=str2bool, default=True,
                        help='using soft_label')
    parser.add_argument('--lrsh', type=str2bool, default=True,
                        help='for learning rate shecduler')

    # Learning Rate Scheduler parametreleri
    parser.add_argument('--use_scheduler', type=str2bool, default=True,
                        help='Learning rate scheduler kullan')
    parser.add_argument('--scheduler_type', type=str, default='multistep',
                        choices=['multistep', 'exponential', 'cosine', 'plateau'],
                        help='Learning rate scheduler tipi')
    parser.add_argument('--scheduler_gamma', type=float, default=0.5,
                        help='LR decay faktörü (multistep ve exponential için)')
    parser.add_argument('--scheduler_milestones', type=str, default='60,90,120',
                        help='LR azaltma noktaları, virgülle ayrılmış sayılar (multistep için)')
    parser.add_argument('--scheduler_patience', type=int, default=10,
                        help='LR düşürme öncesi bekleme epoch sayısı (plateau için)')
    parser.add_argument('--scheduler_min_lr', type=float, default=1e-6,
                        help='Minimum öğrenme oranı')
    parser.add_argument('--scheduler_warmup', type=int, default=0,
                        help='Warmup epoch sayısı')
    
    # Ensemble test parametreleri
    parser.add_argument('--use_ensemble', type=str2bool, default=True,
                        help='Ensemble test yaklaşımı kullan')
    parser.add_argument('--ensemble_epochs', type=str, default='129,139,149',
                        help='Ensemble için kullanılacak epoklar (virgülle ayrılmış)')
    
    # Ayrı scheduler parametreleri
    parser.add_argument('--g_scheduler', type=str2bool, default=True,
                        help='Generator için scheduler kullan')
    parser.add_argument('--d_scheduler', type=str2bool, default=True,
                        help='Discriminator için scheduler kullan')
    parser.add_argument('--e_scheduler', type=str2bool, default=True,
                        help='Encoder için scheduler kullan')

    # Attention mekanizması için yeni parametreler
    parser.add_argument('--use_attention', type=str2bool, default=True,
                        help='Generator için attention mekanizması kullan')
    parser.add_argument('--attention_after', type=int, default=3,
                        help='Hangi katmandan sonra attention uygulanacak (1-4)')
    
    # Wasserstein GAN parametreleri
    parser.add_argument('--wasserstein', type=str2bool, default=True,
                        help='Wasserstein GAN kullan')
    parser.add_argument('--gradient_penalty', type=str2bool, default=True,
                        help='WGAN-GP için gradient penalty kullan')
    parser.add_argument('--clip_value', type=float, default=0.01,
                        help='WGAN için weight clipping değeri (gradient penalty kullanılmıyorsa)')
    parser.add_argument('--n_critic', type=int, default=5,
                        help='WGAN için critic iterasyon sayısı')
    parser.add_argument('--lambda_gp', type=float, default=10.0,
                        help='WGAN-GP için gradient penalty ağırlığı (SN kullanılıyorsa otomatik olarak 1.0 değerine düşürülür)')

    # Spectral Normalization parametresi ekleme
    parser.add_argument('--use_spectral_norm', type=str2bool, default=True,
                        help='Discriminator için Spectral Normalization kullan')

    # dir parameters
    parser.add_argument('--output_dir', type=str, default="output",
                        help='output path')
    parser.add_argument('--input_dir', type=str, default='input',
                        help='input path')
    parser.add_argument('--pickle_dir', type=str, default='/pickle/',
                        help='input path')
    parser.add_argument('--log_dir', type=str, default='/log/',
                        help='for tensorboard log path save in output_dir + log_dir')
    parser.add_argument('--image_dir', type=str, default='/image/',
                        help='for output image path save in output_dir + image_dir')
    parser.add_argument('--data_dir', type=str, default='/watercraft/',
                        help='dataset load path')

    # step parameter
    parser.add_argument('--pickle_step', type=int, default=10,
                        help='pickle save at pickle_step epoch')
    parser.add_argument('--log_step', type=int, default=1,
                        help='tensorboard log save at log_step epoch')
    parser.add_argument('--image_save_step', type=int, default=10,
                        help='output image save at image_save_step epoch')

    # other parameters
    parser.add_argument('--alg_type', type=str, default='3DVAEGAN',
                        help='for test')
    parser.add_argument('--combine_type', type=str, default='mean',
                        help='for test')
    parser.add_argument('--num_views', type=int, default=12,
                        help='for test')

    parser.add_argument('--model_name', type=str, default="watercraft_3DVAEGAN",
                        help='this model name for save pickle, logs, output image path and if model_name contain V2 modelV2 excute')
    parser.add_argument('--use_tensorboard', type=str2bool, default=True,
                        help='using tensorboard logging')
    parser.add_argument('--test_iter', type=int, default=10,
                        help='test_epoch number')
    parser.add_argument('--test', type=str2bool, default=True,
                        help='for test')

    # Yeni optimizasyon parametreleri
    parser.add_argument('--num_workers', type=int, default=4,
                        help='number of data loading workers')
    parser.add_argument('--prefetch_factor', type=int, default=2,
                        help='prefetch factor for dataloader')
    parser.add_argument('--optimize_memory', type=str2bool, default=True,
                        help='optimize GPU memory usage')
    parser.add_argument('--pin_memory', type=str2bool, default=True,
                        help='pin memory in dataloader for faster GPU transfer')

    parser.add_argument('--use_amp', type=str2bool, default=True,
                        help='number of data loading workers')

    # MixUp parametreleri
    parser.add_argument('--use_2d_mixup', type=str2bool, default=True,
                        help='2D görüntülerde MixUp kullan')
    parser.add_argument('--use_3d_mixup', type=str2bool, default=False,
                        help='3D voxellerde MixUp kullan')
    parser.add_argument('--use_latent_mixup', type=str2bool, default=False,
                        help='Latent uzayda MixUp kullan')
    parser.add_argument('--mixup_alpha', type=float, default=0.2,
                        help='MixUp beta dağılımı parametresi')
    parser.add_argument('--mixup_prob', type=float, default=0.5,
                        help='MixUp uygulanma olasılığı')

    # Makalem için en iyi örnekleri test etme parametresi
    parser.add_argument('--test_best_samples', type=str2bool, default=True,
                        help='En iyi örnekleri test et ve makale için görselleştir')
    parser.add_argument('--num_best_samples', type=int, default=10,
                        help='Kaç tane en iyi örneği seçip görselleştireceğiz')
                        
    # 3D model rotasyon parametreleri
    parser.add_argument('--gen_rot_x', type=float, default=90,
                        help='Üretilen nesne için x ekseni rotasyonu (elev)')
    parser.add_argument('--gen_rot_y', type=float, default=0,
                        help='Üretilen nesne için y ekseni rotasyonu')
    parser.add_argument('--gen_rot_z', type=float, default=0,
                        help='Üretilen nesne için z ekseni rotasyonu (azim)')
    parser.add_argument('--gt_rot_x', type=float, default=90,
                        help='Gerçek nesne için x ekseni rotasyonu (elev)')
    parser.add_argument('--gt_rot_y', type=float, default=0,
                        help='Gerçek nesne için y ekseni rotasyonu')
    parser.add_argument('--gt_rot_z', type=float, default=0,
                        help='Gerçek nesne için z ekseni rotasyonu (azim)')

    args = parser.parse_args()
    main(args)

