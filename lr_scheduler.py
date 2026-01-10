import torch
import math
from torch.optim.lr_scheduler import _LRScheduler, StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau

class WarmupLRScheduler(_LRScheduler):
    """
    Warmup learning rate scheduler. 
    İlk n_epochs boyunca öğrenme oranını doğrusal olarak artırır,
    ardından verilen başka bir scheduler'a geçer.
    """
    
    def __init__(self, optimizer, warmup_epochs, after_scheduler, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.after_scheduler = after_scheduler
        self.finished = False
        super(WarmupLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch >= self.warmup_epochs:
            if not self.finished:
                self.after_scheduler.base_lrs = self.base_lrs
                self.finished = True
            return self.after_scheduler.get_lr()

        return [base_lr * ((self.last_epoch + 1) / self.warmup_epochs) for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if self.finished and epoch is None:
            # Normal scheduler step
            self.after_scheduler.step(None)
            self._last_lr = self.after_scheduler._last_lr
        else:
            # Warmup step
            return super(WarmupLRScheduler, self).step(epoch)


def get_scheduler(optimizer, scheduler_type='multistep', milestones=None, gamma=0.5, 
                 patience=10, min_lr=1e-6, warmup_epochs=0, **kwargs):
    """
    Belirtilen tipte bir öğrenme oranı scheduler'ı oluşturur ve döndürür.
    
    Args:
        optimizer: Optimizer instance
        scheduler_type: 'multistep', 'exponential', 'cosine' veya 'plateau'
        milestones: MultiStepLR için epoch kilometre taşları 
        gamma: öğrenme oranı düşürme faktörü
        patience: ReduceLROnPlateau için bekleme epoch sayısı
        min_lr: minimum learning rate
        warmup_epochs: warmup epoch sayısı
        
    Returns:
        Bir scheduler nesnesi
    """
    if milestones is None:
        milestones = [50, 100, 150]
    
    # Ana scheduler'ı oluşturma
    if scheduler_type == 'multistep':
        scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    elif scheduler_type == 'exponential':
        scheduler = ExponentialLR(optimizer, gamma=gamma)
    elif scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=kwargs.get('T_max', 100), eta_min=min_lr)
    elif scheduler_type == 'plateau':
        scheduler = ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=gamma, 
            patience=patience,
            min_lr=min_lr,
            verbose=True
        )
    else:
        raise ValueError(f"Desteklenmeyen scheduler tipi: {scheduler_type}")
    
    # Warmup eklenecek mi kontrol et
    if warmup_epochs > 0:
        # ReduceLROnPlateau için warmup özel işlem gerektiriyor
        if scheduler_type == 'plateau':
            print("Uyarı: ReduceLROnPlateau ile warmup birlikte kullanılamaz. Warmup devre dışı bırakılıyor.")
            return scheduler
        else:
            return WarmupLRScheduler(optimizer, warmup_epochs, scheduler)
    
    return scheduler
