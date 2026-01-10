import sys
import os
import numpy as np
import torch
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QSlider, QComboBox, 
                            QPushButton, QGridLayout, QSpinBox, QSplitter)
from PyQt5.QtCore import Qt, QTimer
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import rotate

# Model dosyalarını yüklemek için
from utils import ShapeNetPlusImageDataset
from model import _G, _D, _E

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100, projection=None):
        if projection:
            self.fig = Figure(figsize=(width, height), dpi=dpi, tight_layout=True)
            self.axes = self.fig.add_subplot(111, projection=projection)
        else:
            self.fig = Figure(figsize=(width, height), dpi=dpi, tight_layout=True)
            self.axes = self.fig.add_subplot(111)
        
        super(MplCanvas, self).__init__(self.fig)
        self.setParent(parent)

class RotationFinder(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        
        # Veri yükleme ve model için değişkenler
        self.dataset = None
        self.models = {}
        self.current_sample = 0
        self.current_sample_data = None
        
        # Renkler
        self.color_gen = '#3498db'  # Mavi
        self.color_gt = '#e74c3c'  # Kırmızı
        
        # Slider değişiklik bayrağı - sürekli güncelleme yapmayı önlemek için
        self.slider_changed = False
        
        # 3D Model yükleme
        self.load_model_and_data()
        
        # Her 500ms'de bir rotasyonu güncelle (bu değeri arttırdık - daha az kasa)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_rotation_if_changed)
        self.timer.start(500)  # 50ms yerine 500ms
    
    def initUI(self):
        self.setWindowTitle('SWAGE-3D: 3D Model Rotation Finder')
        self.setGeometry(100, 100, 1200, 800)
        
        # Ana widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Info Label
        info_label = QLabel("SWAGE-3D rotasyon değerleri bulucu. Sliderlar ile rotasyonu ayarlayın.")
        info_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(info_label)
        
        # Matplotlib figürü için widget
        plot_widget = QWidget()
        plot_layout = QGridLayout(plot_widget)
        
        # RGB görüntüsü
        self.rgb_canvas = MplCanvas(plot_widget, width=6, height=4)
        self.rgb_canvas.axes.set_title("RGB Input Image")
        plot_layout.addWidget(self.rgb_canvas, 0, 0, 1, 2)
        
        # Üretilen 3D Model
        self.gen_canvas = MplCanvas(plot_widget, width=5, height=5, projection='3d')
        self.gen_canvas.axes.set_title("Generated 3D Model")
        plot_layout.addWidget(self.gen_canvas, 1, 0)
        
        # Ground Truth 3D Model
        self.gt_canvas = MplCanvas(plot_widget, width=5, height=5, projection='3d')
        self.gt_canvas.axes.set_title("Ground Truth 3D Model")
        plot_layout.addWidget(self.gt_canvas, 1, 1)
        
        main_layout.addWidget(plot_widget, stretch=10)
        
        # Kontrol widget'ı
        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)
        
        # Sample seçimi
        sample_layout = QHBoxLayout()
        self.sample_spinner = QSpinBox()
        self.sample_spinner.setMinimum(0)
        self.sample_spinner.setMaximum(999)
        self.sample_spinner.valueChanged.connect(self.load_sample)
        sample_layout.addWidget(QLabel("Sample Index:"))
        sample_layout.addWidget(self.sample_spinner)
        
        self.next_button = QPushButton("Next Sample")
        self.next_button.clicked.connect(self.next_sample)
        sample_layout.addWidget(self.next_button)
        
        self.prev_button = QPushButton("Previous Sample")
        self.prev_button.clicked.connect(self.prev_sample)
        sample_layout.addWidget(self.prev_button)
        
        control_layout.addLayout(sample_layout)
        
        # Slider'lar için Grid
        slider_grid = QGridLayout()
        
        # Gen X
        self.gen_x_slider = self.create_slider_with_label(
            "Generated X (elev):", slider_grid, 0, 0, 0, 180, 10)
        
        # Gen Y
        self.gen_y_slider = self.create_slider_with_label(
            "Generated Y:", slider_grid, 1, 0, -180, 180, 0)
        
        # Gen Z
        self.gen_z_slider = self.create_slider_with_label(
            "Generated Z (azim):", slider_grid, 2, 0, 0, 360, 180)
        
        # GT X
        self.gt_x_slider = self.create_slider_with_label(
            "Ground Truth X (elev):", slider_grid, 0, 2, 0, 180, 10)
        
        # GT Y
        self.gt_y_slider = self.create_slider_with_label(
            "Ground Truth Y:", slider_grid, 1, 2, -180, 180, 0)
        
        # GT Z
        self.gt_z_slider = self.create_slider_with_label(
            "Ground Truth Z (azim):", slider_grid, 2, 2, 0, 360, 180)
        
        control_layout.addLayout(slider_grid)
        
        # Komut için metin kutusu
        self.command_label = QLabel()
        self.command_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.command_label.setWordWrap(True)
        self.command_label.setMinimumHeight(60)
        control_layout.addWidget(QLabel("Komut Satırı Parametreleri:"))
        control_layout.addWidget(self.command_label)
        
        main_layout.addWidget(control_widget, stretch=6)
        
        # İlk durumda komut satırını güncelle
        self.update_command_text()
    
    def create_slider_with_label(self, label_text, layout, row, col, min_val, max_val, default_val):
        """Bir slider ve değerini gösteren label oluşturur"""
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.setValue(default_val)
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setTickInterval((max_val - min_val) // 10)
        
        label = QLabel(label_text)
        value_label = QLabel(str(default_val))
        
        # valueChanged yerine sliderReleased sinyaline bağlanalım
        slider.valueChanged.connect(lambda val: value_label.setText(str(val)))
        slider.valueChanged.connect(lambda: self.set_slider_changed())
        slider.sliderReleased.connect(self.update_command_text)
        
        layout.addWidget(label, row, col)
        layout.addWidget(slider, row, col+1)
        layout.addWidget(value_label, row, col+1, 1, 1, Qt.AlignRight)
        
        return slider
    
    def set_slider_changed(self):
        """Slider değiştiyse bayrak ayarla"""
        self.slider_changed = True
    
    def update_command_text(self):
        """Komut satırı parametrelerini günceller"""
        command = (f"--gen_rot_x {self.gen_x_slider.value()} "
                  f"--gen_rot_y {self.gen_y_slider.value()} "
                  f"--gen_rot_z {self.gen_z_slider.value()} "
                  f"--gt_rot_x {self.gt_x_slider.value()} "
                  f"--gt_rot_y {self.gt_y_slider.value()} "
                  f"--gt_rot_z {self.gt_z_slider.value()}")
        
        self.command_label.setText(command)
    
    def rotate_voxel(self, voxel, rot_x=0, rot_y=0, rot_z=0):
        """3D voxel verisine rotasyon uygular"""
        rotated = np.copy(voxel)
        
        # X ekseni etrafında rotasyon (yukarı/aşağı eğim - pitch)
        if rot_x != 0:
            # (y,z) düzleminde rotasyon
            rotated = rotate(rotated, angle=rot_x, axes=(1, 2), reshape=False, order=0, mode='constant', cval=0.0)
        
        # Y ekseni etrafında rotasyon (sağa/sola dönme - yaw)
        if rot_y != 0:
            # (x,z) düzleminde rotasyon
            rotated = rotate(rotated, angle=rot_y, axes=(0, 2), reshape=False, order=0, mode='constant', cval=0.0)
        
        # Z ekseni etrafında rotasyon (saat yönü/tersi yukarıdan bakış - roll)
        if rot_z != 0:
            # (x,y) düzleminde rotasyon
            rotated = rotate(rotated, angle=rot_z, axes=(0, 1), reshape=False, order=0, mode='constant', cval=0.0)
        
        return rotated
    
    def load_model_and_data(self):
        """Model ve veriyi yükler"""
        try:
            # Argümanları manuel olarak oluştur
            class Args:
                def __init__(self):
                    # Model parametreleri
                    self.cube_len = 32
                    self.image_size = 224
                    self.z_size = 200
                    self.use_attention = True
                    self.attention_after = 3
                    self.bias = False
                    self.wasserstein = True
                    self.use_spectral_norm = True
                    self.soft_label = True
                    self.beta = (0.5, 0.5)
                    self.z_dis = "norm"
                    self.leak_value = 0.2
                    self.use_scheduler = True
                    
                    # Path'ler
                    base_dir = os.path.dirname(os.path.abspath(__file__))
                    # Eğik çizginin sonuna eklendiğinden emin olunuyor
                    self.data_dir = '/car/'
                    self.input_dir = 'input'
                    
                    # Model dizini için path ayarlaması
                    pattern = 'model=car_3DVAEGAN_cube=32'
                    self.model_dir = None
                    
                    # Doğrudan tanımlı bir model_dir yerine çalışma zamanında bulacağız
            
            args = Args()
            
            # Veri klasörünü doğru şekilde ayarla
            test_data_paths = [
                os.path.join(os.getcwd(), 'input', 'car', 'test'),
                os.path.join(os.path.dirname(os.getcwd()), 'input', 'car', 'test'),
                'input/car/test',
                '../input/car/test',
                '../../input/car/test',
            ]
            
            valid_data_path = None
            for path in test_data_paths:
                if os.path.exists(path):
                    valid_data_path = path
                    print(f"Test veri dizini bulundu: {path}")
                    break
            
            if valid_data_path is None:
                print("Veri dizini bulunamadı!")
                return
                
            # Model dizinini bul
            pickle_dirs = [
                os.path.join(os.getcwd(), 'output', 'pickle'),
                os.path.join(os.path.dirname(os.getcwd()), 'output', 'pickle'),
                'output/pickle',
                '../output/pickle',
                '../../output/pickle',
            ]
            
            model_dir = None
            for pickle_dir in pickle_dirs:
                if os.path.exists(pickle_dir):
                    print(f"Pickle dizini bulundu: {pickle_dir}")
                    for item in os.listdir(pickle_dir):
                        if 'model=car' in item and os.path.isdir(os.path.join(pickle_dir, item)):
                            model_dir = os.path.join(pickle_dir, item)
                            print(f"Model dizini bulundu: {model_dir}")
                            break
                    if model_dir:
                        break
            
            if not model_dir:
                print("Model dizini bulunamadı!")
                return
                
            args.model_dir = model_dir
                
            # Model oluştur
            print("Modeller oluşturuluyor...")
            self.models['E'] = _E(args)
            self.models['G'] = _G(args)
            self.models['D'] = _D(args)
            
            # GPU'ya taşı
            if torch.cuda.is_available():
                self.models['E'].cuda()
                self.models['G'].cuda()
                self.models['D'].cuda()
            
            # Model ağırlıklarını yükle
            from utils import read_pickle
            import torch.optim as optim
            
            G_solver = optim.Adam(self.models['G'].parameters(), lr=0.0025, betas=(0.5, 0.5))
            D_solver = optim.Adam(self.models['D'].parameters(), lr=0.001, betas=(0.5, 0.5))
            E_solver = optim.Adam(self.models['E'].parameters(), lr=1e-4, betas=(0.5, 0.5))
            
            print(f"Model yükleniyor: {args.model_dir}")
            
            try:
                from utils import read_pickle
                # Model dosyalarını bul
                files = os.listdir(args.model_dir)
                generator_files = [f for f in files if f.startswith("G_") and f.endswith(".pkl")]
                generator_epochs = [int(f.split('_')[-1].split('.')[0]) for f in generator_files]
                
                if not generator_epochs:
                    raise FileNotFoundError("Model ağırlıkları bulunamadı")
                    
                # En yüksek epoch değerini al
                latest_epoch = max(generator_epochs)
                print(f"En son epoch: {latest_epoch}")
                
                read_pickle(args.model_dir, self.models['G'], G_solver, 
                            self.models['D'], D_solver, 
                            self.models['E'], E_solver)
                print("Model başarıyla yüklendi!")
            except Exception as e:
                print(f"Model yüklenirken hata: {e}")
                raise
            
            # Modeli değerlendirme moduna al
            for model in self.models.values():
                model.eval()
            
            # ShapeNetPlusImageDataset sınıfını özel veri yükleme sınıfımızla değiştirelim
            class ModifiedShapeNetDataset:
                def __init__(self, data_path, args):
                    self.root = data_path + "/"  # Sonuna / ekleyerek test/ klasörün içindeki dosyalara erişimi sağla
                    self.data_path = data_path
                    self.args = args
                    self.image_size = int(args.image_size)
                    
                    # Tüm binvox dosyalarını bul
                    self.model_3d_files = []
                    for file in os.listdir(self.data_path):
                        if file.endswith('.binvox'):
                            self.model_3d_files.append(file)
                    
                    print(f"Toplam {len(self.model_3d_files)} adet binvox dosyası bulundu.")
                
                def __len__(self):
                    return len(self.model_3d_files)
                
                def __getitem__(self, idx):
                    from utils import getVolumeFromBinvox
                    import torch
                    import numpy as np
                    
                    model_3d_file = self.model_3d_files[idx]
                    
                    # Voxel dosya yolunu düzelt - tam yolu kullan
                    full_path = os.path.join(self.data_path, model_3d_file)
                    print(f"Voxel dosyası yükleniyor: {full_path}")
                    
                    try:
                        # 3D voxel modelini yükle
                        volume = np.array(getVolumeFromBinvox(full_path), dtype=np.float32).copy()
                        
                        # Voxel formatını düzenle
                        volume = volume.reshape((32, 32, 32))
                        model_3d = torch.FloatTensor(volume)
                        
                        # Fotoğraf yerine sadece sahte bir görüntü oluşturuyoruz
                        # Normalizasyon parametrelerine dikkat et
                        image = torch.zeros(4, 224, 224)
                        
                        # İlk 3 kanal (RGB) için rastgele değerler
                        image[0:3, :, :] = torch.rand(3, 224, 224) * 0.5 + 0.25
                        
                        # 4. kanal (derinlik) için rastgele değerler
                        image[3, :, :] = torch.rand(224, 224) * 0.3 + 0.2
                        
                        return image, model_3d
                    except Exception as e:
                        print(f"Dosya yüklenirken hata: {full_path}, Hata: {e}")
                        # Hata durumunda dummy veri döndür
                        dummy_image = torch.zeros(4, 224, 224)
                        dummy_model = torch.zeros(32, 32, 32)
                        return dummy_image, dummy_model
            
            # Test veri setini yükle
            self.dataset = ModifiedShapeNetDataset(valid_data_path, args)
            print(f"Veri seti başarıyla yüklendi! Toplam örnek sayısı: {len(self.dataset)}")
            
            # Spinner max değerini güncelle
            self.sample_spinner.setMaximum(len(self.dataset) - 1)
            
            # İlk örneği yükle
            self.load_sample(0)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Model veya veri yüklenirken hata: {e}")
    
    def load_sample(self, sample_idx):
        """Belirtilen indeksteki örneği yükler"""
        if self.dataset is None:
            return
        
        if sample_idx < 0 or sample_idx >= len(self.dataset):
            return
        
        self.current_sample = sample_idx
        self.sample_spinner.setValue(sample_idx)
        
        try:
            # Veriyi yükle
            image, model_3d = self.dataset[sample_idx]
            
            # Batch boyutu ekle
            image = image.unsqueeze(0)
            model_3d = model_3d.unsqueeze(0)
            
            # GPU'ya taşı
            if torch.cuda.is_available():
                image = image.cuda()
                model_3d = model_3d.cuda()
            
            # Modeli çalıştır
            with torch.no_grad():
                try:
                    z_mu, z_var = self.models['E'](image)
                    Z_vae = self.models['E'].reparameterize(z_mu, z_var)
                    G_vae = self.models['G'](Z_vae)
                    
                    # Voxel'leri CPU'ya taşı ve numpy array'e dönüştür
                    generated_voxel = G_vae[0].cpu().squeeze().numpy()
                    gt_voxel = model_3d.view(-1, 1, 32, 32, 32)[0].cpu().squeeze().numpy()
                    
                    # RGB görüntüsünü de al (rastgele görüntüler için)
                    rgb_img = image[0, :3].cpu()
                    
                    # Verileri sakla
                    self.current_sample_data = {
                        'generated': generated_voxel,
                        'ground_truth': gt_voxel,
                        'rgb_image': rgb_img.permute(1, 2, 0).numpy()
                    }
                    
                    # RGB görüntüsünü göster
                    self.rgb_canvas.axes.clear()
                    self.rgb_canvas.axes.imshow(self.current_sample_data['rgb_image'])
                    self.rgb_canvas.axes.set_title(f"Generated RGB Image (Sample #{sample_idx})")
                    self.rgb_canvas.axes.axis('off')
                    self.rgb_canvas.draw()
                    
                    # Rotasyonları güncelle
                    self.update_rotation()
                    
                except Exception as e:
                    print(f"Model çalıştırılırken hata: {e}")
                    traceback.print_exc()
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Örnek yüklenirken hata: {e}")
    
    def next_sample(self):
        """Bir sonraki örneği yükler"""
        if self.current_sample < len(self.dataset) - 1:
            self.load_sample(self.current_sample + 1)
    
    def prev_sample(self):
        """Bir önceki örneği yükler"""
        if self.current_sample > 0:
            self.load_sample(self.current_sample - 1)
    
    def update_rotation_if_changed(self):
        """Sadece slider değiştiğinde rotasyonu güncelle"""
        if self.slider_changed:
            self.update_rotation()
            self.slider_changed = False
    
    def update_rotation(self):
        """3D model görüntülerini günceller"""
        if self.current_sample_data is None:
            return
        
        # GUI kasmasın diye cursor'u değiştir
        QApplication.setOverrideCursor(Qt.WaitCursor)
        
        try:
            # Rotasyon değerlerini al
            gen_rot_x = self.gen_x_slider.value()
            gen_rot_y = self.gen_y_slider.value()
            gen_rot_z = self.gen_z_slider.value()
            
            gt_rot_x = self.gt_x_slider.value()
            gt_rot_y = self.gt_y_slider.value()
            gt_rot_z = self.gt_z_slider.value()
            
            # Düşük-çözünürlüklü önizleme için voxel ön işleme
            # Düşük çözünürlük için downsampling (her 2 voxel'den birini al)
            gen_preview = self.current_sample_data['generated'][::2, ::2, ::2]
            gt_preview = self.current_sample_data['ground_truth'][::2, ::2, ::2]
            
            # Üretilen model
            self.gen_canvas.axes.clear()
            
            # Şimdi rotasyonları sırayla uygulayalım
            gen_rotated = gen_preview.copy()
            if gen_rot_x != 0 or gen_rot_y != 0 or gen_rot_z != 0:
                gen_rotated = self.rotate_voxel(gen_rotated, 
                                               rot_x=gen_rot_x, 
                                               rot_y=gen_rot_y, 
                                               rot_z=gen_rot_z)
            
            gen_mask = gen_rotated > 0.5
            self.gen_canvas.axes.voxels(gen_mask, facecolors=self.color_gen, edgecolor='k', linewidth=0.5, alpha=0.7)
            
            # Temel görüş açısı olarak varsayılan açıyı kullanalım
            # view_init ile elev ve azim değil, sadece bakış açısını ayarlıyoruz
            self.gen_canvas.axes.view_init(elev=30, azim=30)  # Temel görüş açısı
            
            # View ayarlarını güncelle
            self.gen_canvas.axes.set_xlim(0, gen_rotated.shape[0])
            self.gen_canvas.axes.set_ylim(0, gen_rotated.shape[1])
            self.gen_canvas.axes.set_zlim(0, gen_rotated.shape[2])
            self.gen_canvas.axes.set_xticks([])
            self.gen_canvas.axes.set_yticks([])
            self.gen_canvas.axes.set_zticks([])
            self.gen_canvas.axes.set_title("Generated 3D Model")
            self.gen_canvas.axes.grid(False)
            
            # Gerçek model
            self.gt_canvas.axes.clear()
            
            # Benzer şekilde ground truth için de rotasyonları uygulayalım
            gt_rotated = gt_preview.copy()
            if gt_rot_x != 0 or gt_rot_y != 0 or gt_rot_z != 0:
                gt_rotated = self.rotate_voxel(gt_rotated,
                                              rot_x=gt_rot_x,
                                              rot_y=gt_rot_y,
                                              rot_z=gt_rot_z)
            
            gt_mask = gt_rotated > 0.5
            self.gt_canvas.axes.voxels(gt_mask, facecolors=self.color_gt, edgecolor='k', linewidth=0.5, alpha=0.7)
            
            # Temel görüş açısı
            self.gt_canvas.axes.view_init(elev=30, azim=30)  # Temel görüş açısı
            
            # View ayarlarını güncelle
            self.gt_canvas.axes.set_xlim(0, gt_rotated.shape[0])
            self.gt_canvas.axes.set_ylim(0, gt_rotated.shape[1])
            self.gt_canvas.axes.set_zlim(0, gt_rotated.shape[2])
            self.gt_canvas.axes.set_xticks([])
            self.gt_canvas.axes.set_yticks([])
            self.gt_canvas.axes.set_zticks([])
            self.gt_canvas.axes.set_title("Ground Truth 3D Model")
            self.gt_canvas.axes.grid(False)
            
            # Figürleri güncelle
            self.gen_canvas.draw()
            self.gt_canvas.draw()
            
        finally:
            # İşlem bittiğinde cursor'u geri al
            QApplication.restoreOverrideCursor()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = RotationFinder()
    window.show()
    sys.exit(app.exec_())
