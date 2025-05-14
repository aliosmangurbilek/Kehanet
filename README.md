deepnet/
├── core/                 # Temel sınıflar ve yardımcı fonksiyonlar
│   ├── tensor.py         # Tensor sınıfı ve işlemleri
│   ├── autograd.py       # Otomatik türev hesaplama
│   └── utils.py          # Yardımcı fonksiyonlar
├── layers/               # Katmanlar (örneğin, Dense, Conv2D)
│   ├── base.py           # Tüm katmanlar için temel sınıf
│   ├── dense.py          # Tam bağlantılı katman
│   ├── conv2d.py         # 2D evrişimli katman
│   └── activation.py     # Aktivasyon fonksiyonları
├── models/               # Model tanımlamaları
│   └── sequential.py     # Sıralı model sınıfı
├── optimizers/           # Optimizasyon algoritmaları
│   ├── sgd.py            # Stokastik gradyan inişi
│   └── adam.py           # Adam optimizasyonu
├── losses/               # Kayıp fonksiyonları
│   ├── mse.py            # Ortalama kare hatası
│   └── cross_entropy.py  # Çapraz entropi kaybı
├── data/                 # Veri işleme ve yükleme
│   ├── dataloader.py     # Veri yükleyici sınıfı
│   └── datasets.py       # Örnek veri setleri (MNIST vb.)
├── training/             # Eğitim döngüsü ve değerlendirme
│   ├── trainer.py        # Eğitim döngüsü
│   └── evaluator.py      # Model değerlendirme
├── tests/                # Birim testleri
│   ├── test_tensor.py    # Tensor sınıfı testleri
│   └── test_layers.py    # Katman testleri
├── examples/             # Örnek projeler ve kullanım senaryoları
│   ├── mnist_example.py  # MNIST veri seti ile örnek
│   └── cifar10_example.py# CIFAR-10 veri seti ile örnek
├── docs/                 # Belgeler ve kullanım kılavuzları
│   └── index.md          # Ana belge
├── requirements.txt      # Gerekli Python paketleri
├── setup.py              # Paket kurulum dosyası
└── README.md             # Proje açıklaması
