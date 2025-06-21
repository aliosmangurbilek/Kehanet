# MNIST Sınıflandırıcı Web Uygulaması

El yazısı rakamları yükleyip veya çizebileceğiniz, modelin tüm adımlarını (ham girdi, ön işleme, sınıflandırma olasılıkları) görselleştiren modern bir Flask tabanlı web uygulaması.

---

## Canlı Demo

[https://aliosmangurbilek.online](https://aliosmangurbilek.online)

---

## Özellikler

- **Çift Giriş:**  
  - Resim yükleme veya HTML5 canvas ile çizim
- **Ön İşleme Önizlemesi:**  
  - Orijinal ve 28×28 normalize edilmiş görüntü
- **Tahmin Sonuçları:**  
  - 0–9 arası olasılıkların animasyonlu bar grafiği, vurgulu en yüksek olasılık
- **Modern Arayüz:**  
  - Bootstrap 5, kart tabanlı tasarım, açıklayıcı tooltip’ler ve “Nasıl Çalışır?” modalı
- **Üretim Hazır Dağıtım:**  
  - Minimal Docker imajı, Gunicorn WSGI, non-root kullanıcı, sağlık kontrolleri, docker-compose desteği

---

## Gereksinimler

- Docker ≥ 20.10
- docker-compose ≥ 1.29

**Alternatif:**  
- Python ≥ 3.10  
- pip

---

## Kurulum ve Kullanım

### 1. Depoyu Klonlayın

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

### 2. Docker ile Başlatın

```bash
docker-compose up --build
```

Uygulama varsayılan olarak `http://localhost:5000` adresinde çalışacaktır.

### 3. Manuel (Docker’sız) Çalıştırmak için

```bash
pip install -r requirements.txt
python app.py
```

---
