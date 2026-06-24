# Klasifikasi Intensi Pembelian Pengunjung E-Commerce (Online Shoppers Purchasing Intention)

## Deskripsi Project

### Latar Belakang
Dalam industri e-commerce, memahami perilaku pengunjung situs web merupakan hal yang krusial untuk meningkatkan konversi penjualan. Tidak semua pengunjung yang mengakses sebuah situs web pada akhirnya melakukan transaksi pembelian. Oleh karena itu, diperlukan suatu pendekatan analitik yang dapat mengidentifikasi pola perilaku pengunjung guna memprediksi kemungkinan terjadinya transaksi (revenue) berdasarkan aktivitas mereka selama melakukan sesi kunjungan, seperti jumlah halaman yang diakses, durasi kunjungan, tingkat bounce rate, exit rate, hingga waktu kunjungan (bulan, hari kerja/akhir pekan).

### Tujuan Project
Project ini bertujuan untuk membangun model klasifikasi machine learning yang dapat memprediksi apakah seorang pengunjung situs web e-commerce akan melakukan transaksi pembelian (Revenue = True) atau tidak (Revenue = False), berdasarkan data perilaku kunjungan (clickstream behavior) yang tercatat dalam dataset.

### Target Klasifikasi
Variabel target yang diprediksi adalah **Revenue**, yaitu variabel biner yang menunjukkan:
- **True / 1** : Pengunjung melakukan transaksi pembelian
- **False / 0** : Pengunjung tidak melakukan transaksi pembelian

### Gambaran Umum Alur Pengerjaan
Project ini dikerjakan melalui beberapa tahapan utama, yaitu:
1. Import library yang dibutuhkan
2. Input dan pemeriksaan awal dataset
3. Exploratory Data Analysis (EDA) meliputi visualisasi distribusi target, matriks korelasi, histogram, dan boxplot
4. Data preprocessing (encoding, normalisasi, standardisasi)
5. Penanganan ketidakseimbangan kelas menggunakan SMOTE
6. Analisis feature importance menggunakan XGBoost dan seleksi fitur
7. Pembagian data training dan testing
8. Pembangunan dan pelatihan beberapa model klasifikasi
9. Evaluasi dan perbandingan performa antar model
10. Hyperparameter tuning pada model terbaik menggunakan RandomizedSearchCV
11. Analisis learning curve untuk mengevaluasi overfitting/underfitting

## Anggota Tim

| No | Nama | NIM |
|----|------|-----|
| 1 | Faza Hamdi Yogaswara | K1D024011 |
| 2 | Nayla Safira Aulia | K1D024012 |
| 3 | Naila Dwina Azzahra | K1D024020 |
| 4 | Muhammad Alief Badrudin | K1D024030 |
| 5 | Rahma Aulia Fatma | K1D024032 |

## Dataset

- **Nama Dataset**: Online Shoppers Intention (`online_shoppers_intention.csv`)
- **Sumber Data**: Dataset dimuat melalui Google Colab yang terhubung dengan Google Drive (`/content/drive/MyDrive/ML/online_shoppers_intention.csv`)
- **Jumlah Observasi**: 12.330 baris data
- **Jumlah Variabel (awal)**: 18 kolom

### Daftar Variabel
| Variabel | Keterangan |
|----------|------------|
| Administrative | Jumlah halaman administratif yang dikunjungi |
| Administrative_Duration | Durasi kunjungan pada halaman administratif |
| Informational | Jumlah halaman informasi yang dikunjungi |
| Informational_Duration | Durasi kunjungan pada halaman informasi |
| ProductRelated | Jumlah halaman produk yang dikunjungi |
| ProductRelated_Duration | Durasi kunjungan pada halaman produk |
| BounceRates | Tingkat pengunjung yang langsung meninggalkan situs |
| ExitRates | Tingkat keluar dari halaman situs |
| PageValues | Nilai halaman yang dikunjungi |
| SpecialDay | Kedekatan waktu kunjungan dengan hari spesial |
| Month | Bulan kunjungan |
| OperatingSystems | Sistem operasi yang digunakan pengunjung |
| Browser | Jenis browser yang digunakan |
| Region | Wilayah asal pengunjung |
| TrafficType | Jenis sumber trafik |
| VisitorType | Tipe pengunjung (New/Returning/Other) |
| Weekend | Indikator kunjungan pada akhir pekan |
| **Revenue** | **Variabel target** — status terjadinya transaksi |

### Karakteristik Dataset
- Dataset **tidak memiliki missing value** pada seluruh 18 variabel (dikonfirmasi melalui `df.isna().sum()`)
- Dataset **tidak memiliki nilai negatif** pada variabel numerik
- Tipe data awal terdiri dari 7 variabel `int64`, 7 variabel `float64`, 2 variabel `object` (Month, VisitorType), dan 2 variabel `bool` (Weekend, Revenue)
- Distribusi variabel target **tidak seimbang (imbalanced)**:
  - Revenue = False: 10.422 observasi (84,53%)
  - Revenue = True: 1.908 observasi (15,47%)
- Ukuran memori dataset sekitar 1,5 MB

## Metodologi

### 1. Import Library
Mengimpor seluruh library yang diperlukan untuk pengolahan data (Pandas, NumPy), visualisasi (Matplotlib, Seaborn), dan pemodelan machine learning (Scikit-learn).

### 2. Input Data
Dataset dimuat dari Google Drive melalui Google Colab, kemudian dibaca menggunakan `pandas.read_csv()` dan diperiksa menggunakan `df.head()`, `df.info()`, serta `df.isna().sum()`.

### 3. Exploratory Data Analysis (EDA)
- Visualisasi distribusi variabel target (`countplot`) untuk melihat proporsi kelas Revenue
- Visualisasi matriks korelasi antar variabel numerik (`heatmap`)
- Visualisasi histogram untuk seluruh variabel numerik, baik secara umum maupun dikelompokkan berdasarkan Revenue
- Visualisasi boxplot untuk mendeteksi keberadaan outlier pada setiap variabel numerik

### 4. Data Preprocessing
- **Encoding**: 
  - `LabelEncoder` digunakan untuk mengubah variabel `Weekend` dan `Revenue` (boolean) menjadi numerik biner
  - `pd.get_dummies()` (One-Hot Encoding) digunakan untuk variabel kategorik `Month` dan `VisitorType`, sehingga jumlah variabel bertambah dari 18 menjadi 29
- **Penanganan Missing Value**: Tidak diperlukan karena dataset tidak memiliki missing value pada seluruh tahap pemeriksaan
- **Feature Selection**: Berdasarkan hasil feature importance dari XGBoost, variabel `Browser`, `OperatingSystems`, `Region`, dan `TrafficType` dihapus karena memiliki kontribusi yang relatif rendah terhadap prediksi
- **Normalisasi**: Menggunakan `MinMaxScaler` pada variabel ProductRelated, ProductRelated_Duration, PageValues, BounceRates, dan ExitRates untuk menyamakan rentang skala ke 0–1
- **Standardisasi**: Dilakukan secara manual (z-score) pada variabel numerik kontinu (Administrative, Administrative_Duration, Informational, Informational_Duration, ProductRelated, ProductRelated_Duration, BounceRates, ExitRates, PageValues, SpecialDay)
- **Penanganan Imbalance Data**: Menggunakan **SMOTE (Synthetic Minority Over-sampling Technique)** sehingga distribusi kelas Revenue menjadi seimbang (50% : 50%), dengan total data setelah SMOTE menjadi 20.844 sampel (10.422 data per kelas)

### 5. Pembagian Data Training dan Testing
Data hasil SMOTE dibagi menggunakan `train_test_split` dengan rasio 80:20 dan `stratify` pada variabel target, menghasilkan:
- Data Training: 16.675 sampel
- Data Testing: 4.169 sampel

### 6. Pembangunan Model
Lima algoritma klasifikasi dilatih dan dievaluasi pada data yang sama, kemudian model terbaik (Random Forest) dioptimasi lebih lanjut menggunakan `RandomizedSearchCV`.

### 7. Evaluasi Model
Evaluasi dilakukan menggunakan **accuracy**, **precision**, **recall**, **f1-score**, dan **classification report**, baik pada data training maupun data testing, untuk mengidentifikasi indikasi overfitting/underfitting pada masing-masing model. Analisis tambahan dilakukan melalui **learning curve** pada model terbaik hasil tuning.

## Model yang Digunakan

| Model | Penjelasan Singkat |
|-------|---------------------|
| **Logistic Regression** | Model linear yang memprediksi probabilitas kelas menggunakan fungsi logistik, cocok sebagai baseline model klasifikasi biner |
| **Decision Tree Classifier** | Model berbasis struktur pohon keputusan yang membagi data berdasarkan aturan-aturan fitur secara hierarkis |
| **Random Forest Classifier** | Model ensemble berbasis kumpulan (forest) decision tree yang menggabungkan hasil prediksi banyak pohon untuk meningkatkan akurasi dan mengurangi overfitting |
| **Gradient Boosting (XGBoost Classifier)** | Model ensemble berbasis boosting yang membangun pohon secara bertahap dengan memperbaiki kesalahan model sebelumnya, serta digunakan juga untuk analisis feature importance |
| **Support Vector Machine (SVM)** | Model klasifikasi yang mencari hyperplane optimal untuk memisahkan kelas data pada ruang fitur |

Selain kelima model di atas, project juga memanfaatkan **K-Nearest Neighbor (KNN)** sebagai salah satu library yang diimpor untuk mendukung proses klasifikasi, meskipun evaluasi performa secara eksplisit pada laporan difokuskan pada kelima model di atas.

## Cara Menjalankan Project

### 1. Clone Repository
```bash
git clone https://github.com/username/online-shoppers-intention-classification.git
cd online-shoppers-intention-classification
```

### 2. Install Dependencies
```bash
pip install pandas numpy seaborn matplotlib scikit-learn xgboost imbalanced-learn
```

### 3. Menyiapkan Dataset
Letakkan file dataset `online_shoppers_intention.csv` pada folder `data/`, atau sesuaikan path dataset pada notebook jika menjalankan melalui Google Colab dengan Google Drive:
```python
data_path = '/content/drive/MyDrive/ML/online_shoppers_intention.csv'
df = pd.read_csv(data_path)
```

### 4. Menjalankan Notebook/Script
```bash
jupyter notebook notebook/Tugas_Klasifikasi_Kelompok_F.ipynb
```
Atau jika dijalankan di Google Colab, unggah notebook dan dataset, kemudian jalankan seluruh sel secara berurutan (Run All).

## Struktur Folder Project

```
online-shoppers-intention-classification/
│
├── data/
│   └── online_shoppers_intention.csv
│
├── notebook/
│   └── Tugas_Klasifikasi_Kelompok_F.ipynb
│
├── README.md
└── requirements.txt
```

## Hasil dan Temuan Utama

### Kondisi Dataset
- Dataset bersih, tanpa missing value dan tanpa nilai negatif pada variabel numerik
- Sebagian besar variabel numerik bersifat right-skewed (miring ke kanan), seperti Administrative, ProductRelated, dan PageValues

### Ketidakseimbangan Kelas
- Sebelum SMOTE: Revenue = False sebanyak 84,53%, Revenue = True sebanyak 15,47%
- Setelah penerapan SMOTE, distribusi kelas menjadi seimbang sempurna (50% : 50%) dengan total 20.844 sampel

### Variabel yang Paling Berpengaruh
Berdasarkan analisis feature importance menggunakan XGBoost, variabel dengan kontribusi tertinggi terhadap prediksi Revenue adalah:
1. **PageValues** (importance gain tertinggi sebesar 0,2307)
2. **ProductRelated_Duration** (skor importance 403)
3. **ExitRates** (skor importance 402)
4. **ProductRelated** (skor importance 344)
5. **Administrative_Duration** (skor importance 314)

Selain itu, berdasarkan matriks korelasi terhadap Revenue, **PageValues** memiliki korelasi positif tertinggi (0,49), diikuti oleh ProductRelated (0,16) dan ProductRelated_Duration (0,15). Sebaliknya, ExitRates (-0,21) dan BounceRates (-0,15) menunjukkan korelasi negatif terhadap Revenue.

### Model dengan Performa Terbaik
Berdasarkan perbandingan akurasi pada data testing:

| Model | Akurasi Testing |
|-------|-----------------|
| **Random Forest Classifier** | **0,9319** |
| XGBoost Classifier | 0,9249 |
| Decision Tree Classifier | 0,8887 |
| Logistic Regression | 0,8671 |
| SVM Classifier | 0,6085 |

**Random Forest Classifier** terpilih sebagai model terbaik dengan akurasi 93,19% dan recall kelas Revenue=True sebesar 94,53%. Setelah dilakukan hyperparameter tuning menggunakan `RandomizedSearchCV` (parameter terbaik: `n_estimators=490`, `max_depth=None`, `min_samples_split=3`, `min_samples_leaf=1`), akurasi model pada data testing menjadi 92,73% dengan recall 94,05% — menunjukkan penurunan tipis (kurang dari 1%) dibandingkan model awal, sehingga konfigurasi default sudah cukup optimal untuk dataset ini.

### Insight Utama
- Model Random Forest dan Decision Tree menunjukkan indikasi **overfitting** yang kuat (akurasi training mendekati 100%), namun Random Forest tetap mampu menjaga generalisasi yang baik pada data testing
- Model SVM menunjukkan **underfitting**, dengan akurasi training dan testing yang rendah dan hampir identik (~61%)
- Model Logistic Regression menunjukkan performa paling stabil dengan selisih akurasi training dan testing yang sangat tipis (sekitar 1%), mengindikasikan generalisasi yang baik tanpa overfitting
- Analisis learning curve pada model Random Forest hasil tuning menunjukkan kurva cross-validation yang masih meningkat secara bertahap (belum mencapai plateau), sehingga penambahan data atau regularisasi lebih lanjut berpotensi meningkatkan performa model

## Library yang Digunakan

- `pandas` — manipulasi dan pengelolaan dataset
- `numpy` — operasi komputasi numerik
- `seaborn` — visualisasi data statistik
- `matplotlib` — visualisasi data secara umum
- `scikit-learn` — meliputi:
  - `train_test_split`, `StandardScaler`, `MinMaxScaler`, `LabelEncoder`
  - `LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier`, `GradientBoostingClassifier`
  - `SVC`, `KNeighborsClassifier`
  - `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `confusion_matrix`, `classification_report`
  - `RandomizedSearchCV`, `learning_curve`
- `xgboost` — algoritma XGBClassifier dan analisis feature importance
- `imbalanced-learn (imblearn)` — penerapan SMOTE untuk penanganan imbalanced data
- `scipy` — distribusi parameter pada proses hyperparameter tuning (`randint`)

## Kesimpulan

Berdasarkan keseluruhan proses analisis dan pemodelan yang telah dilakukan, dapat disimpulkan bahwa perilaku pengunjung situs e-commerce, khususnya terkait nilai halaman yang dikunjungi (PageValues), durasi dan jumlah halaman produk yang diakses (ProductRelated, ProductRelated_Duration), serta tingkat keluar dari situs (ExitRates), merupakan faktor-faktor yang paling berpengaruh dalam menentukan apakah seorang pengunjung akan melakukan transaksi pembelian. Ketidakseimbangan kelas pada variabel target berhasil ditangani secara efektif menggunakan teknik SMOTE, sehingga proses pemodelan dapat berjalan lebih objektif dalam mengenali pola dari kedua kelas.

Dari kelima algoritma klasifikasi yang diuji, **Random Forest Classifier** terbukti memberikan performa terbaik dengan akurasi sebesar 93,19% pada data testing, mengungguli XGBoost, Decision Tree, Logistic Regression, dan SVM. Meskipun model Random Forest maupun Decision Tree menunjukkan indikasi overfitting pada data training, kemampuan generalisasi keduanya—khususnya Random Forest—tetap terjaga dengan baik pada data testing. Hasil ini menunjukkan bahwa model berbasis ensemble tree lebih sesuai digunakan untuk memprediksi intensi pembelian pada dataset Online Shoppers Intention dibandingkan model linear maupun SVM dengan konfigurasi standar.
