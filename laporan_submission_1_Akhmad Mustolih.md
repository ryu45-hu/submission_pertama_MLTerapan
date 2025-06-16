# Laporan Proyek Machine Learning - Akhmad Mustolih

## Domain Proyek

Penyakit diabetes merupakan salah satu masalah kesehatan global yang semakin meningkat prevalensinya setiap tahun. Menurut laporan dari World Health Organization (WHO), lebih dari 422 juta orang di seluruh dunia menderita diabetes, dan angka ini diperkirakan akan terus meningkat secara signifikan dalam beberapa dekade ke depan [[1]](https://www.who.int/news-room/fact-sheets/detail/diabetes). Diabetes, khususnya tipe 2, dapat menyebabkan komplikasi serius seperti penyakit jantung, gagal ginjal, kebutaan, hingga kematian dini jika tidak terdeteksi dan ditangani sejak dini.

Di Indonesia sendiri, berdasarkan data dari Riskesdas (Riset Kesehatan Dasar) 2018, prevalensi diabetes melitus yang terdiagnosis oleh tenaga kesehatan maupun tidak terdiagnosis terus meningkat, yakni dari 6,9% pada tahun 2013 menjadi 10,9% pada tahun 2018 [[2]](https://www.litbang.kemkes.go.id/laporan-riset-kesehatan-dasar-riskesdas/). Hal ini menandakan bahwa banyak kasus diabetes tidak terdeteksi sejak awal, yang pada akhirnya menimbulkan beban ekonomi dan kesehatan yang lebih besar.

Dalam konteks ini, penerapan analisis prediktif berbasis data menjadi semakin penting. Dengan memanfaatkan teknologi machine learning dan data kesehatan seperti tekanan darah, kadar glukosa, insulin, hingga indeks massa tubuh (BMI), kita dapat membangun sistem prediksi yang mampu mengidentifikasi potensi risiko diabetes pada seseorang dengan akurasi yang tinggi.

Proyek ini menggunakan dataset Healthcare-Diabetes.csv yang berisi 2.768 observasi dan 10 variabel, termasuk variabel target Outcome yang merepresentasikan diagnosis diabetes (1 = diabetes, 0 = tidak diabetes). Model prediktif dikembangkan menggunakan tiga algoritma machine learning, yaitu Linear Regression, Random Forest, dan K-Nearest Neighbors (KNN). Evaluasi dilakukan menggunakan metrik Mean Squared Error (MSE) untuk menilai kinerja masing-masing model dalam memprediksi risiko diabetes.

Melalui pendekatan ini, diharapkan tercipta sistem deteksi dini diabetes yang akurat dan efisien, yang dapat membantu lembaga kesehatan, klinik, maupun individu dalam mengambil keputusan preventif sebelum kondisi diabetes berkembang menjadi kronis.

## Business Understanding

Penerapan machine learning dalam bidang kesehatan membuka peluang besar dalam upaya preventif terhadap penyakit kronis seperti diabetes. Dalam konteks ini, pemahaman yang kuat terhadap tujuan bisnis dan formulasi masalah sangat penting untuk mengarahkan proses analisis data dan pembangunan model prediktif yang akurat.

### Problem Statements

- Bagaimana mengidentifikasi pasien yang berisiko tinggi terkena diabetes berdasarkan data kesehatan dasar seperti kadar glukosa, tekanan darah, BMI, dan usia?
- Model machine learning apa yang paling optimal dalam memprediksi risiko diabetes berdasarkan performa metrik Mean Squared Error (MSE)?
- Apakah preprocessing data seperti standarisasi dan penanganan outlier dapat meningkatkan performa model prediktif secara signifikan?

### Goals

- Mengembangkan sistem prediktif yang mampu mengidentifikasi pasien berisiko diabetes hanya dengan data medis dasar.
- Mengevaluasi dan membandingkan performa beberapa model prediktif, yaitu Linear Regression, Random Forest, dan K-Nearest Neighbors, berdasarkan metrik MSE.
- Menilai dampak preprocessing data terhadap performa model. 

### Solution statements

- Menggunakan lebih dari satu algoritma machine learning (Linear Regression, Random Forest, KNN) untuk membangun model prediktif.
- Melakukan preprocessing data melalui tahapan:
  - Penghapusan outlier dengan metode Interquartile Range (IQR)
  - Standarisasi fitur numerik menggunakan StandardScaler
  - Penghapusan kolom yang tidak relevan (misal kolom Id)

## Data Understanding
Dataset yang digunakan dalam proyek ini merupakan dataset Healthcare-Diabetes yang bersumber dari platform Kaggle. Dataset ini berisi data medis dasar yang sering dijadikan indikator risiko diabetes pada pasien. Dataset dapat diunduh melalui tautan berikut: [Diabetes Dataset]([https://archive.ics.uci.edu/ml/datasets/Restaurant+%26+consumer+data](https://www.kaggle.com/datasets/nanditapore/healthcare-diabetes)). Dataset ini terdiri dari 2.768 baris dan 10 kolom, dengan 9 kolom sebagai fitur dan 1 kolom target (label). Data ini digunakan untuk memprediksi kemungkinan seseorang menderita diabetes berdasarkan beberapa fitur kesehatan.

### Variabel-variabel pada Diabetes dataset adalah sebagai berikut:
- Id: Nomor unik identifikasi pasien. Kolom ini dihapus karena tidak memiliki nilai prediktif.
- Pregnancies: Jumlah kehamilan yang pernah dialami oleh pasien.
- Glucose: Kadar glukosa dalam darah.
- BloodPressure: Tekanan darah diastolik (mm Hg).
- SkinThickness: Ketebalan lipatan kulit trisep (mm).
- Insulin: Kadar insulin dalam darah (mu U/ml).
- BMI: Indeks Massa Tubuh, dihitung sebagai berat (kg) dibagi kuadrat tinggi badan (m²).
- DiabetesPedigreeFunction: Skor riwayat keluarga penderita diabetes (faktor genetik).
- Age: Usia pasien (dalam tahun).
- Outcome: Variabel target, bernilai 1 jika pasien menderita diabetes dan 0 jika tidak.

### Exploratory Data Analysis (EDA)
1. Informasi Struktur Data
   Dataset tidak mengandung missing value, sehingga tidak perlu imputasi data.

2. Distribusi dan Outlier
   Visualisasi distribusi menggunakan boxplot menunjukkan adanya outlier pada beberapa fitur.

3. Distribusi Histogram
   Histogram seluruh fitur ditampilkan untuk melihat penyebaran data secara visual.

4. Korelasi Antar Fitur
   Korelasi dihitung dan divisualisasikan dalam bentuk heatmap.

## Data Preparation
Tahapan ini menjelaskan proses pra-pemrosesan data yang dilakukan sebelum membangun model prediktif. Tujuan dari data preparation adalah memastikan bahwa data bersih, relevan, dan berada dalam bentuk yang sesuai untuk digunakan dalam algoritma machine learning.

1. Menghapus Kolom Tidak Relevan (```Id```)

   Kolom Id hanya berfungsi sebagai penanda identitas unik pasien dan tidak memiliki pengaruh terhadap prediksi. Oleh karena itu, kolom ini dihapus menggunakan:
   ```python
   df.drop('Id', axis=1, inplace=True)
   ```
   
2. Menghapus Duplikasi dan Mengecek Missing Value

   Mengecek data duplikat:
   ```python
   df.duplicated().sum()
   ```
   Mengecek Missing value:
   ```python
   df.isna().sum()
   ```
   
3. Deteksi dan Penanganan Outlier

   Outlier dapat menyebabkan model menjadi bias atau overfitting. Deteksi menggunakan boxplot untuk setiap fitur. Penanganan dilakukan dengan metode Interquartile Range (IQR):
   ```python
   Q1 = df.quantile(0.25)
   Q3 = df.quantile(0.75)
   IQR = Q3 - Q1
   # Menghapus baris yang mengandung outlier
   df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
   df.info()
   ```
   
4. Pembagian Data Training dan Testing

   Data dibagi menggunakan stratified split dengan rasio 70:30 untuk menjaga distribusi kelas pada label Outcome.
   ```python
   X = df.drop(["Outcome"],axis=1)
   y = df["Outcome"]
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42, stratify = y)
   ```
   
5. Standardisasi Data

   Dilakukan karena beberapa algoritma seperti KNN dan regresi linier sensitif terhadap skala.
   ```python
   scaler = StandardScaler()
   scaler.fit(X_train)
   X_train = scaler.transform(X_train)
   X_train = pd.DataFrame(X_train, columns=X.columns)
   X_train
   X_test = scaler.transform(X_test)
   ```

## Modeling
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.
1. Linear Regression
   - Parameter: default (tanpa penyesuaian).
   - Deskripsi: model regresi sederhana yang mengasumsikan hubungan linier antara fitur dan target.
   - Kelebihan: Cepat dalam pelatihan, Mudah dipahami dan diinterpretasi.
   - Kekurangan: Tidak cocok untuk hubungan non-linear, Rentan terhadap multikolinearitas dan outlier.

2. Random Forest Regressor
   - Parameter:
     - n_estimators = 200: jumlah pohon dalam hutan.
     - max_depth = 20: kedalaman maksimum pohon.
     - random_state = 42: untuk reprodusibilitas hasil.
   - Deskripsi: model ensembel berbasis pohon keputusan yang menggabungkan banyak pohon untuk meningkatkan akurasi.
   - Kelebihan: Mampu menangani hubungan non-linear dan interaksi anatara fitur, tahan terhadap overfitting, tidak memerlukan scaling fitur.

3. K-Nearest Neighbors (KNN) Regressor
   - Parameter: n_neighbors = 2
   - Deskripsi: model prediksi berdasarkan kemiripan dengan titik data tetangga terdekat.
   - Kelebihan: Sederhana dan efektif untuk dataset kecil, tidak membutuhkan asumsi distribusi data.
   - Kekurangan: Sensitif terhadap skala dan outlier (karena berbasis jarak), tidak efisien untuk dataset besar, sangat tergantung pada pemilihan parameter k.

Model terbaik yang dipilih adalah Random Forest, karena menghasilkan nilai Mean Squared Error (MSE) terkecil pada data uji, yaitu sebesar 0.000009, dibandingkan dengan model KNN dan Linear Regression. Selain itu, model ini juga menunjukkan performa yang konsisten antara data latih dan data uji, menandakan kemampuan generalisasi yang baik tanpa overfitting.

Random Forest juga unggul dalam menangkap hubungan non-linier antar fitur, sehingga lebih efektif dalam menyelesaikan permasalahan regresi pada dataset ini. Oleh karena itu, Random Forest ditetapkan sebagai model terbaik berdasarkan metrik evaluasi yang diperoleh.

## Evaluation

### Metrik Evaluasi yang Digunakan: Mean Squared Error (MSE)\
Dalam proyek ini, metrik evaluasi yang digunakan adalah Mean Squared Error (MSE) karena permasalahan yang diangkat merupakan regresi, yaitu memprediksi nilai kontinu (dalam hal ini nilai "Outcome" diabetes dalam skala tertentu). MSE sangat umum digunakan dalam regresi karena memberikan penalti besar terhadap kesalahan prediksi yang signifikan.

Rumus MSE adalah sebagai berikut:

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

dengan : <br>
${n}$ : jumlah data<br>
$y_i$ : nilai aktual<br>
$\hat{y}_i$ : nilai prediksi<br>
$(y_i - \hat{y}_i)^2$ : selisih kuadrat antara nilai aktual dan prediksi<br>

MSE mengukur rata-rata kuadrat selisih antara nilai aktual dan prediksi. Semakin kecil nilai MSE, maka semakin akurat model dalam memprediksi data.

### Hasil Evaluasi
| model | train	| test | 
|---|---|--- |
| Linear Regression |	0.000145 |	0.000144 |
| Random Forest |	0.000002	| 0.000009 |
| KNN |	0.000004 |	0.000022 |

### Interpretasi Hasil
- Random Forest memberikan performa terbaik dengan nilai MSE terkecil, yaitu 0.000009 pada data testing. Ini menunjukkan bahwa model ini sangat akurat dalam memprediksi data baru dan memiliki kemampuan generalisasi yang baik.
- KNN memiliki performa cukup baik, namun masih lebih tinggi MSE-nya dibandingkan Random Forest.
- Linear Regression menunjukkan performa paling lemah di antara ketiganya, karena kemungkinan tidak mampu menangkap pola non-linear antar fitur.
