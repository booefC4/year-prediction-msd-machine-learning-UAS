# 🎵 Regression Deep Learning — YearPredictionMSD

End-to-end regression pipeline untuk memprediksi **tahun rilis lagu** dari 90 fitur audio (timbre) menggunakan **deep learning (PyTorch MLP)**, lengkap dengan **Optuna** (tuning), **LIME** (interpretasi), dan **MLflow** (tracking).

---

## 📦 Isi
| File | Keterangan |
|------|------------|
| `Regression_DL_YearPrediction.ipynb` | Notebook utama (jalankan ini) |
| `requirements.txt` | Dependency, pinned untuk Python 3.11.9 |
| `README.md` | File ini |

---

## ▶️ Cara Menjalankan

```bash
# 1. (opsional) buat virtual env
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

# 2. install dependency
pip install -r requirements.txt
# torch CPU-only (kalau tidak punya GPU):
# pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

# 3. taruh dataset di folder yang sama -> midterm-regresi-dataset.csv
# 4. buka notebook
jupyter notebook Regression_DL_YearPrediction.ipynb
```

Di dalam notebook (cell **Section 0**):
- Set `DATA_PATH` ke nama file CSV kamu.
- `QUICK = True` → test cepat. `QUICK = False` → hasil final (lebih banyak trial & epoch).
- Lalu **Run All**.

> Kalau `DATA_PATH` tidak ditemukan, notebook otomatis pakai **data sintetis** (struktur sama) supaya tetap bisa demo.

---

## 🔄 Alur Pipeline
1. **Load** — auto-detect header (MSD asli tanpa header; kolom-1 = tahun).
2. **EDA** — distribusi target, missing values, korelasi fitur.
3. **Preprocessing** — impute median → winsorize outlier (P1–P99) → StandardScaler (fitur & target). Semua di-*fit* di train (no leakage).
4. **Model** — MLP PyTorch (`Linear→BatchNorm→ReLU→Dropout`) + Adam + MSE + early stopping.
5. **Tuning** — Optuna (TPE) mencari layer/unit/dropout/lr/wd/batch, objektif = min val RMSE.
6. **Evaluation** — MSE, RMSE, MAE, R² + plot prediksi-vs-aktual & residual + baseline Ridge/Random Forest.
7. **Interpretasi** — LIME per-instance (fitur pendorong prediksi).
8. **Tracking** — semua run/param/metrik ke MLflow.

---

## 📊 Melihat MLflow
```bash
mlflow ui        # jalankan di folder yang sama
```
Buka **http://127.0.0.1:5000** → tab *Experiments* → `year_prediction_dl`.

---

## 📝 Catatan untuk Laporan
- **RMSE/MAE** satuan **tahun** (makin kecil makin bagus). Di MSD asli, RMSE NN yang baik ~**9–10 tahun** (dataset memang sulit).
- **R²** = proporsi variansi tahun yang dijelaskan model.
- **Producer effect:** split resmi MSD = 463.715 baris pertama (train) + 51.630 terakhir (test) agar lagu artis sama tidak bocor. Notebook pakai split acak — bisa diganti di Section 3 kalau perlu.

Alif Gunawan Yudiansyah
1103220221
TK-46-Gab
