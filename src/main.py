import pandas as pd
import glob
import os
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
import sklearn.metrics as metrics
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Path folder
folder_path = "data"

# Mengambil semua file csv di folder data kecuali file sample_submission.csv
files = [f for f in glob.glob(os.path.join(folder_path, "*.csv")) if "sample" not in f.lower()]

# Membuat folder data cleaned
cleaned_path = os.path.join(folder_path, "data cleaned")
os.makedirs(cleaned_path, exist_ok=True)

# Data understanding dan data cleaning untuk semua file csv di folder data
for file in files:

    #Membaca file csv dan memisahkan kolom data dengan delimiter koma (,)
    df = pd.read_csv(file, delimiter = ",")

    # ===========================
    # DATA UNDERSTANDING
    # ===========================
    print(f"\n" + "="*60)
    print(f"Data understanding for {file}:")
    print(f"="*60)

    # 1. Melihat ukuran data
    print("\n1. Ukuran data:")
    print(f"Jumlah baris : {df.shape[0]}")
    print(f"Jumlah kolom : {df.shape[1]}")

    # 2. Melihat nama kolom
    print("\n2. Nama kolom:")
    print(df.columns.tolist())

    # 3. Melihat 5 data pertama
    print("\n3. 5 data pertama:")
    print(df.head())

    # 4. Melihat informasi data
    print("\n4. Informasi data:")
    df.info()

    # 5. Mengecek missing value
    print("\n5. Missing values:")
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0])

    # 6. Mengecek jumlah data duplikat
    print("\n6. Jumlah data duplikat:")
    print(df.duplicated().sum())

    # 7. Melihat distribusi kolom kategorikal
    print("\n7. Distribusi data kategorikal:")
    kolom_kategorikal = df.select_dtypes(
        include = ["object", "string"]
    ).columns

    for col in kolom_kategorikal:
        print(f"\nDistribusi kolom {col}:")
        print(df[col].value_counts())
    

    # ===========================
    # DATA CLEANING
    # ===========================

    # Mengubah data age menjadi numerik
    if "age" in df.columns:
        df["age"] = pd.to_numeric(df["age"], errors="coerce")

    # Mengubah semua kolom teks ke lowercase
    # Pada bagian respondent_id akan diubah menjadi uppercase
    # Menghapus spasi di awal dan akhir teks
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype(str).str.lower().str.strip()

    if "respondent_id" in df.columns:
        df["respondent_id"] = df["respondent_id"].str.upper()

    
    # ===========================
    # MENANGANI MISSING VALUES
    # ===========================

    # Mengisi missing values numerik dengan median
    kolom_numerik = df.select_dtypes(include = "number").columns

    for col in kolom_numerik:
        if df[col].isnull().any():
            median = df[col].median()
            df[col] = df[col].fillna(median)

    # Mengisi missing values kategorikal dengan modus
    kolom_kategorikal = df.select_dtypes(include = ["object", "string"]).columns
    
    for col in kolom_kategorikal:
        df[col] = df[col].replace("nan", pd.NA)
        if df[col].isnull().any():
            modus = df[col].mode()[0]
            df[col] = df[col].fillna(modus)

    print(f"\nData cleaning selesai untuk file: {file}")

    if "train" in file.lower():
        output_file = os.path.join(cleaned_path, "train_cleaned.xlsx")
    elif "test" in file.lower():
        output_file = os.path.join(cleaned_path, "test_cleaned.xlsx")

    # Menyimpan file yang sudah dibersihkan ke dalam format Excel
    df.to_excel(output_file, index=False)

    # Menyimpan file
    print(f"File cleaned saved to: {output_file}")


# ========================================================
# DATA PREPARATION / TRANSFORMATION
# ========================================================

print(f"\n" + "="*60)
print(f"Transformasi data train_cleaned.xlsx:")
print(f"="*60)

# Membaca file train_cleaned.xlsx
train_path = os.path.join(cleaned_path, "train_cleaned.xlsx")
df_train = pd.read_excel(train_path)

# Melihat informasi data train_cleaned.xlsx
print("\nInformasi data:")
df_train.info()

# Melihat distribusi kolom kategorikal
print("\nDistribusi data kategorikal:")
kolom_kategorikal = df_train.select_dtypes(
        include = ["object", "string"]
    ).columns

for col in kolom_kategorikal:
    print(f"\nDistribusi kolom {col}:")
    print(df_train[col].value_counts())

# Menentukan nama target
TARGET = "digital_academic_wellbeing_level"

# Memisahkan feature (X) dan target (y)
# respondent_id tidak digunakan sebagai feature
# X adalah seluruh variable predikator kecuali respondent_id dan target
X = df_train.drop(columns=["respondent_id", TARGET])

# y adalah target yang akan diprediksi
y = df_train[TARGET]

# Visualisasi distribusi kelas target
fig, ax = plt.subplots(figsize=(6, 5))
y.value_counts().plot(kind='bar', color=['#4C72B0', '#DD8452', '#55A868'], ax=ax)
ax.set_xlabel('Kelas')
ax.set_ylabel('Jumlah')
ax.set_title('Distribusi Kelas Target')
ax.tick_params(axis='x', rotation=0)
plt.tight_layout()
plt.savefig('data/class_distribution.png', dpi=150)
plt.show()

print(f"\nJumlah feature (X) yang digunakan: ", X.shape[1])
print(f"\nNama feature (X) yang digunakan:")
print(X.columns.tolist())

print(f"\nJumlah target (y) yang digunakan: ", y.shape[0])
print(f"\nNama target (y) yang digunakan: {TARGET}")

# Membagi data untuk train dan validation
print(f"\n=== Membagi data menjadi train dan validation dengan rasio 80:20 ===")
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

kategorikal = X_train.select_dtypes(
    include=["object", "string"]
    ).columns.to_list()

skf = StratifiedKFold(
    n_splits=5, shuffle=True, random_state=42
    )
f1_scores = []
acc_scores = []

print("\n" + "="*60)
print("CROSS-VALIDATION (5-FOLD)")
print("="*60)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]

    fold_model = CatBoostClassifier(
        iterations=1000, 
        depth=6, 
        learning_rate=0.05,
        loss_function='MultiClass', 
        random_seed=42,
        early_stopping_rounds=50, 
        verbose=False
    )
    fold_model.fit(X_tr, y_tr, eval_set=(X_va, y_va), cat_features=kategorikal)

    pred = fold_model.predict(X_va).flatten()
    f1 = metrics.f1_score(y_va, pred, average='macro')
    acc = metrics.accuracy_score(y_va, pred)
    f1_scores.append(f1)
    acc_scores.append(acc)
    print(f"Fold {fold+1} - Accuracy: {acc:.4f} | Macro F1: {f1:.4f}")

accuracy = np.mean(acc_scores)
f1_macro = np.mean(f1_scores)
print(f"\nRata-rata Accuracy : {accuracy:.4f} (+/- {np.std(acc_scores):.4f})")
print(f"Rata-rata Macro F1 : {f1_macro:.4f} (+/- {np.std(f1_scores):.4f})")

# Visualisasi hasil cross-validation per fold
fig, axes = plt.subplots(figsize=(8, 5))
folds = range(1, len(f1_scores) + 1)
axes.plot(folds, acc_scores, marker='o', label='Accuracy')
axes.plot(folds, f1_scores, marker='s', label='Macro F1')
axes.axhline(y=np.mean(f1_scores), color='gray', linestyle='--', alpha=0.5, label='Rata-rata Macro F1')
axes.set_xlabel('Fold')
axes.set_ylabel('Skor')
axes.set_title('Performa Cross-Validation per Fold')
axes.set_xticks(list(folds))
axes.legend()
axes.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('data/cv_performance.png', dpi=150)
plt.show()

# ========================================================
# PEMODELAN DENGAN CATBOOST
# ========================================================

model = CatBoostClassifier(
    iterations=1000,
    depth=6,
    learning_rate=0.05,
    loss_function='MultiClass',
    random_seed=42,
    early_stopping_rounds=50,
    verbose=100
)

model.fit(
    X_train,
    y_train,
    cat_features=kategorikal
)

y_pred = model.predict(X_val).flatten()

print("\n" + "="*60)
print("HASIL EVALUASI")
print("="*60)

print(f"\nAccuracy : {accuracy:.4f}")
print(f"Macro F1 : {f1_macro:.4f}")

print("\nClassification Report:")
print(
    metrics.classification_report(
        y_val,
        y_pred
    )
)


# =======================================================
# CONFUSION MATRIX
# =======================================================

print("\n" + "="*60)
print("CONFUSION MATRIX")
print("="*60)

cm = metrics.confusion_matrix(
    y_val,
    y_pred,
    labels = ["low", "medium", "high"]
)

print ("\nConfusion Matrix:")
print(cm)

# Visualiasasi confusion matrix
fig, axes = plt.subplots(figsize=(6, 5))
sns.heatmap(
    cm, annot=True, fmt='d', ax=axes, cmap='Blues',
    xticklabels=['low', 'medium', 'high'],
    yticklabels=['low', 'medium', 'high']
    axes=axes
    )
axes.set_xlabel('Prediksi')
axes.set_ylabel('Asli')
axes.set_title('Confusion Matrix')
plt.tight_layout()
plt.savefig('data/confusion_matrix.png', dpi=150)
plt.show()

# =======================================================
# FEATURE IMPORTANCE
# =======================================================

feature_importance = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="importance",
    ascending=False
)

print("\n" + "="*60)
print("FEATURE IMPORTANCE")
print("="*60)

print(feature_importance)

# Visualisasi feature importance
fig, axes = plt.subplots(figsize=(9, 8))
top_features = feature_importance.head(15)
axes.barh(top_features['feature'], top_features['importance'], color='skyblue')
axes.invert_yaxis()
axes.set_xlabel('Importance')
axes.set_title('Top 15 Feature Importance')
plt.tight_layout()
plt.savefig('data/feature_importance.png', dpi=150)
plt.show()

# ========================
# PREDIKSI DATA TEST
# ========================

print("\n" + "="*60)
print("PREDIKSI DATA TEST")
print("="*60)

# Membaca file test_cleaned.xlsx
test_path = os.path.join(cleaned_path, "test_cleaned.xlsx")
df_test = pd.read_excel(test_path)

# Feature yang digunakan untuk prediksi adalah seluruh kolom kecuali respondent_id
X_test_pred = df_test.drop(columns=["respondent_id"])

# Memprediksi data test
y_test_pred = model.predict(X_test_pred)


# Mengubah bentuk array menjadi 1 dimensi
y_test_pred = y_test_pred.flatten()

y_test_pred = pd.Series(y_test_pred).str.capitalize()

print(f"\nHasil prediksi 5 data pertama:")
print(y_test_pred[:5])

# =====================================
# MENYIMPAN HASIL PREDIKSI KE FILE CSV
# =====================================
submission = pd.DataFrame({
    "respondent_id": df_test["respondent_id"],
    "digital_academic_wellbeing_level": y_test_pred
})

submission["respondent_id"] = submission["respondent_id"].str.upper()

print("\n" + "="*60)
print("SUBMISSION")
print("="*60)

print("\n5 data pertama submission:")
print(submission.head())


submission_path = os.path.join("data/submission", "submission.csv")
submission.to_csv(submission_path, index=False)

print(f"\nFile submission berhasil disimpan: {submission_path}")

