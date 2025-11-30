import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE, RandomOverSampler

# 1. VERİ HAZIRLIĞI
df = pd.read_csv('framingham.csv')
df_clean = df.dropna()

def categorize_bp(row):
    if row['sysBP'] >= 140 or row['diaBP'] >= 90:
        return 2 # Hipertansiyon
    elif row['sysBP'] < 90 and row['diaBP'] < 60:
        return 1 # Hipotansiyon
    else:
        return 0 # Normal

df_clean['BP_Status'] = df_clean.apply(categorize_bp, axis=1)

feature_cols = ['male', 'age', 'education', 'currentSmoker', 'cigsPerDay', 'BPMeds', 
                'prevalentStroke', 'prevalentHyp', 'diabetes', 'totChol', 'BMI', 'heartRate', 'glucose']

X = df_clean[feature_cols]
y = df_clean['BP_Status']

# Train/Test Ayrımı
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. SMOTE İLE VERİ ÇOĞALTMA
print("SMOTE uygulanıyor...")
# Hipo sınıfı (1) çok az (3 kişi) olduğu için k_neighbors=1 yapıyoruz.
try:
    smote = SMOTE(random_state=42, k_neighbors=1)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    print("SMOTE başarıyla uygulandı.")
except:
    print("SMOTE hatası! RandomOverSampler kullanılıyor.")
    ros = RandomOverSampler(random_state=42)
    X_train_resampled, y_train_resampled = ros.fit_resample(X_train_scaled, y_train)

# 3. MODEL EĞİTİMİ VE K-FOLD CV
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# 10-Katlı Çapraz Doğrulama (K-Fold Cross Validation)
# Bu işlem, modelin başarısının 'şans eseri' olmadığını kanıtlar.
cv_results = cross_val_score(rf_model, X_train_resampled, y_train_resampled, cv=10, scoring='accuracy')

print(f"\n10-Fold CV Ortalama Doğruluk Skoru: %{cv_results.mean()*100:.2f}")

# Modeli Eğit ve Test Et
rf_model.fit(X_train_resampled, y_train_resampled)
y_pred = rf_model.predict(X_test_scaled)

print("Test Seti Doğruluğu:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, zero_division=0))

# 4. GÖRSELLEŞTİRME
plt.figure(figsize=(12, 5))

# Sınıf Dağılımı Grafiği (SMOTE Sonrası)
plt.subplot(1, 2, 1)
unique, counts = np.unique(y_train_resampled, return_counts=True)
plt.bar(unique, counts, color=['blue', 'orange', 'green'])
plt.xticks(unique, ['Normal', 'Hipo', 'Hiper'])
plt.title('SMOTE Sonrası Eğitim Verisi Dağılımı')
plt.ylabel('Örnek Sayısı')

# Confusion Matrix
plt.subplot(1, 2, 2)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (SMOTE Modeli)')
plt.xlabel('Tahmin')
plt.ylabel('Gerçek')

plt.tight_layout()
plt.savefig('smote_sonuc.png')
print("Grafik 'smote_sonuc.png' olarak kaydedildi.")