# Klasifikasi Iris Dataset menggunakan Logistic Regression
# Pada modul kali ini, kita akan melakukan klasifikasi untuk dataset iris dengan menggunakan algoritma Regresi Logistik.
# Adapun prediksi didasarkan pada fitur/karakteristik dari bunga Iris, yaitu: Sepal Length, Sepal Width, Petal Length,
# Petal Width yang sudah memiliki kelas (species). Kelasnya adalah iris-setosa, iris-virginica, iris-versicolor.

# MODUL REGRESI LOGISTIK
# 1. Impor Library/Packages

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# 2. Load Dataset
#Membaca dataset
dataset = pd.read_csv("Iris.csv")
#Tampilan 5 data teratas
dataset.head()

# 3. Data Preprocessing
#Mengecek apakah dataset memiliki nilai NaN
dataset.isna().values.any()
# Pengecekan nilai NaN dilakukan untuk melihat apakah ada data yang tidak memiliki nilai atau NaN.
# Jika ada, lakukan pengisian terhadap nilai kosong dengan variabel 0. Akan tetapi, pada data
# ini tertulis false yang berarti semua data memiliki nilai sehingga dapat langsung diproses.

# Untuk mempermudah melihat fitur-fitur yang terdapat pada dataset, kita dapat menggunakan fungsi dtypes.

#Mengecek fitur yang digunakan
print(dataset.dtypes)

#Splitting Data fitur dan label(target)
X = dataset.iloc[:, :-1] #X = memilih semua fitur kecuali kolom terakhir
y = dataset.iloc[:, -1] #y = memilih target yaitu kolom terakhir

#Plotting relasi anata variabel fitur dan target
plt.xlabel("Features")
plt.ylabel("Species")

pltX = dataset.loc[:, "SepalLengthCm"]
pltY = dataset.loc[:, "Species"]
plt.scatter(pltX, pltY, color="blue", label="SepalLengthCm")

pltX = dataset.loc[:, "SepalWidthCm"]
pltY = dataset.loc[:, "Species"]
plt.scatter(pltX, pltY, color="green", label="SepalWidthCm")

pltX = dataset.loc[:, "PetalLengthCm"]
pltY = dataset.loc[:, "Species"]
plt.scatter(pltX, pltY, color="red", label="PetalLengthCm")

pltX = dataset.loc[:, "PetalWidthCm"]
pltY = dataset.loc[:, "Species"]
plt.scatter(pltX, pltY, color="black", label="PetalWidthCm")

# 4. Pemisahan data untuk training dan testing
# Tahapan selanjutnya adalah melakukan pemisahan data untuk training dan testing.
# Hal ini diperlukan agar kita bisa melihat bagaimana algoritma belajar untuk melakukan prediksi pada testing data.
# Kita akan membagi data menjadi 80% training dan 20% testing menggunakan fungsi train_test_split() dari sklearn.model_selection.

#Data dibagi menjadi 80% untuk training dan 20% untuk testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Modelling
# Pada tahapan ini yang dilakukan adalah mentraining data yang ada menggunakan algoritma logistic regression. Berikut cara membuat dan melatih logistic regression model.
#train the model
model = LogisticRegression()
model.fit(X_train, y_train)
#test the model
predictions = model.predict(X_test)
print(predictions)
print()

# 6. Evaluasi Model
# Untuk mempermudah dalam melihat performa atau kinerja dari model, kita akan menggunakan beberapa metrics seperti precision, recall, f1-score.
#Mengevaluasi kinerja model dengan metrices precision, recall, f1-score
print(classification_report(y_test, predictions))
print("accuracy: ", accuracy_score(y_test, predictions))
