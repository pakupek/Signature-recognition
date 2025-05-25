import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def load_data(filename="features.csv"):
    print(f"Ładuję dane z {filename}...")
    labels = []
    features = []
    with open(filename, newline='') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            labels.append(row['label'])
            feat = [float(row[k]) for k in row if k != 'label']
            features.append(feat)
            if i < 3:
                print(f"Przykład {i+1}: label={row['label']}, features={feat}")
    print(f"Załadowano {len(labels)} próbek.")
    return np.array(features), np.array(labels)

def plot_pca(X, y_true, y_pred=None, title="PCA visualization"):
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    plt.figure(figsize=(8,6))
    unique_labels = np.unique(y_true)

    # Rysuj prawdziwe klasy
    for label in unique_labels:
        idx = np.where(y_true == label)
        plt.scatter(X_2d[idx, 0], X_2d[idx, 1], label=f"Prawdziwa klasa {label}", marker='o', alpha=0.5)

    if y_pred is not None:
        for i in range(len(y_pred)):
            if y_pred[i] == y_true[i]:
                # dobrze sklasyfikowane: zielony x
                plt.scatter(X_2d[i, 0], X_2d[i, 1], color='green', marker='x')
                plt.text(X_2d[i, 0] - 12, X_2d[i, 1] - 12, f"{y_true[i]}", color='green', fontsize=9, fontweight='bold')
            else:
                # zle sklasyfikowane: czerwony x + tekst z przewidziana klasa
                plt.scatter(X_2d[i, 0], X_2d[i, 1], color='red', marker='x')
                plt.text(X_2d[i, 0] + 2, X_2d[i, 1] + 2, f"{y_pred[i]}", color='red', fontsize=9, fontweight='bold')
                plt.text(X_2d[i, 0] - 12, X_2d[i, 1] - 12, f"{y_true[i]}", color='green', fontsize=9, fontweight='bold')
                

    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.show()

def run_knn():
    features, labels = load_data()
    print("Dzielę dane na train i test (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.5, random_state=42)
    print(f"Train: {len(X_train)} próbek, Test: {len(X_test)} próbek")

    knn = KNeighborsClassifier(n_neighbors=3)
    print("Trenuję klasyfikator k-NN...")
    knn.fit(X_train, y_train)
    print("Robię predykcje na zbiorze testowym...")
    y_pred = knn.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Dokładność: {acc:.4f}")
    print("Szczegółowy raport klasyfikacji:")
    print(classification_report(y_test, y_pred))

    print("Wizualizacja danych treningowych w PCA:")
    plot_pca(X_train, y_train, title="PCA - dane treningowe")

    print("Wizualizacja danych testowych w PCA (prawdziwe klasy i przewidywania):")
    plot_pca(X_test, y_test, y_pred=y_pred, title="PCA - dane testowe (prawdziwe vs przewidziane)")

if __name__ == "__main__":
    run_knn()
