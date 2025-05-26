import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors


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


colors = {
    '1': 'red',
    '2': 'green',
    '3': 'blue',
    '4': 'purple',
    '5': 'pink',
    '6': 'yellow',
    '7': 'orange',
    '8': 'cyan'
}


def plot_all_points(features, labels, ax):
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(features)

    unique_labels = np.unique(labels)

    for label in unique_labels:
        idx = np.where(labels == label)
        ax.scatter(X_2d[idx, 0], X_2d[idx, 1], label=f"Klasa {label}", alpha=0.6, color=colors[label])

    ax.set_title("Wszystkie Punkty")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()

    return pca


def plot_test_with_predictions(pca, X_test, y_test, y_pred, ax):
    X_test_2d = pca.transform(X_test)

    unique_labels = np.unique(y_test)
    for label in unique_labels:
        idx = np.where(y_test == label)
        ax.scatter(X_test_2d[idx, 0], X_test_2d[idx, 1], label=f"Klasa {label}", alpha=0.6, color=colors[label])

    for i in range(len(y_test)):
        color = 'green' if y_test[i] == y_pred[i] else 'red'
        ax.scatter(X_test_2d[i, 0], X_test_2d[i, 1], marker='o', color=color, s=100, linewidths=1, facecolors='none')

    ax.set_title("Dane testowe z poprawnością predykcji")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()


def plot_wrong_predictions(pca, X_test, y_test, y_pred, X_train, y_train, knn, n_neighbors, ax):
    X_test_2d = pca.transform(X_test)
    X_train_2d = pca.transform(X_train)

    wrong_idx = [i for i in range(len(y_test)) if y_test[i] != y_pred[i]]
    unique_wrong_labels = np.unique(y_test[wrong_idx])

    shown_labels = set()

    for label in unique_wrong_labels:
        idx = [i for i in wrong_idx if y_test[i] == label]
        ax.scatter(X_test_2d[idx, 0], X_test_2d[idx, 1], label=f"Klasa {label}", alpha=0.6, color=colors[label])
        shown_labels.add(label)

    for i in wrong_idx:
        test_point = [X_test[i]]
        neighbors_idx = knn.kneighbors(test_point, n_neighbors, return_distance=False)[0]     

        for neighbor in neighbors_idx:
            
            label = None
            if y_train[neighbor] not in shown_labels:
                label = f'Klasa {y_train[neighbor]}'
                shown_labels.add(y_train[neighbor])

            ax.annotate(
            '',
            xy=(X_train_2d[neighbor, 0], X_train_2d[neighbor, 1]),  # koniec strzałki (sąsiad)
            xytext=(X_test_2d[i, 0], X_test_2d[i, 1]),             # początek strzałki (punkt testowy)
            arrowprops=dict(arrowstyle='->', color='red', lw=1)
            )

            ax.scatter(
            X_train_2d[neighbor, 0],
            X_train_2d[neighbor, 1],
            color=colors[y_train[neighbor]],
            alpha=0.4,
            label=label
            )

    ax.set_title("Tylko błędne predykcje z liniami do najbliższych sąsiadów")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()


def plot_right_predictions(pca, X_test, y_test, y_pred, X_train, y_train, knn, n_neighbors, ax):
    X_test_2d = pca.transform(X_test)
    X_train_2d = pca.transform(X_train)

    right_idx = [i for i in range(len(y_test)) if y_test[i] == y_pred[i]]
    unique_right_labels = np.unique(y_test[right_idx])

    shown_labels = set()

    for label in unique_right_labels:
        idx = [i for i in right_idx if y_test[i] == label]
        ax.scatter(X_test_2d[idx, 0], X_test_2d[idx, 1], label=f"Klasa {label}", alpha=0.6, color=colors[label])
        shown_labels.add(label) 

    for i in right_idx:
        test_point = [X_test[i]]
        neighbors_idx = knn.kneighbors(test_point, n_neighbors, return_distance=False)[0]     

        for neighbor in neighbors_idx:
            
            label = None
            if y_train[neighbor] not in shown_labels:
                label = f'Klasa {y_train[neighbor]}'
                shown_labels.add(y_train[neighbor])

            ax.annotate(
            '',
            xy=(X_train_2d[neighbor, 0], X_train_2d[neighbor, 1]),  # koniec strzałki (sąsiad)
            xytext=(X_test_2d[i, 0], X_test_2d[i, 1]),             # początek strzałki (punkt testowy)
            arrowprops=dict(arrowstyle='->', color='green', lw=1)
            )

            ax.scatter(
            X_train_2d[neighbor, 0],
            X_train_2d[neighbor, 1],
            color=colors[y_train[neighbor]],
            alpha=0.4,
            label=label
            )

    ax.set_title("Tylko poprawne predykcje z liniami do najbliższych sąsiadów")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()


def plot_final(features, labels, X_test, y_test, y_pred, X_train, y_train, knn, n_neighbors):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 6))
    pca = plot_all_points(features, labels, ax1)

    X_2d_all = pca.transform(features)

    x_min, x_max = X_2d_all[:, 0].min() - 25, X_2d_all[:, 0].max() + 25
    y_min, y_max = X_2d_all[:, 1].min() - 25, X_2d_all[:, 1].max() + 25

    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)

    plot_test_with_predictions(pca, X_test, y_test, y_pred, ax2)
    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(y_min, y_max)

    plot_wrong_predictions(pca, X_test, y_test, y_pred, X_train, y_train, knn, n_neighbors, ax3)
    ax3.set_xlim(x_min, x_max)
    ax3.set_ylim(y_min, y_max)

    plot_right_predictions(pca, X_test, y_test, y_pred, X_train, y_train, knn, n_neighbors, ax4)
    ax4.set_xlim(x_min, x_max)
    ax4.set_ylim(y_min, y_max)

    plt.tight_layout()
    plt.show()


def run_knn():
    n_neighbors = 3
    features, labels = load_data()
    print("Dzielę dane na train i test...")
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    print(f"Train: {len(X_train)} próbek, Test: {len(X_test)} próbek")

    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    print("Trenuję klasyfikator k-NN...")
    knn.fit(X_train, y_train)
    print("Robię predykcje na zbiorze testowym...")
    y_pred = knn.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Dokładność: {acc:.4f}")
    print("Szczegółowy raport klasyfikacji:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    plot_final(features, labels, X_test, y_test, y_pred, X_train, y_train, knn, n_neighbors)


if __name__ == "__main__":
    run_knn()
