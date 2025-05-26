from skimage import data
from skimage.morphology import skeletonize
from skimage.feature import hog
import cv2
import os
from PIL import Image
import numpy as np
from scipy.ndimage import label
import csv

def con_img_gray_scale():
    path = "img/in"
    output_path = "img/out"
    rows = []

    if not os.path.isdir(path):
        print(f"Błąd: Folder {path} nie istnieje.")

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for filename in os.listdir(path):
        if filename.lower().endswith(('.png','.jpg','.jpeg')):
            file_path = os.path.join(path,filename)
            try:
                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    raise ValueError("Nie udało wczytać obrazu")  

                img = cv2.resize(img, (512, 128), interpolation=cv2.INTER_AREA)

                _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

                binary_bool = binary > 0
                skeleton = skeletonize(binary_bool)
                skeleton_uint8 = (skeleton * 255).astype(np.uint8)

                cv2.imwrite(os.path.join(output_path, f"binary_{filename}"), binary)
                cv2.imwrite(os.path.join(output_path, f"skeleton_{filename}"), skeleton_uint8)

                features = extract_features(skeleton_uint8)
                label = filename[6].upper()
                rows.append([label] + list(features.values()))


            except Exception as e:
                print(f"Uszkodzony plik: {filename}\t| Błąd {e}")
    if rows:
        header = ['label'] + list(features.keys())
        with open("features.csv", "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)

def extract_features(final_skeleton):
    features = {}

    endpoints = count_endpoints(final_skeleton)
    branches = count_branch_points(final_skeleton)

    features["endpoints"] = len(endpoints)
    features["branch_points"] = len(branches)
    features["skeleton_length"] = np.count_nonzero(final_skeleton)

    features["avg_distance"] = average_distance_between_points(endpoints, branches)
    features["connected_components"] = count_connected_components(final_skeleton)
    features["extent"] = skeleton_extent(final_skeleton)
    features["density"] = skeleton_density(final_skeleton)

    return features

def count_endpoints(skel):
    skel = skel // 255
    endpoints = []
    for i in range(1, skel.shape[0] - 1):
        for j in range(1, skel.shape[1] - 1):
            if skel[i, j] == 1:
                region = skel[i-1:i+2, j-1:j+2]
                if np.sum(region) - 1 == 1:
                    endpoints.append((i, j))
    return endpoints

def count_branch_points(skel):
    skel = skel // 255
    branches = []
    for i in range(1, skel.shape[0] - 1):
        for j in range(1, skel.shape[1] - 1):
            if skel[i, j] == 1:
                region = skel[i-1:i+2, j-1:j+2]
                neighbors = np.sum(region) - 1
                if neighbors >= 3:
                    branches.append((i, j))
    return branches

def average_distance_between_points(endpoints, branches):
    points = np.array(endpoints + branches)
    if len(points) < 2:
        return 0
    dists = []
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            dists.append(np.linalg.norm(points[i] - points[j]))
    return np.mean(dists)

def count_connected_components(skel):
    skel_bin = skel > 0
    labeled, num = label(skel_bin)
    return num

def skeleton_extent(skel):
    coords = np.column_stack(np.where(skel == 255))
    if coords.size == 0:
        return 0
    height = coords[:, 0].max() - coords[:, 0].min() + 1
    width = coords[:, 1].max() - coords[:, 1].min() + 1
    return height / width if width != 0 else 0

def skeleton_density(skel):
    coords = np.column_stack(np.where(skel == 255))
    if coords.size == 0:
        return 0
    h = coords[:, 0].max() - coords[:, 0].min() + 1
    w = coords[:, 1].max() - coords[:, 1].min() + 1
    return len(coords) / (h * w)

def print_features(features_dict, filename):
    print(f"{filename}:")
    for feature_name, feature_value in features_dict.items():
        print(f"  {feature_name}: {feature_value}")
    print()  # Pusta linia

if __name__ == "__main__":
    con_img_gray_scale()