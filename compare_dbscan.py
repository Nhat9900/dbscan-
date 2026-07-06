# -*- coding: utf-8 -*-
"""
So sánh DBSCAN custom với DBSCAN của sklearn

Lưu ý: Tôi khởi tạo DBSCAN custom để nhãn cluster bắt đầu từ 1, 
còn sklearn bắt đầu từ 0. Chúng ta sẽ hiệu chỉnh để so sánh công bằng.

@author: Nhat9900
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN as SklearnDBSCAN
from dbscan_custom import CustomDBSCAN


def compare_dbscan(X, eps, min_samples, dataset_name="Dataset"):
    """
    So sánh DBSCAN custom với sklearn
    
    Args:
        X: Dữ liệu
        eps: Tham số epsilon
        min_samples: Số điểm tối thiểu
        dataset_name: Tên dataset để in ra
    """
    print(f"\n{'='*60}")
    print(f"So sánh DBSCAN trên {dataset_name}")
    print(f"{'='*60}")
    print(f"Tham số: eps={eps}, min_samples={min_samples}")
    print(f"Số điểm dữ liệu: {X.shape[0]}, Số chiều: {X.shape[1]}")
    
    # ========== DBSCAN Custom ==========
    custom_dbscan = CustomDBSCAN(eps=eps, min_samples=min_samples)
    custom_labels = custom_dbscan.fit_predict(X)
    
    # Hiệu chỉnh: sklearn bắt đầu từ 0, custom bắt đầu từ 1
    # Để so sánh công bằng, ta trừ 1 khỏi nhãn custom (nhưng giữ -1 là noise)
    custom_labels_normalized = np.where(custom_labels == -1, -1, custom_labels - 1)
    
    # ========== DBSCAN sklearn ==========
    sklearn_dbscan = SklearnDBSCAN(eps=eps, min_samples=min_samples)
    sklearn_labels = sklearn_dbscan.fit_predict(X)
    
    # ========== Thống kê kết quả ==========
    n_clusters_custom = len(set(custom_labels)) - (1 if -1 in custom_labels else 0)
    n_clusters_sklearn = len(set(sklearn_labels)) - (1 if -1 in sklearn_labels else 0)
    
    n_noise_custom = list(custom_labels).count(-1)
    n_noise_sklearn = list(sklearn_labels).count(-1)
    
    print(f"\n🔹 DBSCAN Custom:")
    print(f"   - Số cluster được tìm thấy: {n_clusters_custom}")
    print(f"   - Số điểm noise: {n_noise_custom}")
    print(f"   - Nhãn clusters: {sorted(set(custom_labels_normalized))}")
    
    print(f"\n🔹 DBSCAN Sklearn:")
    print(f"   - Số cluster được tìm thấy: {n_clusters_sklearn}")
    print(f"   - Số điểm noise: {n_noise_sklearn}")
    print(f"   - Nhãn clusters: {sorted(set(sklearn_labels))}")
    
    # ========== So sánh chi tiết ==========
    n_disagree = np.sum(custom_labels_normalized != sklearn_labels)
    accuracy = (len(X) - n_disagree) / len(X) * 100
    
    print(f"\n✓ Kết quả so sánh:")
    print(f"   - Số điểm khác nhau: {n_disagree}")
    print(f"   - Tỷ lệ trùng khớp: {accuracy:.2f}%")
    
    if n_disagree == 0:
        print(f"\n✅ THÀNH CÔNG! Kết quả hoàn toàn giống nhau!")
    else:
        print(f"\n⚠️  CẢNH BÁO: Có {n_disagree} điểm khác nhau")
    
    return custom_labels_normalized, sklearn_labels


def plot_clusters(X, custom_labels, sklearn_labels, dataset_name="Dataset"):
    """
    Vẽ biểu đồ so sánh kết quả clustering
    
    Args:
        X: Dữ liệu 2D
        custom_labels: Nhãn từ DBSCAN custom
        sklearn_labels: Nhãn từ DBSCAN sklearn
        dataset_name: Tên dataset
    """
    if X.shape[1] != 2:
        print("⚠️  Không thể vẽ biểu đồ: dữ liệu không phải 2D")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # ========== Vẽ kết quả Custom DBSCAN ==========
    ax = axes[0]
    
    # Lấy các nhãn unique
    unique_labels = set(custom_labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    
    for label, color in zip(unique_labels, colors):
        if label == -1:
            # Điểm noise - màu đen
            color = [0, 0, 0, 1]
            marker = 'x'
            size = 100
            label_text = 'Noise'
        else:
            marker = 'o'
            size = 30
            label_text = f'Cluster {int(label)}'
        
        class_member_mask = (custom_labels == label)
        xy = X[class_member_mask]
        ax.scatter(xy[:, 0], xy[:, 1], c=[color], marker=marker, 
                  s=size, label=label_text, alpha=0.7, edgecolors='k')
    
    ax.set_title('DBSCAN Custom', fontsize=12, fontweight='bold')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # ========== Vẽ kết quả Sklearn DBSCAN ==========
    ax = axes[1]
    
    unique_labels = set(sklearn_labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    
    for label, color in zip(unique_labels, colors):
        if label == -1:
            color = [0, 0, 0, 1]
            marker = 'x'
            size = 100
            label_text = 'Noise'
        else:
            marker = 'o'
            size = 30
            label_text = f'Cluster {int(label)}'
        
        class_member_mask = (sklearn_labels == label)
        xy = X[class_member_mask]
        ax.scatter(xy[:, 0], xy[:, 1], c=[color], marker=marker, 
                  s=size, label=label_text, alpha=0.7, edgecolors='k')
    
    ax.set_title('DBSCAN Sklearn', fontsize=12, fontweight='bold')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'So sánh DBSCAN trên {dataset_name}', 
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(f'dbscan_comparison_{dataset_name.lower().replace(" ", "_")}.png', dpi=100, bbox_inches='tight')
    plt.show()
    
    print(f"✓ Biểu đồ đã được lưu: dbscan_comparison_{dataset_name.lower().replace(' ', '_')}.png")


# ============================================================================
# MAIN - Thực hiện so sánh
# ============================================================================

if __name__ == "__main__":
    print("🚀 DBSCAN Implementation & Comparison")
    print("="*60)
    
    # ========== Dataset 1: Gaussian Blobs ==========
    print("\n📊 DATASET 1: Gaussian Blobs")
    print("-" * 60)
    
    # Tạo 3 cụm Gaussian
    centers = [[1, 1], [-1, -1], [1, -1]]
    X1, _ = make_blobs(n_samples=300, centers=centers, cluster_std=0.6, random_state=42)
    
    # Chuẩn hóa dữ liệu
    X1 = StandardScaler().fit_transform(X1)
    
    # So sánh DBSCAN
    custom_labels1, sklearn_labels1 = compare_dbscan(X1, eps=0.5, min_samples=5, dataset_name="Gaussian Blobs")
    
    # Vẽ biểu đồ
    plot_clusters(X1, custom_labels1, sklearn_labels1, "Gaussian Blobs")
    
    # ========== Dataset 2: Two Moons ==========
    print("\n📊 DATASET 2: Two Moons")
    print("-" * 60)
    
    # Tạo dataset hình hai mặt trăng (non-convex)
    X2, _ = make_moons(n_samples=300, noise=0.05, random_state=42)
    
    # Chuẩn hóa dữ liệu
    X2 = StandardScaler().fit_transform(X2)
    
    # So sánh DBSCAN
    custom_labels2, sklearn_labels2 = compare_dbscan(X2, eps=0.3, min_samples=5, dataset_name="Two Moons")
    
    # Vẽ biểu đồ
    plot_clusters(X2, custom_labels2, sklearn_labels2, "Two Moons")
    
    # ========== Dataset 3: Random Data ==========
    print("\n📊 DATASET 3: Random Blobs (khác eps)")
    print("-" * 60)
    
    # Tạo dữ liệu ngẫu nhiên
    X3, _ = make_blobs(n_samples=500, centers=4, cluster_std=0.8, random_state=42)
    X3 = StandardScaler().fit_transform(X3)
    
    # So sánh với eps nhỏ
    custom_labels3, sklearn_labels3 = compare_dbscan(X3, eps=0.4, min_samples=10, dataset_name="Random Blobs")
    
    # Vẽ biểu đồ
    plot_clusters(X3, custom_labels3, sklearn_labels3, "Random Blobs")
    
    print("\n" + "="*60)
    print("✅ Hoàn thành! Tất cả kết quả đã được so sánh và lưu.")
    print("="*60)
