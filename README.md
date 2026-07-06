# DBSCAN Implementation & Comparison

🎯 **Mục đích:** Cài đặt DBSCAN từ đầu (không dùng sklearn) và so sánh kết quả với DBSCAN của sklearn.

## 📁 Cấu trúc Project

```
dbscan-/
├── dbscan_custom.py      # Cài đặt DBSCAN từ đầu
├── compare_dbscan.py     # So sánh với sklearn
└── README.md             # Hướng dẫn này
```

## 🔍 Giải thích DBSCAN

**DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** là một thuật toán clustering dựa trên mật độ.

### 📊 Các khái niệm chính:

1. **eps (epsilon)**: Bán kính vùng lân cận quanh mỗi điểm
2. **min_samples**: Số điểm tối thiểu trong vùng lân cận để tạo thành core point
3. **Core point**: Điểm có ít nhất `min_samples` điểm trong vùng lân cận (kể cả chính nó)
4. **Border point**: Điểm không phải core point nhưng nằm trong vùng lân cận của một core point
5. **Noise point**: Điểm không phải core point và không nằm trong vùng lân cận của core point nào

### 🔄 Quy trình thuật toán:

1. **Khởi tạo**: Tất cả điểm được đánh dấu là chưa xử lý (label = 0)
2. **Lặp qua từng điểm**:
   - Nếu điểm đã được xử lý, bỏ qua
   - Tìm tất cả điểm lân cận (khoảng cách <= eps)
   - Nếu số lân cận < min_samples → điểm này là noise (-1)
   - Nếu số lân cận >= min_samples → bắt đầu tạo cluster mới
3. **Mở rộng cluster**: Từ core point, duyệt tất cả lân cận:
   - Thêm lân cận vào cluster
   - Nếu lân cận là core point, tiếp tục mở rộng từ nó

### 📈 Ví dụ:

```
eps=0.5, min_samples=5

Diểm A: 6 lân cận (>= min_samples) → Core point → Tạo Cluster 1
Diểm B: 3 lân cận (< min_samples) → Noise (-1)
Diểm C: 5 lân cận (>= min_samples) → Core point → Tạo Cluster 2
```

## 🚀 Cách sử dụng

### 1️⃣ Cài đặt dependencies:

```bash
pip install numpy scikit-learn matplotlib
```

### 2️⃣ Chạy script so sánh:

```bash
python compare_dbscan.py
```

### 3️⃣ Output:

Script sẽ:
- ✅ So sánh kết quả trên 3 datasets khác nhau
- ✅ In ra số cluster, số noise, tỷ lệ trùng khớp
- ✅ Vẽ 3 biểu đồ so sánh trực quan
- ✅ Lưu biểu đồ thành file PNG

### 4️⃣ Sử dụng CustomDBSCAN trong code của bạn:

```python
from dbscan_custom import CustomDBSCAN
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

# Tạo dữ liệu
X, _ = make_blobs(n_samples=100, centers=3, random_state=42)
X = StandardScaler().fit_transform(X)

# Khởi tạo và huấn luyện DBSCAN
dbscan = CustomDBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(X)

print(f"Nhãn cluster: {labels}")
print(f"Số cluster: {len(set(labels)) - (1 if -1 in labels else 0)}")
print(f"Số noise: {list(labels).count(-1)}")
```

## 📊 Kết quả kỳ vọng

### Dataset 1: Gaussian Blobs
```
DBSCAN Custom:
   - Số cluster được tìm thấy: 3
   - Số điểm noise: 2
   
DBSCAN Sklearn:
   - Số cluster được tìm thấy: 3
   - Số điểm noise: 2
   
✅ THÀNH CÔNG! Kết quả hoàn toàn giống nhau!
```

### Dataset 2: Two Moons
```
DBSCAN Custom:
   - Số cluster được tìm thấy: 2
   - Số điểm noise: 5
   
DBSCAN Sklearn:
   - Số cluster được tìm thấy: 2
   - Số điểm noise: 5
   
✅ THÀNH CÔNG! Kết quả hoàn toàn giống nhau!
```

## 💡 Chú thích mã nguồn

### `dbscan_custom.py`:

| Phương thức | Mô tả |
|------------|------|
| `__init__` | Khởi tạo với `eps` và `min_samples` |
| `fit()` | Huấn luyện DBSCAN trên dữ liệu |
| `_get_neighbors()` | Tìm tất cả điểm lân cận trong vùng eps |
| `_expand_cluster()` | Mở rộng cluster từ core point |
| `fit_predict()` | Huấn luyện và trả về nhãn cluster |

### `compare_dbscan.py`:

| Hàm | Mô tả |
|-----|------|
| `compare_dbscan()` | So sánh custom DBSCAN với sklearn |
| `plot_clusters()` | Vẽ biểu đồ so sánh kết quả |

## 🔧 Tùy chỉnh

### Thay đổi tham số:

```python
# eps nhỏ hơn → cluster nhỏ hơn, nhiều noise hơn
compare_dbscan(X, eps=0.3, min_samples=5, dataset_name="Test")

# min_samples lớn hơn → ít cluster hơn
compare_dbscan(X, eps=0.5, min_samples=10, dataset_name="Test")
```

### Sử dụng dataset khác:

```python
from sklearn.datasets import load_iris

# Tải iris dataset
iris = load_iris()
X = iris.data

# Chuẩn hóa
X = StandardScaler().fit_transform(X)

# So sánh
compare_dbscan(X, eps=0.5, min_samples=5, dataset_name="Iris")
```

## 📝 Chi tiết Implementation

### Độ phức tạp thuật toán:

- **Thời gian**: O(n²) trong trường hợp xấu (n = số điểm)
  - Với mỗi điểm, cần tìm tất cả lân cận
  - Có thể tối ưu bằng spatial index (KD-tree, Ball-tree)
  
- **Không gian**: O(n)
  - Lưu trữ nhãn cluster cho n điểm

### Ưu điểm:

✅ Không cần biết số cluster trước  
✅ Phát hiện được noise  
✅ Xử lý tốt các cluster non-convex  
✅ Dễ hiểu và implement  

### Nhược điểm:

❌ Độ phức tạp O(n²)  
❌ Nhạy cảm với tham số eps và min_samples  
❌ Khó xử lý dữ liệu high-dimensional  
❌ Khó xử lý các cluster có mật độ khác nhau  

## 🎓 Học hỏi thêm

- 📖 [DBSCAN Wikipedia](https://en.wikipedia.org/wiki/DBSCAN)
- 📖 [Scikit-learn DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)
- 📖 [DBSCAN Original Paper](https://en.wikipedia.org/wiki/DBSCAN#Original_paper)

## 📄 License

MIT License - Tự do sử dụng cho mục đích học tập

## 👨‍💻 Tác giả

Nhat9900 - 2026
