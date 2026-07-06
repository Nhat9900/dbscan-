# -*- coding: utf-8 -*-
"""
Cài đặt DBSCAN từ đầu (không dùng sklearn)
Có chú thích tiếng Việt chi tiết

@author: Nhat9900
"""

import numpy as np
from typing import List, Tuple


class CustomDBSCAN:
    """
    Lớp cài đặt thuật toán DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
    
    DBSCAN là thuật toán clustering dựa trên mật độ:
    - Các điểm được nhóm lại dựa trên khoảng cách và mật độ
    - Điểm noise (-1) là những điểm không thuộc cluster nào
    - Cluster được đánh số từ 0, 1, 2, ...
    """
    
    def __init__(self, eps: float = 0.5, min_samples: int = 5):
        """
        Khởi tạo DBSCAN
        
        Args:
            eps: Khoảng cách epsilon - bán kính vùng lân cận
            min_samples: Số điểm tối thiểu trong vùng lân cận để tạo thành core point
        """
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None
    
    def fit(self, X: np.ndarray) -> 'CustomDBSCAN':
        """
        Huấn luyện DBSCAN trên dữ liệu X
        
        Args:
            X: Mảng numpy chứa dữ liệu (n_samples, n_features)
            
        Returns:
            self: Trả về đối tượng CustomDBSCAN
        """
        # Lưu số lượng điểm dữ liệu
        n_samples = X.shape[0]
        
        # Khởi tạo mảng nhãn: 0 = chưa được xử lý, -1 = noise, >= 1 = cluster ID
        self.labels_ = np.zeros(n_samples, dtype=int)
        
        # Biến đếm cluster hiện tại
        cluster_id = 0
        
        # Lặp qua từng điểm dữ liệu
        for point_idx in range(n_samples):
            # Nếu điểm đã được gán nhãn (không phải 0), bỏ qua
            if self.labels_[point_idx] != 0:
                continue
            
            # Tìm tất cả điểm lân cận của điểm hiện tại
            neighbors = self._get_neighbors(X, point_idx)
            
            # Nếu số điểm lân cận < min_samples, đánh dấu là noise
            if len(neighbors) < self.min_samples:
                self.labels_[point_idx] = -1
            else:
                # Đây là core point - bắt đầu tạo cluster mới
                cluster_id += 1
                # Mở rộng cluster từ điểm này
                self._expand_cluster(X, self.labels_, point_idx, neighbors, cluster_id)
        
        return self
    
    def _get_neighbors(self, X: np.ndarray, point_idx: int) -> List[int]:
        """
        Tìm tất cả các điểm lân cận trong bán kính eps
        
        Args:
            X: Dữ liệu
            point_idx: Chỉ số của điểm cần tìm lân cận
            
        Returns:
            List chứa chỉ số của các điểm lân cận
        """
        # Lấy tọa độ của điểm cần xét
        point = X[point_idx]
        
        # Tính khoảng cách từ điểm này đến tất cả các điểm khác
        distances = np.linalg.norm(X - point, axis=1)
        
        # Trả về chỉ số của các điểm có khoảng cách <= eps
        neighbors = np.where(distances <= self.eps)[0].tolist()
        
        return neighbors
    
    def _expand_cluster(self, X: np.ndarray, labels: np.ndarray, 
                       point_idx: int, neighbors: List[int], cluster_id: int):
        """
        Mở rộng cluster từ một core point
        
        Args:
            X: Dữ liệu
            labels: Mảng nhãn
            point_idx: Chỉ số core point
            neighbors: Danh sách điểm lân cận
            cluster_id: ID của cluster hiện tại
        """
        # Gán cluster_id cho core point
        labels[point_idx] = cluster_id
        
        # Sử dụng FIFO queue để xử lý các điểm lân cận
        # Danh sách neighbors sẽ phát triển khi ta tìm thêm core points
        queue_idx = 0
        
        while queue_idx < len(neighbors):
            # Lấy điểm tiếp theo từ hàng đợi
            neighbor_idx = neighbors[queue_idx]
            
            # Nếu điểm này là noise, chuyển thành điểm biên của cluster
            if labels[neighbor_idx] == -1:
                labels[neighbor_idx] = cluster_id
            
            # Nếu điểm này chưa được gán nhãn
            elif labels[neighbor_idx] == 0:
                # Gán cluster_id cho điểm này
                labels[neighbor_idx] = cluster_id
                
                # Tìm tất cả lân cận của điểm này
                neighbor_neighbors = self._get_neighbors(X, neighbor_idx)
                
                # Nếu có đủ lân cận, nó cũng là core point - thêm lân cận vào queue
                if len(neighbor_neighbors) >= self.min_samples:
                    # Thêm các lân cận mới vào danh sách để tiếp tục xử lý
                    neighbors.extend(neighbor_neighbors)
            
            # Tiến tới phần tử tiếp theo
            queue_idx += 1
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Huấn luyện và dự đoán nhãn cluster trong một bước
        
        Args:
            X: Dữ liệu
            
        Returns:
            Mảng nhãn cluster
        """
        self.fit(X)
        return self.labels_
