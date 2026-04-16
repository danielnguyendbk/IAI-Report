# 6.1. Trade-off & Nhược điểm của phương pháp

Mặc dù Transformer Chain (TC) đã chứng minh được hiệu quả vượt trội trong việc phát hiện dữ liệu OOD (AUROC đạt 84.0%, cao hơn Baseline 18.8% và Deep Ensemble 14.5%), phương pháp này vẫn tồn tại một số hạn chế và sự đánh đổi (trade-off) cần được xem xét kỹ lưỡng trước khi áp dụng vào thực tế. Việc nhận diện những nhược điểm này không làm giảm giá trị của TC, mà ngược lại, giúp định hướng phát triển và cải tiến trong tương lai.

## 6.1.1. Đánh đổi giữa độ chính xác trên ID và khả năng phát hiện OOD
Một quan sát quan trọng từ thực nghiệm là độ chính xác trên tập ID (In-Distribution) của TC có thể giảm nhẹ so với Baseline hoặc Deep Ensemble.

| Mô hình | Accuracy trên ID (khoảng) | AUROC phát hiện OOD |
| :--- | :--- | :--- |
| **Baseline** (Single Transformer) | ~96 - 97% | 65.2% |
| **Deep Ensemble** | ~98 - 100% | 69.5% |
| **Transformer Chain** (TC) | ~94 - 95% | 84.0% |

**Giải thích nguyên nhân:** Baseline và DE chỉ tập trung vào *strong features* (những đặc trưng có tương quan mạnh nhất với nhãn), giúp đạt độ chính xác cực cao trên ID nhưng lại "mù" trước OOD. Ngược lại, TC buộc mô hình phải vắt kiệt và phân tích cả *weak features*. Điều này giúp TC nhạy bén hơn hẳn với OOD, nhưng đôi khi lượng thông tin yếu này gây nhiễu nhẹ cho quá trình phân loại ID, dẫn đến accuracy giảm từ 1-3%.

**Đánh giá:** Đây là một sự đánh đổi có chủ đích. Trong y tế (ví dụ: chẩn đoán đột quỵ), việc khăng khăng tin vào một dự đoán sai (OOD) có thể gây tai biến chết người, trong khi chẩn đoán nhầm một ca bệnh thông thường (ID) hoàn toàn có thể phát hiện và điều chỉnh ở các bước xét nghiệm sau. Do đó, hy sinh 1-3% accuracy ID để đổi lấy ~19% năng lực phòng vệ OOD là một trade-off hoàn toàn xứng đáng.

## 6.1.2. Chi phí tính toán và thời gian huấn luyện
TC sử dụng chuỗi các Transformer phụ thuộc lẫn nhau, do đó chi phí tính toán cao hơn đáng kể.

| Mô hình | Số lượng Transformer | Thời gian train (tương đối) | Yêu cầu Bộ nhớ (Memory) |
| :--- | :--- | :--- | :--- |
| **Baseline** | 1 | 1x | 1x |
| **Deep Ensemble** | 5 (độc lập) | ~5x (Có thể song song hóa) | ~5x |
| **Transformer Chain** | $T = \lfloor \log_{1/q} n \rfloor$ (Ví dụ: 8) | ~8-10x (Tuần tự, mút nối tiếp) | ~2-3x |

**Hệ quả:** Do các Transformer trong TC phụ thuộc tuần tự (mạng sau cần đầu ra của mạng trước), mô hình **không thể train song song** như Deep Ensemble. Nếu mỗi mạng train 40 epochs, TC cần tổng cộng 320 epochs tuần tự. Phương pháp này chỉ thực sự phù hợp với các hệ thống có thể train Offline thong thả rồi mới mang đi Deploy.

## 6.1.3. Yêu cầu tài nguyên bộ nhớ
Mặc dù TC có cơ chế giảm dần số lượng đặc trưng qua các vòng lặp (Feature Decomposition) giúp tổng bộ nhớ chỉ gấp khoảng 2-3 lần Baseline (ưu việt hơn mức đội bộ nhớ x5 của Deep Ensemble), nhưng nó vẫn nặng nề hơn Baseline tiêu chuẩn. Với các dataset siêu cao chiều (hàng chục nghìn đặc trưng), bắt buộc phải kết hợp thêm kỹ thuật giảm chiều (PCA, Autoencoder) trước khi đưa vào TC.

## 6.1.4. Hạn chế về kiến trúc: Chưa tối ưu cho dữ liệu ảnh
Như đề cập trong Section 6.3 của bài báo gốc, TC được thiết kế rành mạch cho dữ liệu dạng bảng (Tabular data), nơi mỗi cột/đặc trưng mang một ý nghĩa ngữ nghĩa rõ ràng (ví dụ: tuổi, huyết áp). Trên dữ liệu ảnh, một pixel đơn lẻ hầu như không mang ý nghĩa ngữ cảnh độc lập, khiến cơ chế xếp hạng đặc trưng yếu/mạnh của TC bị mất phương hướng (thực tế chỉ cải thiện ~4.8% trên CIFAR10).

## 6.1.5. Phụ thuộc vào siêu tham số $q$ (Tỷ lệ phân chia)
Hiệu quả của TC phụ thuộc vào việc lựa chọn tham số $q$ (tỉ lệ cắt bỏ đặc trưng sau mỗi vòng).

| Giá trị $q$ | Số lượng Transformer (T) | Đặc điểm |
| :--- | :--- | :--- |
| $q = 0.3$ (Nhỏ) | $T \approx 5-6$ | Train nhanh hơn, nhưng có thể cắt lẹm và bỏ sót weak clues. |
| $q = 0.5$ (Trung bình) | $T \approx 8$ | Cân bằng lý tưởng giữa hiệu năng và chi phí (Khuyến nghị). |
| $q = 0.7$ (Lớn) | $T \approx 12-14$ | OOD Score có thể tốt nhất, nhưng chi phí train phình to. |

## 6.1.6. Kết luận về Trade-off
Transformer Chain là một phương pháp mạnh nhưng "không hề miễn phí". Sự cải thiện nhảy vọt trong phát hiện OOD đi kèm với cái giá phải trả về độ phức tạp huấn luyện và sự hi sinh nhỏ nhoi ở accuracy ID. Tuy nhiên, đặt trong bối cảnh các hệ thống rủi ro cao (AI Y tế, AI Tài chính), nơi ưu tiên hàng đầu là **tính an toàn và khước từ dự đoán bừa**, sự đánh đổi này là bước lùi cần thiết để tiến xa hơn.
