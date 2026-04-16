# 5.2 Ablation Study: Đánh giá Baseline và Deep Ensemble

Trong phần này, chúng tôi tiến hành phân tích bóc tách (Ablation Study) nhằm đánh giá khả năng xử lý dữ liệu Out-of-Distribution (OOD) của mô hình Transformer cốt lõi (Baseline), đồng thời xem xét hiệu quả của phương pháp Deep Ensemble (DE) trong việc khắc phục điểm yếu của Baseline. Dựa trên Bảng tổng hợp Metrics và biểu đồ phân bố độ tin cậy, chúng tôi rút ra kết luận sau:

## 5.2.1. Sự thất bại của Baseline trước hiện tượng Overconfidence (Tự tin thái quá)

Mô hình Baseline (chỉ sử dụng một biến thể Transformer duy nhất) thể hiện sự yếu kém khi đối mặt với dữ liệu OOD. Hiệu năng của mô hình chỉ đạt **AUROC là 65.2%**, thuộc mức thấp đối với bài toán nhận diện ngoài phân phối, cùng Tỉ lệ lỗi phát hiện (**Detection Error**) lên tới **35.7%**.

Lý do cốt lõi cho sự thất bại này là hiện tượng **Overconfidence** phổ biến ở học sâu:
- Khi kiểm tra trên tập dữ liệu chuẩn (In-Distribution), nhận diện trả về độ tin cậy cực cao (gần 1.0) là điều hiển nhiên hợp lý.
- Tuy nhiên, khi đối diện tập dữ liệu rác/xáo trộn (OOD), Baseline **không chịu hạ độ tin cậy xuống**. Thay vì xuất ra phân phối đều để biểu thị sự "không chắc chắn", mô hình tiếp tục dựa vào một vài đặc trưng mạnh (strong features) học vẹt được để ép ra một dự đoán sai với **niềm tin cực kỳ cao** (score $\geq 0.95$). Điều này khiến ranh giới quyết định (decision boundary) giữa ID và OOD cực kỳ mờ nhạt.

## 5.2.2. Sự cải thiện và giới hạn bão hòa của Deep Ensemble (DE)

Để giảm thiểu tình trạng tự tin thái quá, kỹ thuật Deep Ensemble (tạo nhiều mô hình và kết hợp dự đoán) đã được áp dụng. Kết quả cho thấy DE mang lại hiệu ứng tích cực nhưng **chưa đủ đột phá**:

- **Mức độ cải thiện:** Deep Ensemble đã tăng **AUROC từ 65.2% lên 69.5%** và giảm nhẹ **Detection Error xuống còn 35.1%**.
- **Cơ chế cải thiện:** Có được nhờ phương sai giữa các thành viên. Đối với các mẫu OOD, các mô hình thành phần có thể "bất đồng". Khi mỗi mô hình (quá tự tin) tự đưa ra một kết quả khác nhau và sau đó bị kết hợp trung bình hóa, chỉ số tự tin trung bình cuối cùng bị kéo tụt đi. Ít nhất có một bộ phận dữ liệu OOD giảm tự tin, giúp cải thiện AUC đôi chút.
- **Tại sao Deep Ensemble bão hòa nhanh?** Con số AUROC $69.5\%$ chưa thể dùng trong thực tế. Lý do DE bão hòa là do **thiếu đa dạng về đặc trưng**. Tất cả các mô hình con trong bầu chọn vẫn có xu hướng bám vào cùng nhóm "đặc trưng mạnh" giống nhau. Khi một mẫu OOD đánh lừa được đặc trưng mạnh này, sự overconfidence **xảy ra đồng loạt** trên tất cả mô hình và hoàn toàn dập tắt hy vọng bầu chọn.

**Tiểu kết:**  
Baseline và Deep Ensemble thất bại triệt để trên OOD vì chỉ tập trung vào tín hiệu mạnh mà bỏ rơi các đặc trưng yếu. Sự bão hòa sớm của DE cho thấy cần một phương pháp bóc tách đặc trưng sâu hơn, và đó là tiền đề để áp dụng kiến trúc **Trust-Correction (TC)**, phương pháp đẩy AUROC nhảy vọt lên mức **84.0%**.
