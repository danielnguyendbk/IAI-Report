# BÍ KÍP ÔN TẬP BẢO VỆ ĐỒ ÁN: OUT-OF-DISTRIBUTION (OOD) DETECTION
*(Tài liệu lưu hành nội bộ dành riêng cho bạn Hiệp - Nắm trùm toàn bộ kỹ thuật và Kịch bản Demo)*

---

## PHẦN 1. LÝ THUYẾT NỀN TẢNG (Phải thuộc làu)

### 1. Cuộc chiến giữa ID và OOD
- **ID (In-Distribution - Dữ liệu quen):** Là những dữ liệu có cấu trúc y hệt như những gì con AI đã được nhồi nhét lúc đi học (train). Khi đưa ID cho AI, nó nhận diện đúng và gáy rất to (Max Confidence).
- **OOD (Out-of-Distribution - Dữ liệu lạ/rác):** Là dữ liệu AI chưa từng thấy bao giờ. Có thể là một bản ghi điện tâm đồ của bệnh nhân mắc căn bệnh lạ hoắc, hoặc một tờ giấy nhiễu sóng điện từ.

### 2. Kẻ phản diện: Bệnh Ocerconfidence (Tự tin thái quá)
Khi AI đối mặt với ID, nó tự tin 99% là một điều tốt. 
Nhưng khi AI đối mặt với một tờ giấy rác (OOD), lẽ ra nó phải sợ hãi và hạ điểm tự tin xuống (ví dụ 10-20%) để báo động cho con người *"Tôi không biết, có biến!"*.
Thực tại phũ phàng là các mạng Deep Learning như Transformer hiện nay bị mắc định kiến **Overconfidence**: Khi gặp OOD, nó không chịu nhận dốt, nó tình cờ bám vào 1-2 vệt nhiễu trên tờ giấy, rồi võ đoán ngay lập tức với sự chắc nịch lên tới **99%**. Trong lĩnh vực y tế, một con AI chẩn đoán bừa nhưng mạnh miệng kiểu này sẽ gây chết người.

---

## PHẦN 2. DATASET (Tài nguyên dữ liệu)

- **Tên Dataset:** Arrhythmia (Dữ liệu Nhịp tim / Điện tâm đồ từ UCI).
- **Đặc điểm:** Đây là dữ liệu **Dạng bảng (Tabular)**, không phải dữ liệu Ảnh hay Chữ.
- **Input:** 279 cột (Nghĩa là quét 1 bệnh nhân sẽ thu được 279 thông số chiều cao, nhịp sóng, khoảng cách điện tim...).
- **Output:** Có 13 loại nhãn (Từ class 0 là bình thường, đến các vách ngăn suy tim, thiếu máu cục bộ...).

---

## PHẦN 3. PHẪU THUẬT AI (Phân công và Thuật toán của Nhóm)

Thái (Trọng tài) đã lo phần sinh số rác và chấm điểm qua file CSV. Còn đây là cuộc đấu trí của Bạn và An:

### Kẻ lót đường 1: Baseline (Single Transformer) - Phần của bạn
- **Cách hoạt động:** Dùng 1 mạng AI duy nhất (Single Transformer) cày qua 279 cột.
- **Vì sao nó ngu với OOD?** Nó quá lười! Nó bị dính hiệu ứng *Information Bottleneck (Nút thắt thông tin)*. Nó chỉ chăm chăm tìm vài cột thông số bự nhất, tương quan rõ ràng nhất (**Strong Features**) để đi thi. Các đặc trưng mờ nhạt li ti (**Weak Features**) nó học bị vứt xó.
- **Hậu quả:** Khi gặp giấy rác có làm giả đặc trưng bự, Baseline sập bẫy, gáy to 99% => Overconfidence. Điểm AUROC chỉ lẹt đẹt **65.2%**.

### Kẻ lót đường 2: Deep Ensemble (DE) - Phần của bạn
- **Cách hoạt động:** Gọi vỗ mặt 5 con Transformer (Độc lập, số random khác hẳn nhau) vô cùng dự đoán. Lấy trung bình điểm của 5 đứa. Giống kiểu hội đồng bầu chọn bác sĩ.
- **Vì sao nó có hiệu quả nhẹ?** Khi gặp mẫu OOD, 5 ông AI này hơi "bất đồng quan điểm" một tí do lúc đẻ ra ngẫu nhiên khác nhau. Đứa vạch lỗi này, đứa chỉ lỗi kia, chửi qua chửi lại khiến điểm tự tin trung bình sụt xuống một chút xíu. Nhờ vậy nó đỡ Overconfident hơn (bớt gáy lại).
- **Vì sao nó lại BÃO HÒA (Phế)?** Dù là 5 người khác nhau, nhưng cả 5 đứa rủ nhau... chép phao. Tất cả đều tìm đường tắt (shortcut), vẫn chỉ rình học dăm ba cái Strong Features bự bự. Gặp mồi ngon giăng sẵn của mẫu OOD, cả 5 đứa đều đồng thanh *"Ờ đúng bệnh này rồi"*. Bão hòa tuyệt đối! Cải thiện AUROC lên **69.5%** nhưng chẳng đủ xài.

### Ngôi sao sáng chói: Transformer Chain (TC) - Phần của An
- **Cách hoạt động:** Nó xây một "Trại giam" dài hằng hà sa số các Transformer nối đuôi nhau.
- **Vũ khí tối thượng - Khôi phục Clues:** 
  + Transformer 1 thi xong, code rà soát thấy mảng nào là Đặc trưng mạnh (Strong). Lập tức **Giết/Xóa xổ/Bịt mắt** phần đó lại.
  + Transformer 2 thức dậy, bị bịt mắt mất các lối đi tắt quen thuộc, AI gầm thét vì nó BẮT BUỘC phải cày mòn mắt để học các chỉ số cực kỳ nhỏ bé, vụn vặt (Weak Features).
- **Kết quả:** Vì đã thuộc lòng cả những ngóc ngách siêu nhỏ, khi giấy rác giăng bẫy OOD, bẫy đó làm sao làm giả được cái vụn vặt? AI lập tức đánh mùi ra: *"Láo toét, đây không phải bệnh nhân!"*. Điểm tự tin rớt đài thê thảm, OOD bị tóm cổ thành công! AUROC lấp lánh **84.0%**.

---

## PHẦN 4. KỊCH BẢN MÚA DEMO (Predict Baseline & DE)

Khi nãy bạn đã Train xong (gần mất 1 tiếng chờ đợi đẻ ra 2 file `baseline.pt` và `de_ensemble_full.pt`). Tới ngày thuyết trình lên đứng bục, bạn trỏ chuột gọi lệnh:
```bash
python code/predict_baseline_de.py
```
*(Trên màn hình báo `Chọn số 1. Chạy Demo`). Bạn gõ phím 1 và bắt đầu thao thao bất tuyệt theo mẫu sau:*

**Màn hình hiện kịch bản 1: [MAU CHUAN (IN-DISTRIBUTION) - Bệnh thật]**
- Baseline ra điểm `99.9%`
- Deep Ensemble ra điểm `99.8%`
> **🎤 Lời bạn bình luận:** *"Dạ thưa Thầy/Cô, khi đưa người bệnh quen thuộc (In-Distribution), cả con AI truyền thống và 5 mạng hội đồng (Ensemble) đều nhận diện cực kỳ xuất sắc với điểm tự tin tuyệt đối gần 100%. Mọi thứ vẫn rất tươi đẹp..."*

**Màn hình hiện kịch bản 2 & 3: [MAU OOD: DỮ LIỆU RÁC GAUSSIAN VÀ XÁO TRỘN]**
- Baseline nhảy loạn xạ ra một bệnh gì đó với điểm `95-99%`. 
- Deep Ensemble hụt hơi ra điểm thấp hơn, cỡ `85-90%`.
> **🎤 Lời bạn bình luận:** *"Nhưng thảm họa xảy ra ở đây ạ! Khi em cấp cho nó một tờ giấy nhiễu sóng điện từ (Dữ liệu rác - OOD hoàn toàn). Thay vì nói 'Không biết', con AI Baseline kinh điển vẫn tự tin 99% chẩn đoán bậy bạ rằng bệnh nhân bị 'Thiếu máu cục bộ'. Nó vô tình tạo ra thảm họa y tế (Overconfidence).*
> *Kế tiếp, nhóm em thử cải tiến bằng cách áp dụng Deep Ensemble 5 hội đồng. Kết quả có khá hơn thưa các Thầy Cô! Do bất đồng bầu chọn, độ tự tin giả mạo đã tụt xuống còn 85%. Nhưng 85% vẫn là một cái mốc quá cao và nguy hiểm, chứng tỏ thuật toán Ensemble đã bị bão hòa do học chung tập đặc trưng bịnh (Strong features).*
> *Đó chính là trăn trở và bàn đạp để bạn An sau đây giới thiệu thuật toán xé nhỏ đặc trưng (Transformer Chain), giải thoát hoàn toàn giới hạn hiện tại của AI nhóm em ạ!"*

Bạn pass mic nhường sân khấu vô cùng ngầu cho An lên diễn phần TC. Hết nước chấm!💯
