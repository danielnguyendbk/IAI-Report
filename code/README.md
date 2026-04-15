# README - Chương 4 (Phần A): Data Pipeline & OOD Evaluation

Repo này là phần việc của **Thái** cho đề tài OOD Detection.

Mục tiêu chính:
1. Load dataset
2. Sinh dữ liệu OOD giả lập bằng noise/shuffle/mask
3. Tính các metric OOD (AUROC, Detection Error, AUPR)
4. Tổng hợp kết quả từ Baseline, Deep Ensemble (DE), Transformer Chain (TC) bằng `compare.py`

> Lưu ý phạm vi: Branch này **không** dùng để train toàn bộ model.
>
> - Train model thuộc về Hiệp (B): Single Transformer + Deep Ensemble
> - Train model thuộc về An (C): Transformer Chain (TC)

Repo phần A chủ yếu phục vụ:
- **Chương 4.1**: Data Pipeline & Tạo OOD
- **Chương 5.1**: Biểu đồ & Bảng Metrics

## Mục lục
- [1. Giới thiệu](#1-giới-thiệu)
- [2. Cấu trúc thư mục](#2-cấu-trúc-thư-mục)
- [3. Chức năng từng file](#3-chức-năng-từng-file)
- [4. Tình trạng hiện tại](#4-tình-trạng-hiện-tại)
- [5. Cài môi trường](#5-cài-môi-trường)
- [6. Hướng dẫn chạy từng bước (Windows)](#6-hướng-dẫn-chạy-từng-bước-windows)
- [7. Định dạng file score từ B và C](#7-định-dạng-file-score-từ-b-và-c)
- [8. Cách chạy compare.py](#8-cách-chạy-comparepy)
- [9. Luồng phối hợp trong nhóm](#9-luồng-phối-hợp-trong-nhóm)
- [10. Ghi chú học thuật](#10-ghi-chú-học-thuật)
- [11. Troubleshooting](#11-troubleshooting)
- [12. Kết luận](#12-kết-luận)

## 1. Giới thiệu

Đây là code phần A cho pipeline dữ liệu và đánh giá OOD.

Dataset hiện tại để test pipeline là **Arrhythmia bản raw từ UCI đã làm sạch**.

> Lưu ý quan trọng: Bản raw UCI đang dùng nhằm kiểm tra pipeline và luồng đánh giá. Dữ liệu này **chưa chắc trùng hoàn toàn** với phiên bản Arrhythmia trong paper/repo gốc của tác giả.

## 2. Cấu trúc thư mục

```text
code/
├── data/
│   └── Arrhythmia_raw_clean.csv
├── raw_data/
│   ├── arrhythmia.data
│   └── arrhythmia.names
├── results/
│   └── figures/
├── dataset.py
├── ood_generator.py
├── evaluate_metrics.py
├── compare.py
├── CSV_Converter.py
├── run_ood_demo.py
└── README.md
```

## 3. Chức năng từng file

- `dataset.py`
	- Load file CSV sạch
	- Tách feature/label
	- Chia train/val/test
	- Chuẩn hóa dữ liệu

- `ood_generator.py`
	- Sinh OOD giả từ dữ liệu ID
	- Hỗ trợ các kiểu: `noise`, `shuffle`, `mask`

- `evaluate_metrics.py`
	- Tính AUROC
	- Tính Detection Error
	- Tính AUPR

- `compare.py`
	- Đọc score CSV của Baseline / DE / TC
	- Tổng hợp bảng metrics
	- Vẽ ROC curve
	- Vẽ histogram confidence

- `CSV_Converter.py`
	- Hỗ trợ chuyển Arrhythmia raw sang CSV sạch

- `run_ood_demo.py`
	- Chạy test end-to-end phần A: load dữ liệu + sinh OOD

- `results/`
	- Chứa output đánh giá và các file score đầu vào

## 4. Tình trạng hiện tại

- `dataset.py` đã chạy được
- `ood_generator.py` đã chạy được
- `evaluate_metrics.py` đã chạy được
- `run_ood_demo.py` đã chạy được
- `compare.py` đã viết xong, cần file score thật (hoặc score giả) để chạy
- `results/` hiện là nơi chứa output và file score đầu vào

## 5. Cài môi trường

Yêu cầu:
- Python **3.13**

Cài thư viện:

```bash
pip install numpy pandas scikit-learn matplotlib
```

## 6. Hướng dẫn chạy từng bước (Windows)

Mở terminal tại thư mục `code/`, sau đó chạy theo thứ tự sau.

### Bước 1: Chuẩn bị dữ liệu

- `raw_data/` chứa `arrhythmia.data` và `arrhythmia.names`
- Dùng `CSV_Converter.py` để tạo CSV sạch nếu cần
- File CSV sạch hiện tại đặt tại `data/Arrhythmia_raw_clean.csv`

Lệnh (khi cần convert lại):

```bash
python CSV_Converter.py
```

### Bước 2: Kiểm tra pipeline load dataset

```bash
python dataset.py
```

Kỳ vọng:
- In ra shape của Train / Val / Test

### Bước 3: Kiểm tra sinh OOD

```bash
python ood_generator.py
```

### Bước 4: Kiểm tra tính metric

```bash
python evaluate_metrics.py
```

### Bước 5: Kiểm tra pipeline end-to-end phần A

```bash
python run_ood_demo.py
```

Kỳ vọng:
- In ra shape của Train, Val, Test ID, Test OOD

### Bước 6: Tổng hợp kết quả thật bằng compare.py

`compare.py` chỉ chạy đầy đủ khi đã có các file score CSV trong thư mục `results/` (do B và C cung cấp).

## 7. Định dạng file score từ B và C

Các file bắt buộc:

- `results/baseline_id.csv`
- `results/baseline_ood.csv`
- `results/de_id.csv`
- `results/de_ood.csv`
- `results/tc_id.csv`
- `results/tc_ood.csv`

Mỗi file chỉ có **1 cột**:

```csv
score
0.91
0.88
0.84
```

Quy ước score:
- Score càng cao  => càng giống ID
- Score càng thấp => càng giống OOD

## 8. Cách chạy compare.py

```bash
python compare.py
```

Output mong đợi:
- `results/metrics_summary.csv`
- `results/figures/roc_comparison.png`
- `results/figures/hist_baseline.png`
- `results/figures/hist_deep_ensemble.png`
- `results/figures/hist_tc.png`

## 9. Luồng phối hợp trong nhóm

- **Thái (A)**
	- Chuẩn bị dữ liệu
	- Sinh OOD
	- Tính metric
	- Tổng hợp biểu đồ và bảng kết quả

- **Hiệp (B)**
	- Train Single Transformer
	- Train Deep Ensemble
	- Xuất score CSV cho Baseline và DE

- **An (C)**
	- Train TC
	- Xuất score CSV cho TC

## 10. Ghi chú học thuật

- Repo này phục vụ chính cho **Chương 4.1** và **Chương 5.1**
- Dữ liệu Arrhythmia hiện tại là bản raw UCI đã làm sạch để chạy thử pipeline
- Nếu cần bám paper sát hơn, nhóm cần:
	- Dùng đúng phiên bản dữ liệu theo repo/tác giả
	- Hoặc nêu rõ trong báo cáo đây là bản tái lập từ UCI

## 11. Troubleshooting

### Lỗi: `FileNotFoundError: data/train.csv not found`

Nguyên nhân thường gặp:
- Chạy sai thư mục làm việc
- File dữ liệu đầu vào không đúng tên/đúng chỗ

Cách xử lý:
- Đảm bảo đang chạy lệnh trong thư mục `code/`
- Kiểm tra file CSV sạch có ở `data/Arrhythmia_raw_clean.csv`
- Nếu cần, chạy lại:

```bash
python CSV_Converter.py
```

### Lỗi: file score không tồn tại trong `results/`

Cách xử lý:
- Kiểm tra đã nhận đủ 6 file score từ B/C chưa
- Kiểm tra đúng tên file theo mục 7

### Lỗi: CSV có header lạ hoặc ký tự `?` gây lỗi

Cách xử lý:
- Làm sạch dữ liệu bằng `CSV_Converter.py`
- Mở CSV kiểm tra encoding/header
- Đảm bảo dữ liệu số không chứa ký tự lỗi

### Lỗi: score CSV thiếu cột `score`

Cách xử lý:
- Đảm bảo file chỉ có 1 cột tên chính xác là `score`
- Không dùng tên cột khác (ví dụ `conf`, `prob`, ...)

### Lỗi: `compare.py` không chạy vì chưa có kết quả từ B/C

Cách xử lý:
- Đây là trạng thái bình thường nếu B/C chưa xuất score
- Tạm thời có thể dùng score giả để kiểm tra pipeline vẽ/đọc file

## 12. Kết luận

Phần A đã hoàn thành pipeline dữ liệu và đánh giá nền.

Sau khi B và C xuất score, A chỉ cần chạy:

```bash
python compare.py
```

để tạo bảng metrics và biểu đồ phục vụ **Chương 5.1**.
