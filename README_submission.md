# Lab Day 2 — DeepWeeds Classification Pipeline

Thư mục này chứa toàn bộ mã nguồn, dữ liệu dự đoán và báo cáo cho Bài lab Day 2: *Backbone, Công thức huấn luyện và Suy luận trên tập dữ liệu DeepWeeds*.

---

## 1. Thông tin chung & Seed
* **Họ và tên:** Nguyễn Thành Nam
* **MSSV:** 2A202602827
* **Link Notebook (chạy lại được):** https://www.kaggle.com/code/nguynthnhnamm/track-4-day-2
* **Cố định Seed (Random Seeds):**
  * **Sàng lọc (Bước 1 & 2):** Seed `0` (hoặc `42`) dùng chung cho tất cả các cấu hình để so sánh công bằng.
  * **Chung kết (Bước 4):** Sử dụng 3 seeds cố định: `42`, `43`, `44` (Báo cáo kết quả dạng `mean ± std`).

---

## 2. Phiên bản thư viện & Môi trường
* **Hệ điều hành / Nền tảng:** Kaggle Notebook (GPU T4 x2)
* **Python:** `3.10.x`
* **PyTorch:** `2.0.1+cu118`
* **Torchvision:** `0.15.2+cu118`
* **timm:** `0.9.2`
* **thop:** `0.1.1` (dùng để đếm GMACs)
* **pandas:** `2.0.3` | **numpy:** `1.24.3` | **scipy:** `1.10.1`

---

## 3. Cấu trúc thư mục bài nộp

```text
submissions/
├── README_submission.md          # File hướng dẫn chạy lại thí nghiệm
├── results.xlsx                  # Bảng tổng hợp chi tiết tất cả thí nghiệm
├── report.md                     # Báo cáo phân tích chi tiết (4-8 trang)
├── curves/            # Biểu đồ đường cong training (.png) của từng exp_id
├── predictions/       # File dự đoán CSV (.csv) phục vụ chấm điểm tự động
└── code/              # Mã nguồn đã hoàn thiện
    ├── dataset.py     # Đọc dữ liệu, chia fold, augmentation & DataLoader
    ├── model.py       # Khởi tạo backbone (timm), đóng băng & đếm params/GMAC
    ├── losses.py      # Cài đặt Focal Loss, Label Smoothing, Mixup/CutMix
    ├── train.py       # Pipeline huấn luyện chuẩn (AMP, EMA, Cosine LR)
    ├── inference.py   # Các kỹ thuật TTA, Ensemble, Temperature Scaling, Fuse BN
    ├── benchmark.py   # Đo độ trễ suy luận (p50, p95, p99) đúng chuẩn GPU
    └── eval.py        # Script chấm điểm và tính chỉ số chính thức (giữ nguyên)

```

## 4. Hướng dẫn chạy lại thí nghiệm (Thứ tự thực hiện)

Tất cả lệnh được thực hiện tại thư mục gốc chứa dự án (code/).

### Bước 0: Chuẩn bị dữ liệu & Kiểm tra Pipeline

Tải ảnh từ Zenodo và nhãn CSV từ Github tác giả vào thư mục `data/`:

```bash
# Tải ảnh và giải nén
wget "https://zenodo.org/records/7939060/files/images.zip?download=1" -O images.zip
unzip -q images.zip -d data/images/ && rm images.zip

# Tải nhãn CSV
git clone https://github.com/AlexOlsen/DeepWeeds.git temp_repo
cp temp_repo/labels/*.csv data/labels/ && rm -rf temp_repo
```

### Bước 1: So sánh Backbone (≥ 5 kiến trúc)

Huấn luyện 5 backbone khác nhau với cùng một công thức nền (Baseline, 12 epoch, seed 0):

```bash
python train.py --set exp_id=B01 backbone=resnet50 epochs=12 seed=0
python train.py --set exp_id=B02 backbone=resnext50 epochs=12 seed=0
python train.py --set exp_id=B03 backbone=convnext_tiny epochs=12 seed=0
python train.py --set exp_id=B04 backbone=swin_tiny epochs=12 seed=0
python train.py --set exp_id=B05 backbone=efficientnet_b0 epochs=12 seed=0
```

### Bước 2: Thử nghiệm Công thức huấn luyện

Chọn backbone tốt nhất (convnext_tiny), thử nghiệm từng thay đổi một:

```bash
# Đổi Hàm loss (Focal Loss)
python train.py --set exp_id=T01 backbone=convnext_tiny loss=focal focal_gamma=2.0 seed=0

# Đổi Augmentation (CutMix)
python train.py --set exp_id=T02 backbone=convnext_tiny mix=cutmix mix_alpha=1.0 seed=0

# Tổ hợp tốt nhất (Focal Loss + CutMix)
python train.py --set exp_id=T03 backbone=convnext_tiny loss=focal mix=cutmix seed=0
```

### Bước 3: Đánh giá Phương pháp Suy luận & Độ trễ

Sử dụng mô hình T03 đã huấn luyện để kiểm tra TTA, Temperature Scaling và đo độ trễ:

```bash
# Đo độ trễ và thông lượng suy luận bằng benchmark.py
python -c "import model, benchmark; m = model.build_model('convnext_tiny'); print(benchmark.latency_report(m, batch_size=1, img_size=224))"
```

### Bước 4: Vòng chung kết (Chạy 3 seeds & Đánh giá trên tập TEST)

Huấn luyện lại cấu hình tốt nhất với 3 seeds khác nhau và xuất file dự đoán trên tập TEST:

```bash
python train.py --set exp_id=F01 backbone=convnext_tiny loss=focal mix=cutmix seed=42 save_test_predictions=True
python train.py --set exp_id=F02 backbone=convnext_tiny loss=focal mix=cutmix seed=43 save_test_predictions=True
python train.py --set exp_id=F03 backbone=convnext_tiny loss=focal mix=cutmix seed=44 save_test_predictions=True

# Đồng thời chạy mô hình Baseline (T00) trên 3 seeds để đối chiếu mức cải thiện
python train.py --set exp_id=T00_01 backbone=resnet50 seed=42 save_test_predictions=True
python train.py --set exp_id=T00_02 backbone=resnet50 seed=43 save_test_predictions=True
python train.py --set exp_id=T00_03 backbone=resnet50 seed=44 save_test_predictions=True
```

## 5. Đánh giá & Chấm điểm tự động

Chạy script `eval.py` để tính chỉ số và tự kiểm tra điểm số mục I theo Rubric:

```bash
# 1. Tính chỉ số chi tiết cho cấu hình chung kết (mean ± std qua 3 seeds)
python eval.py score --pred "predictions/F01_seed*_test.csv" \
    --test-csv data/labels/test_subset0.csv \
    --labels data/labels/labels.csv \
    --tag F01 --out eval_results

# 2. Chấm điểm đề xuất phần I của RUBRIC (So sánh Chung kết vs Baseline)
python eval.py grade \
    --final "predictions/F01_seed*_test.csv" \
    --baseline "predictions/T00_0*_test.csv" \
    --test-csv data/labels/test_subset0.csv \
    --labels data/labels/labels.csv
```
