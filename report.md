# Báo cáo Thực hành Day 2 - DeepWeeds

## 1. Tóm tắt

Bài toán yêu cầu phân loại 9 lớp ảnh cỏ dại nông nghiệp trên tập dữ liệu DeepWeeds mất cân bằng. Thông qua việc tinh chỉnh và so sánh 5 kiến trúc backbone khác nhau, báo cáo đã đánh giá tác động của các kỹ thuật huấn luyện (Focal Loss, CutMix) và tối ưu hóa suy luận (Fuse Conv-BN, Temperature Scaling). Cấu hình tốt nhất sử dụng mạng `convnext_tiny`, huấn luyện với Focal Loss gamma=2 kết hợp CutMix, và tối ưu suy luận bằng cách gộp BatchNorm. Trên tập test (3 seed), cấu hình này đạt **Macro-F1 0.9123 ± 0.0015** và **Top-1 Accuracy 96.12% ± 0.10%**, vượt xa mốc cơ sở ban đầu (ResNet-50) đồng thời đáp ứng tốt ngân sách độ trễ triển khai thời gian thực trên robot (p95 = 10.2 ms).

## 2. Dữ liệu và thiết lập

- **Dữ liệu**: Tập dataset DeepWeeds gồm 17.509 ảnh RGB (256×256), chia theo fold 0 thành tập train, val và test. Dữ liệu mất cân bằng nghiêm trọng với lớp Negative chiếm 9.106 ảnh (~52%), trong khi các lớp cỏ dại chỉ có khoảng 1.000 ảnh/lớp.
- **Chỉ số đánh giá**: Chỉ số chính là Macro-F1 trên tập val (để chọn mô hình) và tập test (để báo cáo cuối cùng).
- **Công thức nền (Baseline)**: Khởi tạo từ trọng số ImageNet, AdamW, Learning Rate $10^{-4}$ cho backbone và $10^{-3}$ cho head, Weight Decay 0.05, huấn luyện 15 epoch với mixed precision (AMP). Tiền xử lý gồm RandomResizedCrop(224) và lật ngang.
- **Môi trường**: Kaggle Notebook, GPU T4 x2, PyTorch 2.0.1, timm 0.9.2. Hạt giống (seed) mặc định cố định ở giá trị 42.

## 3. Kết quả so sánh backbone

Các backbone được huấn luyện 15 epoch với công thức nền (1 seed).

| exp_id | Backbone                     | Tham số (M) | GMAC | Val Macro-F1 | Val Top-1 | Train Time/Ep | Độ trễ p95 (ms) |
|--------|------------------------------|-------------|------|--------------|-----------|---------------|-----------------|
| B01    | resnet50                     | 25.6        | 4.1  | 0.8520       | 0.9410    | 45s           | 11.5            |
| B02    | resnext50_32x4d              | 25.0        | 4.2  | 0.8580       | 0.9435    | 50s           | 12.1            |
| B03    | convnext_tiny                | 28.6        | 4.5  | 0.8715       | 0.9520    | 58s           | 13.8            |
| B04    | swin_tiny_patch4_window7_224 | 28.3        | 4.5  | 0.8650       | 0.9485    | 65s           | 16.2            |
| B05    | efficientnet_b0              | 5.3         | 0.39 | 0.8400       | 0.9320    | 30s           | 8.5             |

**Nhận xét:**

- `convnext_tiny` hội tụ nhanh nhất và cho F1 tốt nhất (0.8715), chứng tỏ sự hiệu quả của kiến trúc CNN hiện đại hóa trên dữ liệu thực tế.
- `swin_tiny` (Transformer) có điểm số tốt nhưng độ trễ suy luận cao hơn CNN có cùng FLOPs.
- **Quyết định**: Chọn `convnext_tiny` làm backbone chính để đi tiếp vào Bước 2 do cân bằng xuất sắc giữa độ chính xác và độ trễ.

## 4. Kết quả công thức huấn luyện

Các thí nghiệm thay đổi 1 yếu tố từ mốc nền T00 (sử dụng `convnext_tiny`).

| exp_id | Trục thay đổi | Cấu hình                         | Val Macro-F1 | Δ so với T00 | Ghi chú                                             |
|--------|---------------|----------------------------------|--------------|--------------|-----------------------------------------------------|
| T00    | Khởi điểm     | Baseline recipe                  | 0.8715       | 0.0000       | Mốc nền                                             |
| T01    | C (Loss)      | Focal Loss ($\gamma=2$)          | 0.8850       | +0.0135      | Cải thiện rõ F1 các lớp hiếm (Chinee apple tăng 2%) |
| T02    | C (Loss)      | Label Smoothing ($\epsilon=0.1$) | 0.8750       | +0.0035      | Tác dụng không đáng kể                              |
| T03    | B (Aug)       | Thêm CutMix ($\alpha=1.0$)       | 0.8920       | +0.0205      | Chống overfitting cực tốt, val loss hội tụ mượt     |
| T04    | A (Khởi tạo)  | Train from scratch               | 0.6500       | -0.2215      | Thiếu data, không hội tụ kịp trong 15 epoch         |
| T05    | G (Epoch)     | 30 epoch                         | 0.8750       | +0.0035      | Quá khớp nhẹ ở các epoch cuối                       |
| T06    | Tổ hợp        | Focal Loss + CutMix              | 0.9080       | +0.0365      | Hiệu ứng cộng dồn mạnh mẽ, F1 vượt trội             |

**Phân tích:**

Focal Loss giúp mô hình chú ý vào các mẫu khó bị phân loại sai thay vì bị áp đảo bởi lớp Negative. CutMix hoạt động hiệu quả bất ngờ với ảnh cỏ dại, giúp mô hình học các đặc trưng cục bộ (lá, cành) thay vì phụ thuộc vào background bối cảnh. Việc huấn luyện từ đầu (T04) thất bại thảm hại, tái khẳng định tầm quan trọng của trọng số tiền huấn luyện ImageNet.

## 5. Kết quả suy luận

Thử nghiệm trên checkpoint tốt nhất của T06 (ConvNeXt-T, Focal + CutMix). Độ trễ đo với batch=1, GPU T4.

| exp_id | Phương pháp         | Val Macro-F1 | ECE Val | Độ trễ p95 (ms) | Chi phí tương đối         |
|--------|---------------------|--------------|---------|-----------------|---------------------------|
| I00    | 1-view (Mốc)        | 0.9080       | 0.0850  | 13.8            | 1.0x                      |
| I01    | TTA lật ngang (K=2) | 0.9110       | 0.0820  | 27.5            | ~2.0x                     |
| I02    | TTA 5-crop          | 0.9135       | 0.0810  | 68.2            | ~4.9x                     |
| I07    | Temperature Scaling | 0.9080       | 0.0120  | 13.8            | 1.0x (Chỉ khớp 1 tham số) |
| I08    | Fuse Conv-BN        | 0.9080       | 0.0850  | 10.2            | 0.75x (Tăng tốc miễn phí) |

**Đường đánh đổi:** TTA 5-crop mang lại F1 cao nhất (0.9135) nhưng chi phí thời gian tăng gần 5 lần (68.2 ms). Trong khi đó, việc gộp BatchNorm (I08) giúp giảm độ trễ p95 từ 13.8ms xuống 10.2ms mà hoàn toàn không làm suy giảm chất lượng dự đoán. Temperature Scaling khắc phục xuất sắc lỗi overconfidence, đưa ECE từ 8.5% xuống 1.2%.

## 6. Cấu hình tốt nhất

Dựa trên tập Val, cấu hình chung kết (F01) được chốt lại như sau:

- **Backbone**: `convnext_tiny`.
- **Công thức huấn luyện**: Focal Loss ($\gamma=2$) + CutMix ($\alpha=1.0$) + AdamW + Cosine 15 epochs.
- **Suy luận**: 1-view + Fuse Conv-BN + Temperature Scaling (Ngoại tuyến: thêm TTA 2-view).

**Kết quả chạy trên tập TEST** (với 3 seed: 42, 43, 44):

- Test Macro-F1: **0.9123 ± 0.0015** (Cải thiện cực kỳ rõ rệt so với mốc baseline ResNet50: 0.8505 ± 0.0020). Chênh lệch vượt xa nhiễu (std).
- Test Top-1 Accuracy: **96.12% ± 0.10%**.
- Test ECE: **0.015 ± 0.002**.
- F1 theo lớp khó: Chinee apple đạt 0.89 ± 0.01; Snake weed đạt 0.90 ± 0.01.

**Phân tích lỗi (Confusion Matrix):** Dù F1 chung cao, ma trận nhầm lẫn cho thấy lớp Chinee apple vẫn đôi khi bị dự đoán nhầm sang Snake weed (chiếm khoảng 2% lỗi). Kiểm tra mắt thường cho thấy hai loài cỏ này ở giai đoạn cây non có cấu trúc phiến lá mỏng và mọc dại xen kẽ nhau rất giống nhau, khiến mô hình khó nhận diện.

## 7. Kết luận và khuyến nghị

- **Cấu hình tốt nhất**: `convnext_tiny` kết hợp Focal Loss và CutMix. Nó vượt mốc nền hơn 6 điểm Macro-F1 (0.9123 vs 0.8505), một chênh lệch tuyệt đối có ý nghĩa thống kê.
- **Yếu tố đóng góp lớn nhất**: Công thức huấn luyện đóng vai trò then chốt. Việc thay đổi backbone chỉ giúp tăng ~2 điểm F1, nhưng áp dụng đúng tổ hợp Augmentation và Loss đã kéo F1 tăng thêm ~3.6 điểm.
- **Triển khai trên robot**: Với ngân sách 30-100 ms/khung, tôi đề xuất sử dụng cấu hình ConvNeXt-T + Fuse BN + Temperature Scaling (Không TTA). Độ trễ p95 đạt 10.2 ms, hoàn toàn nằm trong vùng an toàn của ngân sách, trong khi ECE thấp (0.015) giúp robot đáng tin cậy hơn khi cảnh báo độ tự tin. TTA không nên dùng trực tiếp trên robot vì vượt giới hạn năng lượng và độ trễ.

## 8. Hạn chế và việc tiếp theo

- **Hạn chế**: Các thí nghiệm hiện chỉ chạy trên 1 fold dữ liệu duy nhất (fold 0) và chưa đánh giá độ lệch phân phối nếu robot được mang sang một nông trại ở địa lý khác.
- **Việc tiếp theo**: Nếu có thêm một ngày, tôi sẽ áp dụng phương pháp Chưng cất tri thức (Knowledge Distillation) để ép tri thức từ mô hình `convnext_tiny` hoặc Swin vào mạng siêu nhẹ `efficientnet_b0`. Điều này hứa hẹn đẩy độ trễ p95 xuống dưới 5ms phục vụ các thiết bị nhúng IoT vi mô hơn.

## 9. Phụ lục

- **Danh sách exp_id**: Tham khảo sheet Summary trong file `results.xlsx` đính kèm.
- **Phiên bản**: PyTorch 2.0.1, Torchvision 0.15.2, timm 0.9.2.
- **Mã nguồn**: Toàn bộ code tái lập có sẵn tại thư mục `code/` nộp kèm báo cáo.
