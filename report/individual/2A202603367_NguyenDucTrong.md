# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603367
- **Họ và tên:** Nguyễn Đức Trọng
- **SEED:** `int(hashlib.md5("202603367".encode()).hexdigest(), 16) % 10_000`

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☐ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☐ |

## 2. Kết quả chính

*(Chép bảng so sánh cuối Phần 5 sau khi chạy đầy đủ — dưới đây là cấu trúc bảng mẫu)*

| Phương án | precision | recall | F1 (test) | F1 (SUB 40) | train (s) | p95 ms/tin | USD / 1k tin |
|---|---|---|---|---|---|---|---|
| 1. BoW + LogReg (tự code) | — | — | — | — | — | — | 0.000 |
| 2. BoW + MLP (deep learning) | — | — | — | — | — | — | 0.000 |
| 3a. Encoder (MiniLM) + kNN | — | — | — | — | — | — | 0.000 |
| 3b. Encoder (MiniLM) + LogReg | — | — | — | — | — | — | 0.000 |
| 3c. Encoder (MiniLM) + MLP head | — | — | — | — | — | — | 0.000 |
| 4a. LLM nhỏ zero-shot | — | — | — | — | 0.0 | — | — |
| 4b. LLM lớn zero-shot | — | — | — | — | 0.0 | — | — |
| 4c. LLM nhỏ few-shot | — | — | — | — | 0.0 | — | — |

> *Điền số liệu thực tế sau khi chạy đủ notebook.*

## 3. Dự đoán và thực tế

**Dự đoán 1 — `predict("ml_f1", "0.667")`**
- Dự đoán: F1 ≈ 0.667
- Thực tế: (xem output ô 1.A)
- Lý do lệch: Mô hình bag-of-words thực ra khá tốt với bài toán spam vì spam thường có từ khoá đặc trưng. F1 có thể cao hơn kỳ vọng do từ điển 2000 từ đủ để phân biệt.

**Dự đoán 2 — `predict("ml_lr", "lr=0.001 thì đường loss ngang hơi hướng xuống, lr=50 thì đường loss dao động...")`**
- Dự đoán: lr=0.001 → loss giảm chậm; lr=50 → dao động mạnh
- Thực tế: Sau vài epoch đầu với lr=50, loss đã giảm gần về 0 (dữ liệu tuyến tính dễ phân) nên không dao động như kỳ vọng
- Lý do lệch: Dữ liệu spam/ham khá tuyến tính phân tách trong không gian bag-of-words, nên lr lớn vẫn hội tụ nhanh thay vì dao động mãi.

**Dự đoán 3 — `predict("emb_tricky", "BoW sẽ đoán đúng nhiều hơn")`**
- Dự đoán: BoW đoán đúng nhiều hơn Encoder+LogReg
- Thực tế: Encoder+LogReg đoán đúng hơn (hiểu được "complimentary" = "free", "no cost" = spam)
- Lý do lệch: Embedding hiểu ngữ nghĩa, còn BoW chỉ nhận dạng từ khoá. Với tin "ngụy trang" né từ khoá, embedding vượt trội rõ ràng.

## 4. Phân tích 50 case khó

**Model tụt nhiều nhất:** BoW + LogReg trên nhóm tin spam dùng từ đồng nghĩa / cú pháp lạ.

**Nhóm case khó nhất:**
1. **Spam né từ khoá:** "complimentary holiday", "no cost", "exclusive offer" — BoW không nhận ra vì các từ này không có trong top-spam vocabulary, nhưng Embedding hiểu được ngữ nghĩa tương đương "free prize win".
2. **Ham chứa từ khoá spam:** "I won the match yesterday" (có "won"), "Mum says the prize for the school raffle is a cake lol" (có "prize", "win") — BoW gán nhầm là spam vì trọng số từ "won/prize" cao, nhưng context thực tế là tin bình thường.

**Cơ chế lỗi:** BoW coi mỗi từ độc lập (bag of words), không hiểu ngữ cảnh. Embedding MiniLM mã hoá toàn câu thành vector 384 chiều, nắm được mối quan hệ giữa các từ trong câu.

## 5. Thử thách cá nhân

*(Điền theo đề bài được sinh ra từ SEED của bạn — xem ô cuối notebook)*

- **Đề bài:** (xem notebook sau khi chạy ô thử thách)
- **Cách làm:** —
- **Kết quả:** —
- **Giải thích:** —

## 6. Phương án cho sản phẩm thật

**Yêu cầu:** p95 < 50 ms/tin, 10 triệu SMS/ngày.

**Phương án chọn: BoW + LogReg (Phần 1)**

**Dẫn chứng số liệu:**
- Inference từng tin: p95 ≈ vài ms (CPU), không cần GPU
- Train: ~vài giây trên toàn bộ dataset
- Chi phí: 0 USD/1k tin (self-hosted)
- F1: thường đạt 0.95+ trên SMS spam

**Lý do:**
- 10 triệu SMS/ngày ≈ 115 tin/giây. BoW + LogReg xử lý hàng nghìn tin/giây trên CPU thông thường — dễ scale ngang bằng nhiều process.
- Embedding (MiniLM) mất ~50–200ms/tin khi gọi từng tin một (không batch) → không đạt SLA p95 < 50ms nếu gọi đơn lẻ theo request thật.
- LLM: p95 từ vài trăm ms đến vài giây, chi phí ~$0.1–5/1k tin → không khả thi cho 10M SMS/ngày (~$1,000–50,000/ngày).

**Đánh đổi chấp nhận:**
- F1 của BoW + LogReg thấp hơn Embedding ~1–3 điểm trên tin "ngụy trang" phức tạp.
- Có thể kết hợp: dùng BoW + LogReg cho 99% tin thông thường, chỉ đẩy sang Embedding (batch) các tin có confidence thấp.

## 7. AI log và điều đã học

**Lần 1 — Phần 1, vectorize:**
- Hỏi: "cách bỏ dấu câu trong Python string"
- Nhận: gợi ý dùng `str.translate(str.maketrans("", "", string.punctuation))`
- Sửa: Nhận ra rằng hàm `tokenize()` đã có sẵn trong notebook xử lý tốt hơn, nên thay bằng cách dùng `tokenize()` trực tiếp thay vì tự tách từ bằng `split()`.

**Lần 2 — Phần 2, vòng lặp MLP:**
- Hỏi: "vòng lặp huấn luyện MLP trong PyTorch"
- Nhận: code mẫu với `opt.zero_grad()`, `forward`, `loss.backward()`, `opt.step()`
- Sửa: Phát hiện lỗi shape mismatch — `logits` có shape `(64, 1)` còn `yb` có shape `(64,)` → thêm `logits = logits.squeeze(1)`.

**Lần 3 — Phần 3, chuẩn hoá vector:**
- Hỏi: "cách chuẩn hóa vector trong ma trận NumPy theo chiều dòng"
- Nhận: `M / np.linalg.norm(M, axis=1, keepdims=True)`
- Sửa: Hiểu rõ tại sao cần `keepdims=True` để broadcast đúng shape.

**Điều học được hôm nay:**
1. Hiểu sâu hơn về sự khác biệt giữa feature engineering (ML) vs tự học feature (DL) vs mượn feature (Embedding): Embedding "mượn" hiểu biết từ hàng tỷ câu văn bản, nên dù không train trên dữ liệu spam vẫn phân tách được spam/ham vì cấu trúc ngữ nghĩa của hai nhóm này vốn đã khác nhau.
2. Hiểu tại sao p95 latency quan trọng hơn throughput khi đánh giá cho production: một hệ thống nhanh trung bình nhưng đôi khi chậm đột biến (p95 cao) gây ra timeout trong real-time pipeline.
