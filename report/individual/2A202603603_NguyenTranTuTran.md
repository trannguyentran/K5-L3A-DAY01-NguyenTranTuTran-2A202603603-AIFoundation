# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603603
- **Họ và tên:** Nguyen Tran Tu Tran
- **SEED:** 1107

## 1. Bằng chứng

| Tệp | Có? |
| --- | :---: |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☑ |

## 2. Kết quả chính

| Phương pháp | Precision | Recall | F1 (test) | F1 (SUB 40) | Train (s) | Tin/s (cả lô) | p50 (ms) | p95 (ms) | USD / 1k tin |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1. BoW + LogReg (tự code)** | 1.000 | 0.787 | 0.881 | 0.919 | 6.014 | 12501.49 | 0.054 | 0.072 | $0.000 |
| **2. BoW + MLP (deep learning)** | 1.000 | 0.888 | 0.940 | 0.947 | 10.685 | 11242.51 | 0.306 | 3.327 | $0.000 |
| **3a. Encoder (MiniLM) + kNN** | 0.806 | 0.888 | 0.845 | 0.923 | 83.427 | 35.65 | 130.562 | 205.117 | $0.000 |
| **3b. Encoder (MiniLM) + LogReg** | 0.988 | 0.899 | 0.941 | 0.947 | 87.009 | 34.40 | 63.524 | 141.568 | $0.000 |
| **3c. Encoder (MiniLM) + MLP head** | 0.963 | 0.876 | 0.918 | 0.919 | 86.507 | 26.77 | 30.314 | 39.156 | $0.000 |
| **4a. LLM nhỏ zero-shot** | 0.905 | 0.950 | 0.927 | 0.927 | 0.000 | 1.41 | 693.522 | 930.835 | $0.023 |
| **4c. LLM nhỏ few-shot** | 1.000 | 0.950 | 0.974 | 0.974 | 0.000 | 1.42 | 710.139 | 979.882 | $0.051 |

## 3. Dự đoán và thực tế

3 dự đoán lệch nhiều nhất so với kết quả:

1. **`tradeoff` (Đường Pareto và SLA 50ms):**
   - *Dự đoán:* 3b và 3c sẽ nằm trên đường Pareto vì có sự cân bằng tốt giữa F1 và độ trễ; LLM và kNN sẽ vượt vạch SLA 50ms.
   - *Thực tế:* 3b và 3c đều **không** nằm trên đường Pareto. Đường Pareto gồm `1. BoW + LogReg`, `2. BoW + MLP` và `4c. LLM few-shot`.
   - *Vì sao lệch:* Mô hình `2. BoW + MLP` có tốc độ xử lý vượt trội ($p95 = 3.33\text{ ms}$) và đạt F1 SUB bằng 3b ($0.947$). Do đó, `BoW + MLP` đã lấn át hoàn toàn 3b ($p95 = 141.57\text{ ms}$) và 3c ($p95 = 39.16\text{ ms}$).

2. **`llm_models` (Chất lượng LLM vs Embedding 3b):**
   - *Dự đoán:* LLM khó thắng được mô hình Embedding 3b vì không được huấn luyện trực tiếp trên tập dữ liệu này.
   - *Thực tế:* Với `4c. LLM few-shot`, F1 đạt tới **0.974**, vượt qua cả `3b. Encoder + LogReg` ($0.947$) và `2. BoW + MLP` ($0.947$).
   - *Vì sao lệch:* LLM có khả năng suy luận ngữ cảnh và học theo ngữ cảnh (in-context learning) rất mạnh; chỉ cần 4 ví dụ mẫu chuẩn xác là đủ để LLM nắm bắt được ranh giới phân loại mà không cần train lại trọng số.

3. **`hard_eval` (Độ suy giảm trên 50 case khó):**
   - *Dự đoán:* Accuracy của các mô hình sẽ giảm mạnh xuống khoảng 60–70%.
   - *Thực tế:* Các mô hình BoW sụp đổ sâu hơn dự kiến (`1. BoW + LogReg` F1 tụt xuống **0.286**, Accuracy chỉ còn **50%**; `2. BoW + MLP` F1 còn **0.421**, Accuracy **56%**). Ngược lại, LLM lại chống chịu cực tốt (Accuracy đạt **98%** ở zero-shot và **100%** khi có prompt phòng thủ).
   - *Vì sao lệch:* BoW hoàn toàn bất lực trước nhóm spam giọng tự nhiên (đạt 0% accuracy) và né từ khóa do gặp hiện tượng từ ngoài từ điển (OOV), trong khi LLM hiểu trọn vẹn ngữ nghĩa hội thoại thực tế.

## 4. Phân tích 50 case khó

- **Model tụt nhiều nhất:** `1. BoW + LogReg` tụt nhiều nhất (F1 từ **0.919** xuống **0.286**, giảm 68.9%), kế tiếp là `2. BoW + MLP` (F1 từ **0.947** xuống **0.421**).
- **Nhóm case gây tụt nặng nhất:** 
  - *Spam giọng tự nhiên:* BoW đạt **0%** accuracy, MLP chỉ đạt **10%**.
  - *Né từ khoá:* BoW đạt **17%** accuracy, MLP đạt **0%**.
- **2 case cụ thể và cơ chế gây sai:**
  1. **Case #11 (Spam giọng tự nhiên):** *"Hey it's Jess from the gym 😊 we're doing a members-only thing this week, 3 months for the price..."*
     - *Nhãn thật:* `spam`.
     - *Cơ chế sai của BoW:* Tin nhắn dùng giọng điệu thân thiện, lịch sự như bạn bè, không chứa các từ khóa spam cổ điển (`free`, `win`, `prize`, `call now`) trong từ điển 2.000 từ. Vector đặc trưng toàn số 0, khiến mô hình gán nhãn sai thành `ham`.
  2. **Case #29 (Né từ khoá):** *"C0ngr4tulat10ns! Y0u h4ve w0n a £1000 g1ft c4rd. Cl4im n0w at w1n-gift.c0m"*
     - *Nhãn thật:* `spam`.
     - *Cơ chế sai của BoW:* Kẻ lừa đảo cố tình thay thế chữ cái bằng chữ số (Leetspeak: `0` thay `o`, `4` thay `a`, `1` thay `i`). Bộ tokenizer tách từ của BoW coi `C0ngr4tulat10ns`, `w0n`, `g1ft` là các từ hoàn toàn mới lạ (Out-Of-Vocabulary) nên trọng số không kích hoạt được bộ phân loại.

## 5. Thử thách cá nhân

- **Đề bài nhận được (Đề #4 - SEED 1107):** *Embedding · Chạy `knn_predict` với $k = 1, 5, 25$. Báo F1. Vì sao $k$ quá nhỏ hoặc quá lớn đều có hại?*
- **Cách làm:** Gọi hàm `knn_predict(E_tr, y_tr, E_te[SUB], k=k_val)` lần lượt với $k \in \{1, 5, 25\}$, sau đó tính `metrics` trên tập `y_te[SUB]`.
- **Kết quả:**
  - $k = 1$: Accuracy = 0.900, Precision = 0.900, Recall = 0.900, **F1 = 0.900**
  - $k = 5$: Accuracy = 0.925, Precision = 0.947, Recall = 0.900, **F1 = 0.923**
  - $k = 25$: Accuracy = 0.900, Precision = 0.944, Recall = 0.850, **F1 = 0.895**
- **Giải thích:**
  - *Khi $k=1$ quá nhỏ:* Mô hình bị Overfitting (High Variance), cực kỳ nhạy cảm với các điểm nhiễu và ngoại lai gần nhất trong tập train, chỉ cần 1 điểm lân cận bị gán nhãn sai là kéo theo dự đoán sai.
  - *Khi $k=5$:* Cân bằng tối ưu giữa việc giảm nhiễu cục bộ nhờ biểu quyết đa số mà vẫn giữ được tính địa phương của không gian embedding.
  - *Khi $k=25$ quá lớn:* Mô hình bị Underfitting (High Bias), vùng lân cận bị mở rộng ra xa làm mất tính tương đồng ngữ nghĩa, bị ảnh hưởng nặng bởi sự mất cân bằng giữa các lớp dẫn đến Recall giảm mạnh từ 0.90 xuống 0.85.

## 6. Phương án cho sản phẩm thật

- **Yêu cầu:** Lọc 10 triệu SMS/ngày (~116 tin/giây trung bình), yêu cầu $p95 < 50\text{ ms/tin}$.
- **Phương án lựa chọn:** Chọn **`2. BoW + MLP (deep learning)`**.
- **Dẫn chứng số liệu:**
  - *Độ trễ & Thông lượng:* $p95 = 3.327\text{ ms} \ll 50\text{ ms}$, thông lượng xử lý đạt **11.242 tin/giây** (vượt xa yêu cầu 116 tin/giây của nhà mạng).
  - *Chất lượng:* F1 test toàn cục đạt **0.940** (trên tập SUB đạt **0.947**), đặc biệt Precision đạt tuyệt đối **1.000** (không lo chặn nhầm SMS quan trọng/OTP của khách hàng).
  - *Chi phí:* **$0 USD** phí API bên ngoài, triển khai on-premise siêu nhẹ.
- **Sự đánh đổi:** Chấp nhận khả năng chống chịu kém hơn trước các kỹ thuật spam né từ khóa tinh vi (so với LLM/Embedding), nhưng có thể bù đắp bằng tầng tiền xử lý chuẩn hóa ký tự regex.
- **Ước tính nếu dùng LLM:** Với chi phí **$0.05092 / 1k tin** (LLM few-shot), mỗi tháng hệ thống ngốn:
  $$10.000.000 \times \frac{0.05092}{1.000} \times 30 \approx 15.276\text{ USD/tháng}$$
  (khoảng 380 triệu VNĐ/tháng), kèm theo độ trễ $p95 \approx 980\text{ ms}$ (vi phạm nghiêm trọng cam kết SLA).

## 7. AI log và điều đã học

| Phần | Bạn hỏi AI gì (prompt) | AI trả lời/viết gì | Bạn đã sửa / kiểm chứng gì |
|---|---|---|---|
| **Phần 4 & Bắt lỗi 3** | Hàm `parse_label_v2` dùng `"spam" in response.lower()` sai ở đâu? Vì sao model suy luận cần max_tokens lớn? | Giải thích lỗi substring khi gặp `"not spam"` và hiện tượng reasoning token nuốt trọn output limit. | Viết lại hàm regex bắt ranh giới từ `\b(spam\|ham)\b` và chạy test giả lập `_t()`. |
| **Phần 5 & Câu hỏi 5** | Cách đọc đường Pareto và xác định model bị lấn át? | Giải thích cặp chỉ số $(1-F1, p95)$ và định nghĩa lấn át Pareto. | Đối chiếu trực tiếp với kết quả chạy cell 83 và chứng minh `BoW + MLP` lấn át `Encoder + LogReg`. |
| **Phần 5 & Stress test** | Vì sao BoW tụt mạnh ở spam giọng tự nhiên nhưng LLM lại không bị ảnh hưởng? | Phân tích cơ chế túi từ tĩnh đối chiếu với khả năng hiểu ngữ cảnh văn bản của LLM. | Kiểm tra trực tiếp 25 case sai của BoW trong danh sách in ra ở cell 90. |

**Điều hiểu ra hôm nay:**
1. Một mô hình có độ phức tạp cao hơn, hiện đại hơn (như Transformer Embedding hay LLM) chưa chắc đã là lựa chọn tối ưu trong môi trường thực tế khi phải cân bằng giữa F1, độ trễ SLA và chi phí vận hành.
2. Đánh giá mô hình trên tập test cùng phân phối rất dễ tạo cảm giác lạc quan giả tạo; chỉ khi kiểm thử qua các case khó (adversarial/stress testing) thì các điểm mù cốt lõi của từng kiến trúc mới bộc lộ rõ ràng.
