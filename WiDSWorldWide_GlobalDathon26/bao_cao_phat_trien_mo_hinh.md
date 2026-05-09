# Dự Đoán Cháy Rừng — Báo Cáo Phát Triển Mô Hình

**Nhiệm vụ:** Dự đoán xác suất một đám cháy rừng sẽ ảnh hưởng đến một mục tiêu hạ tầng trong 4 mốc thời gian: 12h, 24h, 48h và 72h.  
**Chỉ số đánh giá:** Hybrid score = 0.3 × C-index + 0.7 × (1 − Brier score có trọng số), trong đó Brier có trọng số = 0.3×B(24h) + 0.4×B(48h) + 0.3×B(72h).  
**Kết quả tốt nhất:** Private score **0.96823** | Public score **0.96269** (`gbsa_lightgbm_ver7_5`)

---

## Tổng Quan Quá Trình Phát Triển

Giải pháp phát triển qua 8 phiên bản, từ một mô hình baseline đơn giản tiến đến một ensemble kết hợp hai họ mô hình bổ trợ cho nhau — Gradient Boosting Survival Analysis (GBSA) và LightGBM — được blend theo từng mốc thời gian với phương pháp huấn luyện hiệu chỉnh IPCW. Mỗi phiên bản giải quyết một vấn đề cụ thể được phát hiện từ phiên bản trước.

---

## Phiên Bản 1 — Mô Hình Survival Cơ Bản

### Những gì đã thử
Một mô hình `GradientBoostingSurvivalAnalysis` (GBSA) duy nhất được huấn luyện trên bộ dữ liệu chia 80/20 đơn giản, không có cross-validation. Hàm survival được đọc tại mỗi mốc thời gian trong 4 horizon để tạo ra xác suất. Feature phân loại `low_temporal_resolution_0_5h` được one-hot encode; không có feature engineering nào khác được áp dụng.

### Tại sao không hiệu quả
- **Không có cross-validation:** Mô hình được đánh giá trên một tập holdout cố định 20%, khiến ước tính khả năng tổng quát hóa phụ thuộc rất nhiều vào việc những hàng nào rơi vào tập test. Không có tín hiệu đáng tin cậy để tinh chỉnh.
- **Không có feature engineering:** Dùng các feature thô trực tiếp, bỏ lỡ những thông tin đặc thù của bài toán như tốc độ phát triển đám cháy so với khoảng cách, thời gian dự kiến đến nơi, hay phân loại vùng nguy hiểm.
- **Một seed, một config:** Không có tính đa dạng ensemble, khiến dự đoán của mô hình bị nhiễu.
- **Không xử lý dữ liệu bị kiểm duyệt (censored):** Các quan sát bị kiểm duyệt (đám cháy không ảnh hưởng trong cửa sổ quan sát) bị ngầm định xử lý như sự kiện âm tính, làm mô hình nghiêng về dự đoán xác suất thấp hơn thực tế.

---

## Phiên Bản 2 — Feature Engineering + OOF Đa Seed

### Những gì đã thử
Giới thiệu **Stratified K-Fold cross-validation** đúng chuẩn với 5 seed và **dự đoán out-of-fold (OOF)**, thay thế holdout đơn. Một tập feature đặc thù phong phú được xây dựng:

- **Biến đổi khoảng cách:** log-distance, nghịch đảo khoảng cách, căn bậc hai khoảng cách, percentile xếp hạng khoảng cách
- **Tỉ lệ diện tích/khoảng cách:** bán kính đám cháy, tỉ lệ bán kính/khoảng cách, log tỉ lệ diện tích/khoảng cách
- **Feature động học:** thời gian dự kiến đến nơi (`eta_hours`), tốc độ tiếp cận hiệu quả kết hợp chuyển động đám cháy và tốc độ lan rộng hướng tâm
- **Điểm đe dọa:** alignment × speed / log(distance)
- **Cờ vùng nguy hiểm:** chỉ báo nhị phân cho vùng nguy kịch (<5 km), cảnh báo (5–10 km) và an toàn
- **Feature thời gian:** is_summer, is_afternoon
- Loại bỏ các cột nhiễu/dư thừa: `relative_growth_0_5h`, `projected_advance_m`, `centroid_displacement_m`, v.v.

Target encoding (an toàn theo fold, chỉ fit trên fold huấn luyện) cũng được thêm cho các cột phân loại thời gian.

### Tại sao vẫn chưa đủ tốt
- Mô hình GBSA giờ được đánh giá tốt qua OOF, nhưng **GBSA một mình không phân biệt tốt ở các horizon ngắn** (12h, 24h) vì hàm survival không được huấn luyện để phân biệt nhị phân tại các ngưỡng cụ thể.
- Một **meta-learner / mô hình thứ hai** được lên kế hoạch (`lgbm`, `xbsa`, `xgat`) nhưng chưa được triển khai — chỉ hoàn thành tầng GBSA.
- Tính toán Brier score được thực hiện hậu kỳ trên toàn bộ tập OOF mà không có trọng số IPCW đúng, nghĩa là điểm số bị lệch bởi censoring.

---

## Phiên Bản 3 — Ensemble GBSA + LightGBM Riêng Biệt Cho Mỗi Horizon

### Những gì đã thử
- Mở rộng quy mô GBSA lên **10 fold × 10 seed** với `n_estimators=2000`, `learning_rate=0.005`.
- Giới thiệu **LightGBM classifier riêng biệt cho từng horizon** (12h, 24h, 48h, 72h), mỗi cái được huấn luyện trên nhãn nhị phân `event=1 AND time ≤ horizon`. Điều này cho phép mỗi horizon được coi là bài toán phân loại nhị phân độc lập, trao cho LightGBM sự linh hoạt để học các ranh giới quyết định đặc thù theo horizon mà hàm survival chung của GBSA không thể biểu diễn được.
- Dùng `sksurv.metrics.brier_score` để tính Brier score theo kiểu survival-aware đúng chuẩn cho đánh giá GBSA.
- Áp dụng `np.maximum.accumulate` để đảm bảo ràng buộc đơn điệu (P(12h) ≤ P(24h) ≤ P(48h) ≤ P(72h)).

### Tại sao vẫn chưa đủ tốt
- **IPCW được tính trên toàn bộ tập huấn luyện** trước vòng lặp OOF, gây rò rỉ dữ liệu: mô hình kiểm duyệt G(t) được ước tính trên dữ liệu validation fold, khiến Brier score ở cấp fold trông tốt hơn thực tế.
- LightGBM vẫn bao gồm tất cả mẫu để huấn luyện, kể cả **các quan sát bị kiểm duyệt** (đám cháy kết thúc trước horizon mà không ảnh hưởng). Đây là những ẩn số thực sự — chúng không phải dương tính cũng không phải âm tính — nhưng bị xử lý như âm tính, khiến mô hình hệ thống dự đoán xác suất thấp hơn thực tế.
- LGBM chỉ chạy với **một config cho mỗi horizon**, hạn chế tính đa dạng ensemble.

---

## Phiên Bản 4 — IPCW Cho LightGBM + Chain Classifier + Trọng Số Blend Cố Định

### Những gì đã thử
Ba bổ sung lớn:

**1. Trọng số IPCW cho LightGBM.** Inverse Probability of Censoring Weighting được áp dụng làm `sample_weight` trong quá trình huấn luyện LightGBM. Kaplan-Meier estimator được fit trên thời gian kiểm duyệt của toàn bộ tập huấn luyện (tức là `event_observed = 1 - event`), và mỗi quan sát sự kiện không bị kiểm duyệt được tăng trọng bởi `1 / G(t_i)`. Điều này hiệu chỉnh phân phối nhãn nhị phân để xấp xỉ tốt hơn xác suất sự kiện thực.

**2. Kiến trúc chain classifier.** Bốn dự đoán horizon được xâu chuỗi: xác suất đầu ra từ mô hình 12h được thêm làm feature khi huấn luyện mô hình 24h, đầu ra 24h nuôi mô hình 48h, v.v. Động lực là P(ảnh hưởng trong 24h) chứa thông tin về P(ảnh hưởng trong 48h), nên truyền dự đoán về phía trước có thể cải thiện calibration.

**3. Trọng số blend cố định giữa GBSA và LightGBM theo horizon:**
```
12h: 97% GBSA + 3% LightGBM
24h: 95% GBSA + 5% LightGBM
48h: 45% GBSA + 55% LightGBM
```
Horizon 72h chỉ dùng GBSA (đặt cứng thành 1.0 khi nộp bài). Trọng số được chọn bằng cách xem xét Brier score OOF và nhận thấy LightGBM đóng góp nhiều hơn ở 48h trong khi GBSA chiếm ưu thế ở các horizon ngắn hơn.

### Tại sao vẫn chưa đủ tốt
- **IPCW vẫn được tính trên toàn bộ dataset trước vòng lặp CV**, không phải theo fold. KM estimator cho G(t) đã thấy thời gian kiểm duyệt của validation fold, tạo ra vấn đề rò rỉ dữ liệu giống phiên bản 3.
- **Chain classifier gây ô nhiễm OOF.** Khi các feature huấn luyện của mô hình 24h bao gồm dự đoán OOF 12h, những giá trị OOF đó được tạo ra bằng cách trung bình toàn bộ fold, nhưng khi suy luận trên tập test chúng đến từ mô hình 12h được huấn luyện trên toàn bộ dữ liệu huấn luyện. Sự không nhất quán giữa dự đoán OOF (holdout) và dự đoán test (full-data) khiến chain hoạt động khác nhau lúc đánh giá so với lúc nộp bài, tạo ra một sự dịch chuyển phân phối ẩn.
- Số seed LGBM bị giới hạn ở 5 ([526, 484, 749, 852, 848]), hạn chế khả năng giảm phương sai của ensemble.

---

## Phiên Bản 5 — IPCW Sửa Theo Fold + Early Stopping + Loại Bỏ Feature Phân Loại

### Những gì đã thử
Phiên bản này tập trung sửa các vấn đề rò rỉ dữ liệu được xác định trong phiên bản 4. Từ changelog của notebook:

> 1. Rò rỉ dữ liệu Brier score → Tính Brier score bên trong vòng lặp OOF cho cả GBSA và LGBM  
> 2. IPCW fit trên toàn bộ dữ liệu huấn luyện → Tính IPCW trên từng split CV

**Các thay đổi chính:**
- **IPCW giờ được tính theo từng fold:** Bên trong vòng lặp CV, `KaplanMeierFitter` chỉ được fit trên `train_times[train_idx]` và `train_events[train_idx]`, nên phân phối kiểm duyệt G(t) được ước tính mà không nhìn thấy validation fold. Đây là cách tiếp cận đúng, không bị rò rỉ.
- **Early stopping được thêm vào LightGBM** dùng `early_stopping(stopping_rounds=50)` với validation fold làm `eval_set`, ngăn overfitting trên fold huấn luyện.
- **Các cột phân loại bị loại khỏi features** (`event_start_hour`, `event_start_dayofweek`, `event_start_month`) thay vì target encode, đơn giản hóa pipeline và giảm rủi ro rò rỉ target từ encoding.
- Số estimator LGBM tăng lên 1000 trên tất cả các horizon.
- Brier score cũng được theo dõi theo từng fold để chẩn đoán chất lượng mô hình tốt hơn.

### Tại sao vẫn chưa phải phiên bản cuối
- **Chain classifier vẫn còn**, mang theo vấn đề ô nhiễm OOF.
- GBSA vẫn chạy **10 fold × 10 seed** với một config duy nhất. Tính đa dạng ensemble của GBSA bị giới hạn ở việc thay đổi seed.
- Early stopping dùng **validation fold làm eval set bên trong vòng lặp fold**, điều này đúng để dừng nhưng có nghĩa là số cây thay đổi theo fold — mô hình đầy đủ được huấn luyện lúc nộp bài sẽ dùng số estimator cố định, gây ra sự không khớp.

---

## Phiên Bản 6 — Ensemble Nhiều Config GBSA + Submission Chỉ Dùng GBSA

### Những gì đã thử
Phiên bản này tái cơ cấu đáng kể ensemble GBSA:

- **Nhiều config GBSA thay vì nhiều seed của cùng một config.** 10 tổ hợp siêu tham số khác nhau được định nghĩa, thay đổi `learning_rate`, `subsample`, `max_depth`, `min_samples_leaf`, `min_samples_split` và `n_estimators`. Với mỗi config chạy 15 seed, tổng cộng 10 × 15 = 150 cặp (config, seed) được trung bình hóa.
- Số fold giảm từ 10 xuống **5** để ensemble lớn hơn vẫn khả thi về mặt tính toán.
- **Các hàm metric tùy chỉnh** (`compute_brier_sc`, `compute_hybrid_sc`, `compute_c_index`) được triển khai, thay thế `sksurv.metrics.brier_score` chậm hơn và C-index O(n²). Một stub `compute_ipcw()` được để trống, báo hiệu công việc tương lai được lên kế hoạch.
- **Submission chỉ dùng GBSA** (không có blend LightGBM), với dự đoán 72h được đặt cứng thành 1.0. Đây có thể là một bước lùi để có baseline sạch cho ensemble GBSA mới trước khi thêm lại LightGBM.
- XGBoost được import nhưng chưa được sử dụng.

### Tại sao vẫn chưa đủ tốt
- **Không có blend LightGBM** — horizon 48h đặc biệt được chứng minh trong các phiên bản trước là hưởng lợi đáng kể từ LightGBM.
- **Vấn đề chain classifier không được giải quyết** trong code LightGBM hiện có, nhưng phiên bản này đã loại bỏ LightGBM hoàn toàn khỏi submission.
- Stub `compute_ipcw` cho thấy IPCW chưa được tích hợp lại vào cấu trúc code mới.

---

## Phiên Bản 7 — LightGBM Được Tích Hợp Lại với IPCW + Xử Lý Hàng Bị Kiểm Duyệt

### Những gì đã thử
Phiên bản 7 tích hợp lại LightGBM vào ensemble với những cải tiến kiến trúc quan trọng:

**1. Loại bỏ hàng bị kiểm duyệt khỏi CV fold của LGBM.** Đây là sửa chữa cấu trúc quan trọng nhất: các quan sát bị kiểm duyệt trước horizon (tức là `event == 0 AND time < horizon`) thực sự là không có nhãn — chúng ta không thể biết liệu chúng có trở thành sự kiện không. Các hàng này bị loại khỏi các split StratifiedKFold. Chỉ có các hàng `uncensored_idx` tham gia vòng lặp CV. Sau vòng lặp fold, các hàng bị kiểm duyệt được điền bằng trung bình dự đoán từ tất cả các mô hình fold.

**2. Triển khai IPCW tùy chỉnh.** Một hàm `compute_ipcw(times, events, horizon)` dựa trên Kaplan-Meier được triển khai trực tiếp (không phụ thuộc thư viện ngoài), tính toán:
- Trọng số = 1/G(t_i) cho các sự kiện được quan sát trước horizon
- Trọng số = 1/G(horizon) cho các quan sát sống sót qua horizon
- Trọng số = 1.0 cho mọi trường hợp còn lại

**3. Nhiều config LGBM cho mỗi horizon.** Thay vì một config cho mỗi horizon, nhiều config được định nghĩa cho từng horizon 12h, 24h và 48h, thay đổi độ sâu, learning rate, regularization và leaf count dựa trên mật độ sự kiện đặc thù của horizon.

**4. Chain classifier bị loại bỏ.** Mỗi horizon được huấn luyện độc lập mà không có feature chaining, loại bỏ vấn đề ô nhiễm OOF được xác định trong phiên bản 4–5.

**5. Power calibration cho 48h:** Sau khi tạo dự đoán OOF, `oof_preds_lgbm[:, 2] **= 1.1` được áp dụng cho cột 48h để co các dự đoán quá tự tin một chút về phía 0.

**Lưu ý:** Phiên bản này được chạy ở chế độ debug rút gọn (`gbsa_configs[:2]`, `GBSA_SEEDS[:2]`) để kiểm tra kiến trúc nhanh chóng, nên hiệu suất submission thực tế yếu.

### Các lỗi còn lại được sửa trong phiên bản 7.5
- Một số lỗi còn tồn tại trong vòng lặp LGBM: `lgbm_full` được tham chiếu trước khi được huấn luyện trong vòng lặp seed, `ipcw_weights` bị index kép, và `horizon_oof` được tích lũy bên trong vòng lặp fold (nhiều hơn ×N_FOLDS lần). Các lỗi này được sửa trong 7.5.

---

## Phiên Bản 7.5 — Mô Hình Cuối Cùng (Submission Tốt Nhất)

### Những gì đã thử
Phiên bản 7.5 chạy cấu hình production đầy đủ với tất cả lỗi đã được sửa:

**Ensemble GBSA:**
- 10 config × 15 seed × 5 fold = 750 mô hình tổng cộng được trung bình hóa
- Không có IPCW cho GBSA (mô hình survival xử lý censoring tự nhiên qua log-rank loss)
- Đơn điệu được đảm bảo bằng `np.maximum.accumulate`

**Ensemble LightGBM:**
- 15 seed × 5 fold cho mỗi horizon, riêng biệt cho 12h, 24h, 48h
- IPCW tính theo fold chỉ trên các hàng trong fold huấn luyện (không rò rỉ)
- Các hàng bị kiểm duyệt bị loại khỏi CV và được điền sau vòng lặp bằng mô hình full-data (`lgbm_full`)
- 5 config cho mỗi horizon (được trung bình hóa), được chọn phản ánh mật độ sự kiện đặc thù theo horizon:
  - 12h: rất thận trọng (regularization cao, cây nông, `max_depth=2–3`, `num_leaves=3–6`) vì rất ít sự kiện đủ điều kiện
  - 24h: regularization cân bằng
  - 48h: regularization thấp hơn, cho phép nhiều capacity hơn
- Power calibration: dự đoán 48h được nâng lên lũy thừa 1.1 để giảm over-prediction

**Blending:**
Trọng số blend được grid-search trên dự đoán OOF để tìm tỉ lệ GBSA/LightGBM tối ưu theo từng horizon:
```
12h: W_GBSA = 0.97, W_LGB = 0.03
24h: W_GBSA = tối ưu hóa (grid search)
48h: W_GBSA = tối ưu hóa (grid search)
72h: Chỉ GBSA (LightGBM bị loại khỏi 72h hoàn toàn)
```
Dự đoán GBSA 72h được dùng nguyên vẹn (không bị đặt cứng thành 1.0 trong OOF; lệnh ghi đè 1.0 chỉ xuất hiện trong submission test ở các phiên bản có lỗi trước đó).

Tính đơn điệu được đảm bảo lại trên các dự đoán đã blend.

**Kết quả:** Private 0.96823 / Public 0.96269

---

## Tóm Tắt Đóng Góp Của Từng Phiên Bản

| Phiên bản | Bổ sung chính | Vấn đề được giải quyết |
|-----------|--------------|------------------------|
| v1 | Baseline GBSA, split đơn giản | — |
| v2 | Feature engineering, OOF CV đa seed | Loại bỏ thiên lệch holdout |
| v3 | LightGBM riêng biệt theo horizon, đơn điệu | Phân biệt tốt hơn theo horizon |
| v4 | IPCW cho LightGBM, chain classifier, trọng số blend | Hiệu chỉnh censoring |
| v5 | IPCW tính theo fold, early stopping | Rò rỉ dữ liệu trong IPCW |
| v6 | Ensemble nhiều config GBSA (10 config × 15 seed) | Đa dạng ensemble |
| v7 | Loại bỏ hàng censored, xóa chain classifier, IPCW tùy chỉnh | Ô nhiễm OOF |
| v7.5 | Chạy production đầy đủ + trọng số blend grid-search trên OOF | Sửa lỗi, mở rộng quy mô |

---

## Bài Học Kinh Nghiệm

**Dữ liệu bị kiểm duyệt không phải dữ liệu âm tính.** Xử lý các quan sát bị kiểm duyệt như nhãn 0 (âm tính) làm lệch hệ thống bất kỳ bộ phân loại nhị phân nào. Cách tiếp cận đúng là loại bỏ chúng khỏi huấn luyện (như đã làm với LGBM) hoặc dùng loss survival phù hợp (như GBSA làm tự nhiên).

**IPCW phải được tính theo từng fold.** Tính hàm trọng số kiểm duyệt G(t) trên toàn bộ dataset trước CV khiến phân phối kiểm duyệt của validation fold ảnh hưởng đến trọng số mẫu của mô hình, làm phồng hiệu suất OOF biểu kiến. Chỉ tính trên fold huấn luyện mới là cách tiếp cận đúng, không bị rò rỉ.

**Chain classifier làm ô nhiễm dự đoán OOF.** Đưa dự đoán OOF của một horizon làm feature cho mô hình horizon tiếp theo nghe có vẻ trực quan nhưng tạo ra sự không khớp phân phối lúc test: dự đoán OOF đến từ các fold holdout, trong khi dự đoán test-time đến từ mô hình được huấn luyện trên toàn bộ dữ liệu. Sự không khớp này làm giảm khả năng tổng quát hóa dù có vẻ cải thiện điểm OOF.

**GBSA và LightGBM bổ sung cho nhau.** GBSA xử lý tự nhiên cấu trúc survival và xuất sắc ở 24h và 72h. LightGBM có thể được tinh chỉnh sắc nét theo từng horizon với class balancing tùy chỉnh và đóng góp nhiều nhất ở 48h. Blend theo cấp horizon thay vì toàn cục nắm bắt được sự bất đối xứng này.

**Điểm public cao hơn không có nghĩa là mô hình tốt hơn.** Phiên bản 7 (chạy debug) và phiên bản 8 (trọng số grid-search trên OOF) đều có điểm public cao hơn v7.5 nhưng điểm private tệ hơn, cho thấy chúng đã overfit vào phân phối test public thông qua bước blend được tinh chỉnh quá mức hoặc việc trung bình ensemble không đủ.
