# NLP_HUIT_v2

## Giới thiệu
NLP_HUIT_v2 là dự án thực nghiệm cho đề tài phát hiện tin giả tiếng Việt có khả năng chống chịu trước các tấn công thao túng sắc thái cảm xúc. Dự án tập trung vào việc đánh giá ảnh hưởng của sentiment manipulation đối với mô hình phát hiện tin giả, đồng thời xây dựng mô hình robust hơn bằng chiến lược huấn luyện trên dữ liệu đã được trung hòa cảm xúc.

## Mục tiêu
Dự án được xây dựng nhằm:
- Huấn luyện và đánh giá các mô hình baseline cho bài toán phát hiện tin giả tiếng Việt.
- Tạo dữ liệu đối kháng bằng cách viết lại văn bản theo các sắc thái cảm xúc khác nhau.
- Kiểm tra mức độ suy giảm hiệu năng của mô hình khi đối mặt với adversarial sentiment attacks.
- Thử nghiệm hướng tiếp cận robust hơn thông qua huấn luyện trên dữ liệu trung lập hóa.

## Nội dung chính của dự án
Dự án bao gồm các thành phần:
- Tiền xử lý và làm sạch dữ liệu VFND.
- Chia dữ liệu train / validation / test.
- Huấn luyện baseline truyền thống và baseline PhoBERT.
- Sinh dữ liệu adversarial, consistency và neutralized.
- Đánh giá trên nhiều tập kiểm thử khác nhau.
- Tổng hợp kết quả qua các bảng CSV phục vụ phân tích và báo cáo.

## Cấu trúc thư mục
- `Duy.ipynb`: notebook chính của dự án, chứa toàn bộ quá trình thực nghiệm.
- `app.py`: file ứng dụng hoặc mã chạy hỗ trợ cho dự án.
- `vfnd_experiment/data/`: chứa dữ liệu đã xử lý và các tập train/test/validation.
- `vfnd_experiment/output/`: chứa kết quả thực nghiệm, bảng tổng hợp và các file đầu ra phục vụ đánh giá.
- `.gitignore`: loại bỏ các file không cần đưa lên GitHub.

## Dữ liệu sử dụng
Dự án sử dụng bộ dữ liệu **VFND (Vietnamese Fake News Dataset)** đã được làm sạch và chia tập để phục vụ thực nghiệm. Ngoài dữ liệu gốc, dự án còn tạo thêm các tập:
- `test_adversarial.csv`
- `test_consistency.csv`
- `test_rewritten.csv`
- `train_neutralized.csv`

Các tập này được dùng để đánh giá ảnh hưởng của sentiment manipulation và khả năng ổn định của mô hình.

## Kết quả đầu ra
Thư mục `vfnd_experiment/output/` chứa nhiều file kết quả như:
- `adversarial_results.csv`
- `consistency_results.csv`
- `final_results.csv`
- `final_summary_multirun.csv`
- `final_table_comparison.csv`
- `final_table_multirun.csv`
- `llm_judge_results.csv`
- `multirun_summary.csv`
- `table3_results.csv`

Các file này phục vụ cho việc:
- so sánh baseline và mô hình robust,
- đánh giá consistency,
- thống kê nhiều lần chạy,
- tổng hợp bảng kết quả cuối cùng cho báo cáo.

## Ghi chú về mô hình
Do dung lượng lớn, file trọng số:
`vfnd_experiment/output/phobert_baseline.pt`
không được đưa trực tiếp lên GitHub.

## Ý nghĩa của dự án
NLP_HUIT_v2 không chỉ là một bài toán phân loại tin giả thông thường, mà còn mở rộng sang hướng đánh giá độ bền vững của mô hình trước các thao tác thay đổi sắc thái cảm xúc. Dự án góp phần minh họa rằng một mô hình có thể đạt kết quả tốt trên dữ liệu gốc nhưng vẫn dễ bị ảnh hưởng khi văn bản bị chỉnh sửa cảm xúc theo cách có chủ đích.

## Mục đích sử dụng
Repo này được dùng để:
- lưu trữ mã nguồn và notebook thực nghiệm,
- phục vụ nộp đồ án môn Xử lý ngôn ngữ tự nhiên,
- hỗ trợ tái hiện kết quả và kiểm tra lại quy trình thực nghiệm.
