# CS317.P22-Lab

## Thông tin

Họ Tên: Nguyễn Quế Phong
MSSV: 21520398
Lớp: KHNT2021
Môn: Phát triển và vận hành hệ thống máy học - CS317.P22

## Lab 1

Đề Bài: Lab 1 - Xây dựng training pipeline có sử dụng experiment tracking
- Yêu cầu:
    - Xây dựng được training pipeline với đầy đủ các step (data preprocessing, training, validation, evaluation)
    - Sử dụng experiment tracking framework (ví dụ MLFlow) để log lại tất cả các tham số trong quá trình training bao gồm:  Hyperparameters
        - Training dataset (dataset link, source, dataset version...)
        - Metrics
        - Checkpoints
- Tiêu chí phụ (cộng điểm nếu có, tối đa +3 điểm):
    - Sử dụng framework/công nghệ mới
    - Có hyperparameter tuning
    - Có task orchestration/distributed training (ví dụ có dùng rayio, Clear ML...)
    - Điểm sáng tạo (sáng tạo về pipeline, về cách tuning, cách dùng framework...)

Code:
- Cấu trúc thư mục:
        <pre lang="markdown"> 
            # Directory structure
            ├── lab1
            │   ├── data/
            │   │   └── raw/
            │   ├── model/
            │   ├── src/
            │   │   └── preprocess.py
            │   │   └── train.py
            │   │   └── evaluate.py
            │   ├── mlruns/ (MLflow logs)
            │   ├── requirements.txt
            │   └── Dockerfile
        </pre>