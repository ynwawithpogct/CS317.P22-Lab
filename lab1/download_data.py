import os
import shutil
import time
from huggingface_hub import snapshot_download


def download_animals10(data_path = "data", temp_path = "temp"):
    if os.path.exists(data_path) and os.path.isdir(data_path) and not os.listdir(data_path):
        return 
    os.makedirs(data_path, exist_ok=True)
    os.makedirs(temp_path, exist_ok=False)
    local_dir = snapshot_download(
        repo_id="dgrnd4/animals-10", 
        repo_type="dataset", 
        local_dir=temp_path,
        token=os.environ["HUGGINGFACE_HUB_TOKEN"]
    )

    print(f"Dữ liệu đã tải về: {local_dir}")

    start_time = time.time()
    shutil.unpack_archive(f"{local_dir}/dataset.zip", data_path)

    print(f"Dữ liệu giải nén thành công trong {time.time() - start_time} giây")

    if os.path.exists(temp_path):
        shutil.rmtree(temp_path)
        print(f"Đã xóa toàn bộ thư mục: {temp_path}")
    else:
        print("Thư mục không tồn tại.")