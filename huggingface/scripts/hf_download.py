import sys
from huggingface_hub import snapshot_download

def download_model(model_id):
    # Split the model_id on '/' and take the second part as the model_name
    model_name = model_id.split('/')[1]
    snapshot_download(repo_id=model_id, local_dir=model_name,
                      local_dir_use_symlinks=False, revision="main", max_workers=2)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py model_id")
        sys.exit(1)

    model_id = sys.argv[1]
    download_model(model_id)
