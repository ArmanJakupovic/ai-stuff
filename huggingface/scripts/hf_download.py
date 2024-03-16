import argparse
from huggingface_hub import snapshot_download

def download_model(model_id, max_workers):
    # Split the model_id on '/' and take the second part as the model_name
    model_name = model_id.split('/')[1]
    snapshot_download(repo_id=model_id, local_dir=model_name,
                      local_dir_use_symlinks=False, revision="main", max_workers=max_workers)

if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description='Download a model from the Hugging Face Hub')

    # Add an argument for the model ID
    parser.add_argument('-m', '--model_id', help='The HuggingFace Hub model ID to download. Format should be <user>/<model_name>.', required=True)

    # Add an argument for max_workers with '-w' as the option
    parser.add_argument('-w', '--max_workers', type=int, default=2, help='The maximum number of workers to use for downloading. Default is 2.')

    # Parse the command line arguments
    args = parser.parse_args()

    # Use the model_id and max_workers arguments
    download_model(args.model_id, args.max_workers)
