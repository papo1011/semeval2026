import shutil
import zipfile
import argparse
from pathlib import Path

from utils.logger import setup_global_logger

SCRIPT_DIR = Path(__file__).resolve().parent

# Define the tasks and their target data directories
TASKS = {
    "A": SCRIPT_DIR / "task_A" / "data",
    "B": SCRIPT_DIR / "task_B" / "data",
    "C": SCRIPT_DIR / "task_C" / "data",
}


def download_from_huggingface(task_key, output_dir, logger):
    """Downloads, formats to Parquet, and renames test split."""
    repo_id = "DaniilOr/SemEval-2026-Task13"
    logger.info(f"Downloading data for Task {task_key} from HuggingFace ({repo_id})...")

    try:
        from datasets import load_dataset
        # Load the specific subset for the task.
        # This returns a DatasetDict containing all available splits.

        dataset = load_dataset(repo_id, task_key)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Dynamically iterate through every split available
        for split_name, split_data in dataset.items():
            if split_name == "test":
                logger.warning(
                    f"⚠️ 'test' split found for Task {task_key}. Renaming to 'test_sample' to avoid confusion."
                )
                split_name = "test_sample"

            split_path = output_dir / f"{split_name}.parquet"
            split_data.to_pandas().to_parquet(split_path, index=False)

            logger.info(
                f"✅ Saved '{split_name}' split ({len(split_data)} rows) to {split_path}"
            )

    except Exception as e:
        logger.error(f"❌ Failed to process Task {task_key} from HuggingFace: {e}")


def download_from_kaggle(task_key, output_dir, logger):
    """Direct ZIP download and extraction using the Kaggle API."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        # This will automatically look for ~/.kaggle/kaggle.json
        # or KAGGLE_USERNAME / KAGGLE_KEY environment variables!
        api.authenticate()
    except ImportError:
        logger.error("❌ 'kaggle' library not found. Please install kaggle")
        raise
    except Exception as e:
        logger.error(f"❌ Kaggle Authentication failed: {e}")
        raise

    competition_name = f"sem-eval-2026-task-13-subtask-{task_key.lower()}"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting Kaggle download for {competition_name}...")

    try:
        # The Kaggle API bundles competition files into <competition_name>.zip
        api.competition_download_files(
            competition_name, path=str(output_dir), quiet=False
        )
        logger.info("Download completed. Checking for ZIP archive...")

        # Automatic Extraction & Cleanup
        zip_path = output_dir / f"{competition_name}.zip"
        if zip_path.exists():
            logger.info("Extracting files and flattening files...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                # Iterate through every item in the zip
                for member in zip_ref.infolist():
                    # Skip directories, we only want files
                    if member.is_dir():
                        continue

                    # Get just the filename (strip away all internal zip folders)
                    file_name = Path(member.filename).name
                    target_file_path = output_dir / file_name

                    # Extract the file directly into our output_dir
                    with (
                        zip_ref.open(member, "r") as source,
                        open(target_file_path, "wb") as target,
                    ):
                        shutil.copyfileobj(source, target)

            logger.info(f"✅ Files extracted successfully to: {output_dir}")

            zip_path.unlink()
            logger.info("Cleaned up temporary ZIP file.")
        else:
            logger.info(f"✅ Downloaded files directly to {output_dir}.")

    except Exception as e:
        logger.error(f"❌ Error during Kaggle download: {e}")
        logger.info("Did you accept the competition rules on the Kaggle website?")
        raise e


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Universal Dataset Downloader for SemEval Task 13"
    )

    parser.add_argument(
        "--task",
        type=str,
        choices=["A", "B", "C", "ALL"],
        default="ALL",
        help="Which task dataset to download (A, B, C, or ALL)",
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=["kaggle", "huggingface"],
        default="kaggle",
        help="Where to download the data from",
    )

    args = parser.parse_args()

    # Initialize the global logger for the Data Downloader
    logger = setup_global_logger("./logs", prefix="Data_Downloader")

    # Determine which tasks to process
    tasks_to_process = TASKS.keys() if args.task == "ALL" else [args.task]

    # Execute the requested pipeline
    for task_key in tasks_to_process:
        target_dir = TASKS[task_key]
        logger.info("-" * 40)

        if args.source == "huggingface":
            download_from_huggingface(task_key, target_dir, logger)
        elif args.source == "kaggle":
            download_from_kaggle(task_key, target_dir, logger)

    logger.info("=" * 50)
    logger.info("All requested datasets have been processed!")
