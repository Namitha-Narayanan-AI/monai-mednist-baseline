from pathlib import Path

from monai.apps import MedNISTDataset


def main():
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"

    data_dir.mkdir(parents=True, exist_ok=True)

    print("=== MedNIST Dataset Loader ===")
    print(f"Project root: {project_root}")
    print(f"Data folder : {data_dir}")

    dataset = MedNISTDataset(
        root_dir=str(data_dir),
        section="training",
        download=True,
        seed=0,
    )

    print("\nDataset loaded successfully.")
    print(f"Training samples: {len(dataset)}")

    sample = dataset[0]

    print("\nFirst sample:")
    print(f"Type: {type(sample)}")
    print(f"Keys: {sample.keys()}")
    print(f"Image path: {sample['image']}")
    print(f"Label: {sample['label']}")


if __name__ == "__main__":
    main()