import yaml
from pathlib import Path


def create_data_yaml(dataset_path="pcb_dataset"):
    dataset_path = Path(dataset_path)

    # Verify folder structure
    required_folders = ["train/images", "train/labels",
                        "val/images", "val/labels"]
    for folder in required_folders:
        if not (dataset_path / folder).exists():
            (dataset_path / folder).mkdir(parents=True, exist_ok=True)
            print(f"Created: {dataset_path / folder}")

    # Create data.yaml content
    data = {
        "train": str(dataset_path / "train/images"),
        "val": str(dataset_path / "val/images"),
        "nc": 6,
        "names": [
            "short_circuit",
            "open_circuit",
            "missing_component",
            "solder_bridge",
            "spurious_copper",
            "wrong_drill"
        ]
    }

    # Save YAML file
    yaml_path = dataset_path / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, sort_keys=False)

    print(f"\n✅ Created data.yaml at: {yaml_path}")
    print("Please verify your class names match your actual labels!")


if __name__ == "__main__":
    create_data_yaml()