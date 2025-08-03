from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def scd2binary(semanti_map):
    return (semanti_map != 0).astype(semanti_map.dtype)


def main():
    data_root = Path("data/SECOND")
    modes = ["train", "test"]

    for mode in modes:
        save_dir = data_root / mode / "label_binary"
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"Processing {mode} data...")
        sem_path = data_root / mode / "label1"

        sem_files = list(sem_path.glob("*.png"))

        for sem_file in tqdm(sem_files):
            semanti_map = np.array(Image.open(sem_file), dtype=np.uint8)
            binary_map = scd2binary(semanti_map)
            binary_map = Image.fromarray(binary_map).convert("P")
            binary_map.putpalette([0, 0, 0, 255, 255, 255] + [0] * (256 - 6))
            binary_map.save(save_dir / sem_file.name)


if __name__ == "__main__":
    main()
