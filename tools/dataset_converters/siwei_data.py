from pathlib import Path

import numpy as np
from PIL import Image
from skimage import io
from tqdm import tqdm


def scd2binary(semanti_map):
    return (semanti_map > 0).astype(semanti_map.dtype)


def main1():
    data_root = Path("data/crop512")

    save_dir = data_root / "label_binary"
    save_dir.mkdir(parents=True, exist_ok=True)
    print("Processing data...")
    sem_path = data_root / "masks_t0"

    sem_files = list(sem_path.glob("*.tif"))

    for sem_file in tqdm(sem_files):
        # semanti_map = np.array(Image.open(sem_file), dtype=np.uint8)
        semantic_map = io.imread(sem_file.as_posix())
        binary_map = scd2binary(semantic_map)
        binary_map = Image.fromarray(binary_map).convert("P")
        binary_map.putpalette([0, 0, 0, 255, 255, 255] + [0] * (256 - 6))
        binary_map.save(save_dir.joinpath(sem_file.stem + ".tif"))


def main():
    data_root = Path("data/crop512")
    save_dir = data_root / "label_binary"

    for f in save_dir.glob("*.tif"):
        img = np.array(Image.open(f))
        if img.any():
            print(f"{f} has non-zero values.")


if __name__ == "__main__":
    main()
