import os
from itertools import product

import rasterio
from rasterio import windows


def get_tiles(ds, width=512, height=512):
    nols, nrows = ds.meta["width"], ds.meta["height"]
    offsets = product(range(0, nols, width), range(0, nrows, height))
    big_window = windows.Window(0, 0, nols, nrows)
    for col_off, row_off in offsets:
        window = windows.Window(col_off, row_off, width, height).intersection(
            big_window
        )
        transform = windows.transform(window, ds.transform)
        yield window, transform


def tile_image_and_mask(
    img_path,
    mask_path,
    out_img_dir,
    out_mask_dir,
    tile_size=(512, 512),
):
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)
    with rasterio.open(img_path) as img_ds, rasterio.open(mask_path) as mask_ds:
        meta_img = img_ds.meta.copy()
        meta_mask = mask_ds.meta.copy()

        for win, transform in get_tiles(img_ds, *tile_size):
            img_tile = img_ds.read(window=win)
            mask_tile = mask_ds.read(window=win)

            for meta in (meta_img, meta_mask):
                meta.update(
                    {
                        "height": win.height,
                        "width": win.width,
                        "transform": transform,
                    }
                )

            fname = f"tile_x{int(win.col_off)}_y{int(win.row_off)}.tif"

            with rasterio.open(
                os.path.join(out_img_dir, fname), "w", **meta_img
            ) as dst:
                dst.write(img_tile)
            with rasterio.open(
                os.path.join(out_mask_dir, fname), "w", **meta_mask
            ) as dst:
                dst.write(mask_tile)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Tile an image and its mask into smaller tiles."
    )
    parser.add_argument("img_path", help="Path to the input image file.")
    parser.add_argument("mask_path", help="Path to the input mask file.")
    parser.add_argument("out_img_dir", help="Directory to save the output image tiles.")
    parser.add_argument("out_mask_dir", help="Directory to save the output mask tiles.")
    parser.add_argument(
        "--tile_size",
        type=int,
        nargs=2,
        default=(512, 512),
        help="Size of the tiles (width height). Default is 512x512.",
    )
    args = parser.parse_args()

    tile_image_and_mask(
        args.img_path,
        args.mask_path,
        args.out_img_dir,
        args.out_mask_dir,
        tile_size=args.tile_size,
    )


if __name__ == "__main__":
    main()
