import sys
from pathlib import Path

from osgeo import gdal


def get_file_paths(directory: str, extension: str = ".tif") -> list[Path]:
    """Get all file paths with the specified extension in the given directory.

    Args:
        directory (str): The directory to search for files.
        extension (str): The file extension to look for (e.g., '.txt').

    Returns:
        list[Path]: A list of Path objects representing the file paths.
    """
    dir_path = Path(directory)
    return list(dir_path.rglob(f"*{extension}"))


if __name__ == "__main__":
    p = sys.argv[1]

    tif_list = get_file_paths(p, ".tif")

    vrt_path = "merged.vrt"
    # 可选参数：选择插值方式、是否加入 alpha 通道等
    options = gdal.BuildVRTOptions(resampleAlg="cubic", addAlpha=False)
    # 创建 VRT（构建虚拟图像）
    vrt = gdal.BuildVRT(vrt_path, tif_list, options=options)
    vrt = None

    vrt_ds = gdal.Open(vrt_path, gdal.GA_ReadOnly)
    gdal.Translate(
        "mask_cog_t1.tif",
        vrt_ds,
        format="COG",
        creationOptions=[
            "BLOCKSIZE=512",
            "COMPRESS=DEFLATE",
            "PREDICTOR=2",
        ],
        outputType=gdal.GDT_Byte,
    )
    vrt_ds = None
