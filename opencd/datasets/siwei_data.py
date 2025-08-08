# Copyright (c) Open-CD. All rights reserved.
from opencd.registry import DATASETS

from .basescddataset import BaseSCDDataset


@DATASETS.register_module()
class Siwei_Dataset(BaseSCDDataset):
    METAINFO = dict(
        classes=("unchanged", "changed"),
        palette=[[0, 0, 0], [255, 255, 255]],
        semantic_classes=(
            "unchanged",
            "耕地",
            "林地",
            "草地",
            "大棚",
            "硬化地表",
            "特殊用地",
            "运动场",
            "光伏",
            "风电",
            "高压电塔",
            "通信塔",
            "露天设备",
            "其他构筑物",
            "推堆土",
            "采矿用地",
            "建筑施工地表",
            "高层房屋建筑",
            "低矮房屋建筑",
            "铁路",
            "公路",
            "农村道路",
            "河流水面",
            "湖泊水面",
            "水库水面",
            "坑塘水面",
            "沟渠",
            "水工建筑用地",
            "裸土地",
            "裸岩石砾地",
        ),
        semantic_palette=[
            (200, 200, 200),  # unchanged
            (120, 180, 50),  # 耕地
            (35, 120, 45),  # 林地
            (150, 200, 90),  # 草地
            (180, 220, 180),  # 大棚
            (160, 160, 160),  # 硬化地表
            (255, 165, 0),  # 特殊用地
            (255, 100, 100),  # 运动场
            (60, 60, 100),  # 光伏
            (220, 220, 255),  # 风电
            (128, 128, 255),  # 高压电塔
            (90, 90, 190),  # 通信塔
            (210, 210, 100),  # 露天设备
            (180, 140, 100),  # 其他构筑物
            (210, 180, 140),  # 推堆土
            (150, 110, 80),  # 采矿用地
            (255, 140, 90),  # 建筑施工地表
            (200, 0, 0),  # 高层房屋建筑
            (255, 80, 80),  # 低矮房屋建筑
            (105, 105, 105),  # 铁路
            (50, 50, 50),  # 公路
            (100, 100, 100),  # 农村道路
            (0, 150, 255),  # 河流水面
            (0, 100, 200),  # 湖泊水面
            (0, 120, 220),  # 水库水面
            (50, 150, 200),  # 坑塘水面
            (80, 170, 220),  # 沟渠
            (0, 70, 180),  # 水工建筑用地
            (210, 180, 140),  # 裸土地
            (170, 160, 150),  # 裸岩石砾地
        ],
    )

    def __init__(
        self,
        img_suffix=".tif",
        seg_map_suffix=".tif",
        reduce_semantic_zero_label=True,
        **kwargs,
    ) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_semantic_zero_label=reduce_semantic_zero_label,
            **kwargs,
        )
