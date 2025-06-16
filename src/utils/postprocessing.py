import json
from pathlib import Path

import numpy as np


def combine_maps(
    parcellation_map: np.typing.NDArray[np.uint16],
    hemisphere_map: np.typing.NDArray[np.uint8],
    shift: tuple[int, int, int],
) -> np.typing.NDArray[np.uint16]:

    jhu_atlas_dict: dict[tuple[int, int], np.uint16] = {
        (int(key.split(",")[0]), int(key.split(",")[1])): np.uint16(value)
        for key, value in json.loads((Path(__file__).parent / "split_map.json").read_text()).items()
    }

    def lookup_jhu_atlas(key: np.typing.NDArray) -> np.uint16:
        return jhu_atlas_dict.get((int(key[0]), int(key[1])), np.uint16(0))

    combined_map = np.stack([hemisphere_map, parcellation_map], axis=-1)

    jhu_atlas_map = np.apply_along_axis(lookup_jhu_atlas, axis=-1, arr=combined_map)
    jhu_atlas_map = np.pad(jhu_atlas_map, [(32, 32), (16, 16), (32, 32)], "constant", constant_values=0)
    jhu_atlas_map = np.roll(jhu_atlas_map, (-shift[0], -shift[1], -shift[2]), axis=(0, 1, 2))

    return jhu_atlas_map
