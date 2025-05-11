import json
from pathlib import Path

import numpy as np


def combine_maps(
    parcellation_map: np.typing.NDArray[np.uint16],
    hemisphere_map: np.typing.NDArray[np.uint8],
    shift: tuple[int, int, int],
) -> np.typing.NDArray[np.uint16]:

    jhu_atras_dict: dict[tuple[int, int], np.uint16] = {
        (int(key.split(",")[0]), int(key.split(",")[1])): np.uint16(value)
        for key, value in json.loads((Path(__file__).parent / "split_map.json").read_text()).items()
    }

    def lookup_jhu_atlas(key: np.typing.NDArray) -> np.uint16:
        return jhu_atras_dict.get((int(key[0]), int(key[1])), np.uint16(0))

    combined_map = np.stack([hemisphere_map, parcellation_map], axis=-1)

    jhu_atras_map = np.apply_along_axis(lookup_jhu_atlas, axis=-1, arr=combined_map)
    jhu_atras_map = np.pad(jhu_atras_map, [(32, 32), (16, 16), (32, 32)], "constant", constant_values=0)
    jhu_atras_map = np.roll(jhu_atras_map, (-shift[0], -shift[1], -shift[2]), axis=(0, 1, 2))

    return jhu_atras_map


# def _postprocessing(parcellated, separated, shift, device):
#     # 絶対パスを用いて split_map.pkl を読み込む
#     with open(SPLIT_MAP_PATH, "rb") as tf:
#         dictionary = pickle.load(tf)

#     pmap = torch.tensor(parcellated.astype("int16"), requires_grad=False).to(device)
#     hmap = torch.tensor(separated.astype("int16"), requires_grad=False).to(device)
#     combined = torch.stack((torch.flatten(hmap), torch.flatten(pmap)), axis=-1)
#     output = torch.zeros_like(hmap).ravel()
#     for key, value in dictionary.items():
#         key = torch.tensor(key, requires_grad=False).to(device)
#         mask = torch.all(combined == key, axis=1)
#         output[mask] = value
#     output = output.reshape(hmap.shape)
#     output = output.cpu().detach().numpy()
#     output = output * (np.logical_or(np.logical_or(separated > 0, parcellated == 87), parcellated == 138))
#     output = np.pad(output, [(32, 32), (16, 16), (32, 32)], "constant", constant_values=0)
#     output = np.roll(output, (-shift[0], -shift[1], -shift[2]), axis=(0, 1, 2))
#     return output
