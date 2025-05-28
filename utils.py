import tensorstore as ts
import numpy as np
import os
import json
from urllib.parse import urljoin
from urllib.request import urlopen

def load_translation_from_json(zarr_url, mip_level):
    try:
        if zarr_url.startswith("http"):
            json_url = urljoin(zarr_url.rstrip("/") + "/", "zarr.json")
            try:
                with urlopen(json_url) as f:
                    zattrs = json.load(f)
            except Exception:
                zattrs_url = urljoin(zarr_url.rstrip("/") + "/", ".zattrs")
                with urlopen(zattrs_url) as f:
                    zattrs = json.load(f)
        else:
            json_path = os.path.join(zarr_url, "zarr.json")
            try:
                with open(json_path, "r") as f:
                    zattrs = json.load(f)
            except Exception:
                zattrs_path = os.path.join(zarr_url, ".zattrs")
                with open(zattrs_path, "r") as f:
                    zattrs = json.load(f)

        if "attributes" in zattrs and "multiscales" in zattrs["attributes"]:
            multiscale = zattrs['attributes']["multiscales"][0]
        elif "multiscales" in zattrs:
            multiscale = zattrs["multiscales"][0]
        else:
            multiscale = {}

        translation_mip0 = [0.0, 0.0, 0.0]
        for t in multiscale.get("coordinateTransformations", []):
            if t.get("type") == "translation":
                translation_mip0 = t.get("translation", [0.0, 0.0, 0.0])[-3:]
                break

        translation_voxel = np.array(translation_mip0) // 2 ** int(mip_level)
        return translation_voxel.astype(int)
    except Exception as e:
        print(f"Warning: Failed to load translation: {e}")
        return np.array([0, 0, 0], dtype=int)

def download_cutout_from_zarr(
    zarr_url,
    center,
    size,
    out_path=None,
    mip="0",
    offset=None
):
    print("Opening TensorStore...")

    spec = {
        "driver": "zarr3",
        "kvstore": zarr_url,
        "path": mip,
    }

    store = ts.open(spec).result()
    domain = store.domain
    zyx_min = domain.inclusive_min[2:]
    zyx_max = tuple(x - 1 for x in domain.exclusive_max[2:])
    zyx_shape = domain.shape[2:]

    # Use user-supplied offset if given, else load from JSON
    if offset is None:
        offset = load_translation_from_json(zarr_url, 0)
    else:
        offset = np.array(offset)

    center_local = (np.array([int(c - o) for c, o in zip(center, offset)]) // (2 ** int(mip))).astype(int)
    size = [int(s // (2 ** int(mip))) for s in size]

    spatial_start = [int(max(mins, c - s // 2)) for c, s, mins in zip(center_local, size, zyx_min)]
    spatial_stop = [int(min(c + s // 2, maxs)) for c, s, maxs in zip(center_local, size, zyx_max)]

    final_shape = [stop - start for start, stop in zip(spatial_start, spatial_stop)]

    print(f"Global center={center}, Offset={offset}, Local center={center_local}")
    print(f"Clipped start={spatial_start}, stop={spatial_stop}, shape={final_shape}")
    print(f"Dataset shape: zyx = {zyx_shape}")

    cutout = store[0, 0,
                   spatial_start[0]:spatial_stop[0],
                   spatial_start[1]:spatial_stop[1],
                   spatial_start[2]:spatial_stop[2]].read().result()

    print(f"Saving cutout to {out_path} with shape {cutout.shape}")
    if out_path is not None:
        np.save(out_path, cutout)
    return cutout