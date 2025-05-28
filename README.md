# generate cutout from acdata

This repository provides functions to download, process, and visualize cutout images from exaSPIM datasets stored in OME-Zarr format.

## Installation

1. Clone this repository:
   ```sh
   git clone https://github.com/wanqingy/ac_cutout.git
   cd ac_cutout
   ```

2. (Recommended) Create and activate a virtual environment:
   ```sh
   uv sync
   source .venv/bin/activate
   ```

## Usage

See [`example.ipynb`](example.ipynb) for a complete workflow, including:
- Downloading a cutout from a remote OME-Zarr dataset
- Rescaling and visualizing the image
- Viewing with microviewer or matplotlib
