# RDA Dataset Preparation

The provided dataset contains four subjects recorded six times each, resulting in 24 video sequences. Each video sequence has its corresponding environment map.

## Downloading

### Raw Video Data

The raw video sequences can be downloaded from here: [link](./). Note that *registration* is required for downloading the data. Download and extract the data in `data/RelightableDynamicActor`.

After downloading, extract the ZIP files. The structure of the data, for each video sequence, should be:
```sh
# ./data/RelightableDynamicActor/
|-- S[1..4]
    |-- T[1..6]
        |-- light_probe/
            |-- envmap
                |-- hdri_combined_low_res.hdr # 32x16 environment map
        |-- shot/
            |-- stream%03d.mp4 # RGB video streams [0..MaxVideos]
            |-- foregroundSegmentation_refined/
                |-- stream%03d.mp4 # Foreground matting streams [0..MaxVideos]
        |-- tracking/
```

### Processed Samples

In case you want to quickly try the method without going through all the [processing steps](#processing), 


## Processing

**1. Extract raw data**

After downloading the [raw video data](#raw-video-data), the MP4 files and camera calibration have to be extracted with (example with 300 frames for the first 48 cameras from S1/T1):
```sh
python tools/extract_studio_data.py \
  --input "./data/RelightableDynamicActor/S1/T1/shot" \
  --output "./data/processed/RelightableDynamicActor/S1/T1" \
  --start_frame 100 \
  --num_frames 300 \
  --subsample 1 \
  --start_cam 0 \
  --num_cams 48 \
  --segmentation_folder "foregroundSegmentation_refined"
```
**Important**: this script internally uses `ffmpeg`. Make sure you have it installed in your machine.

**2. Copy (or link) SMPL tracking data**

The SMPL tracking files are read out from the processed data by default. Therefore, copy or link the tracking files from the raw data. Example:
```sh
BASEPATH=`pwd`
cd ./data/processed/RelightableDynamicActor/S1/T1
ln -s "$BASEPATH/data/RelightableDynamicActor/S1/T1/tracking/smpl" .
``` 

**3. Compute coarse normal and texture maps**

Next, we compute coarse normals maps (from posed SMPL) and texture maps, which are used to train the Vid2Vid model from the original [Neural Actor](https://vcai.mpi-inf.mpg.de/projects/NeuralActor/) approach.
```sh
pushd ./videoavatars

# python generate_texture.py \
/CT/NeuralHuman3DScene/work/miniconda3/envs/texmaps/bin/python generate_texture.py \
  "../data/processed/RelightableDynamicActor/S1/T1" \
  "../data/RelightableDynamicActor/S1/T1/tracking/smpl" \
  --start_frame 100 \
  --num_frames 300 \
  --cameras "0..48" \
  --resolution 512

popd
```

<!-- **3. Copy SMPL tracking files** -->







