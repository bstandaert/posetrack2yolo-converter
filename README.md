# posetrack2yolo-converter
Convert your `PoseTrack18` or `PoseTrack21` folder to `YOLO` compatible folder to train your own models on a detection task.

## Installation
```bash
git clone https://github.com/bstandaert/posetrack2yolo-converter.git
cd posetrack2yolo-converter
pip install -r requirements.txt
```

## Usage
```bash
python main.py --annotation-path /path/to/annotations --data-path /path/to/data --output-path /path/to/output
```

### Expected annotation directory structure

```text
/path/to/annotations
├── train
│   └── *.json
├── val
│   └── *.json
└── test
    └── *.json
```

### Expected data directory structure

```text
/path/to/data
└── images
    ├── test
    │   ├── <video_name>
    │   │   └── *.jpg
    │   └── ...
    ├── train
    │   ├── <video_name>
    │   │   └── *.jpg
    │   └── ...
    └── val
        ├── <video_name>
        │   └── *.jpg
        └── ...
```

### Output directory structure

```text
.
├── test
│   ├── <video_name>_<*>.jpg
│   ├── <video_name>_<*>.txt
│   └── ...
├── train
│   ├── <video_name>_<*>.jpg
│   ├── <video_name>_<*>.txt
│   └── ...
└── val
    ├── <video_name>_<*>.jpg
    ├── <video_name>_<*>.txt
    └── ...
```

### Bounding box format conversion

The bounding box format is converted from `PoseTrack -> YOLO` format. 

```text
[x1, y1, w, h] -> [cxn, cyn, wn, hn]
```

Where `x1`, `y1` are the top left bbox coordinates. `w` and `h` are the width and heights of the bbox.
Finally, `cxn` and `cyn` are the normalized center coordinates of the bounding box according to the image dimension.
`wn` and `hn` are the normalized width and height.

#### Example on files

PoseTrack annotation file:
```text
{
    "images": [
        ...
        {
            "id": 10000010023,
            "file_name": "images/train/000001_bonn_train/000023.jpg",
            ...
        },
   ]
   "annotations": [
        ...,
        {
            ...,
            "bbox": [
                174.6530321843037,
                77.0554926387316,
                219.543666404406,
                281.9445073612684
            ],
            "image_id": 10000010023,
        },
        {
            ...
            "bbox": [
                328.2609044374924,
                49.97361641755753,
                181.7501135847342,
                309.02638358244246
            ],
            "image_id": 10000010023,
        },
   ],
   ...
}
```

YOLO annotation file :
```text
1 0.44441385216641666 0.605632628664905 0.3430369787568844 0.7831791871146344
1 0.6548999394216555 0.5680189116910521 0.2839845524761472 0.8584066210623402
```