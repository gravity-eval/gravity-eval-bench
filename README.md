# Gravity Evaluation
This repo contains code and data to run evaluations based on the paper [Objects in Generated Videos Are Slower Than They Appear: Models Suffer Sub-Earth Gravity and Don’t Know Galileo’s Principle...for now](https://gravity-eval.github.io/).
## Dependencies

```bash
pip install torch torchvision
pip install opencv-python
pip install numpy
pip install matplotlib
```

The script also requires a working [SAM3](https://github.com/facebookresearch/sam3) installation. 

## Folder Structure

### Ground-truth Frames and annotations:

```text
gravity/
├── sample_1/
│   ├── frame_0000.png
│   └── metadata.json
├── sample_2/
│   ├── frame_0000.png
│   └── metadata.json
└── ...
```
Use the frames to generate videos. 
We use the prompt:
```text
A video showing two identical balls being dropped from two different heights onto the ground. The camera is static and positioned to clearly capture the vertical motion of both balls. Both balls fall naturally under gravity, accelerating freely with no air resistance and hit the ground. The balls bounce a few times on the ground before coming to rest.
```


### Generated videos:
Structure the generations as shown below.

```text
<sample_folder>/
├── generated_1/
│   └── generated_video.mp4
├── generated_2/
│   └── generated_video.mp4
└── ...
```

## Run

Evaluate all samples:

```bash
python eval_gravity.py \
    --sample_folder /path/to/<sample_folder> \
```

## Output

For each sample, the script saves a `results.json` file inside the corresponding generated sample folder:

```text
<sample_folder>/
├── generated_1/
│   ├── generated_video.mp4
│   └── results.json
├── generated_2/
│   ├── generated_video.mp4
│   └── results.json
└── ...
```

The script creates h_ratio, t_ratio for each sample
where:

- `h_ratio` is the ratio of the measured drop heights.
- `t_ratio` is the squared ratio of the measured times of flight.


## Citation

If you find this evaluation useful in your research, please cite:

```bibtex
@InProceedings{Thozhiyoor_2026_CVPR,
    author    = {Thozhiyoor, Varun Varma and Tripathi, Shivam and Radhakrishnan, Venkatesh Babu and Bhattad, Anand},
    title     = {Objects in Generated Videos Are Slower Than They Appear: Models Suffer Sub-Earth Gravity and Don't Know Galileo's Principle...for now},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Findings},
    month     = {June},
    year      = {2026},
    pages     = {3830--3839}
}
```




