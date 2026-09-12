# VL-Adapter

* Authors: [Yi-Lin Sung](https://ylsung.github.io/), [Jaemin Cho](https://j-min.io/), [Mohit Bansal](https://www.cs.unc.edu/~mbansal/)
* Paper: ["VL-Adapter: Parameter-Efficient Transfer Learning for Vision-and-Language Tasks"](https://arxiv.org/abs/2112.06825) (CVPR 2022)

We evaluate VL-adapter in a unified multi-task
setup on both image-text and video-text benchmarks. For the image-text tasks, we use four diverse V&L datasets: VQAv2, GQA, NLVR2, and MSCOCO image captioning. For video-text tasks, we use TVQA, How2QA, TVC, and YC2C. 

Our results demonstrate that training the adapter with the weight-sharing technique (4.18% of total parameters for image-text tasks and 3.39% for video-text tasks) can match
the performance of fine-tuning the entire model.

![](assets/vl_adapter_teaser.png)

** Note **
Please go into CLIP-ViL folder and follow the README there for running the experiments of adapters on CLIP-ViL. This README is for adapters on VL-Bart.


## Installation

```
# Create python environment (optional)
conda create -n vlt5 python=3.8
source activate vlt5

# Install python dependencies
pip install -r requirements.txt

# Download T5/BART backbone checkpoint
python download_backbones.py

# For MSCOCO captioning evaluation (optional; for captioning only)
python -c "import language_evaluation; language_evaluation.download('coco')"
```

## Code structure
```bash
# Store images, features, and annotations
./datasets
    COCO/
        images/
        clip_featuers/
    VG/
        images/
        clip_features/
    GQA/
        images/
        clip_features/
    nlvr/
        images/
        clip_features/
    vqa/
    lxmert/

    video/
        ann/
        vis_features

# Train VL-T5 with adapters
./VL-T5/
    src/
        modeling_t5.py modeling_bart.py                       <= VL-T5/VL-BART model classes
        pretrain.py, pretrain_data.py, pretrain_model.py      <= pretraining
        vqa.py, vqa_data.py vqa_model.py ...                  <= fine-tuning on downstream tasks (ex. VQA, GQA, NLVR2)
        multitask.py, multitask_data.py multiask_model.py     <= multitask learning on 7 downstream tasks
        param.py                                              <= (argparse) configuration
        tokenization.py                                       <= custom tokenizer
        utils.py, dist_utils.py                               <= utility functions
    snap/                                                     <= store weight checkpoints
    scripts/                                                  <= bash scripts for pretraining and finetuning
```

## Data

### Image-text dataset

The processed CLIP features are hosted on the HuggingFace Hub at
[**ylsung/VL-Adapter-datasets**](https://huggingface.co/datasets/ylsung/VL-Adapter-datasets).
(The previous Google Drive link is no longer available.)

Run the restore script from the root of this repository to rebuild `datasets/` in
exactly the layout shown in "Code structure":

```bash
pip install huggingface_hub pyarrow h5py numpy

# downloads ~49 GB and expands it into ./datasets (~145 GB on disk)
python tools/restore_datasets.py --out ./datasets
```

The script downloads one shard at a time and deletes it right after expanding it,
so it needs less than ~1 GB of scratch space on top of the final tree. It is
resumable: if it is interrupted, rerun the same command and it skips whatever is
already on disk.

To restore only part of the data:

```bash
python tools/restore_datasets.py --out ./datasets --datasets GQA --datasets nlvr
python tools/restore_datasets.py --out ./datasets --only annotations
```

On the Hub the per-image `.h5` feature files are packed into parquet shards,
because 621,783 loose files exceed the Hub's per-repository limits.
`restore_datasets.py` unpacks them back into the individual `<img_id>.h5` files
that `VL-T5/src/*_clip_data.py` reads; the restored feature arrays are bit-exact.

#### A note on raw images

`datasets/*/images/` is **not** needed by any script in `VL-T5/scripts/image/` —
those read `clip_features`. Raw images are only required for the end-to-end pixel
training paths (`VL-T5/src/*_raw_data.py`) and for extracting your own features.

The NLVR2 photographs are not redistributed, because the NLVR2 authors do not own
their copyright. Request them from
[lil-lab/nlvr](https://github.com/lil-lab/nlvr/tree/master/nlvr2) and place them
in `datasets/nlvr/images/`. COCO, GQA and Visual Genome images are available from
[COCO](https://cocodataset.org/#download),
[GQA](https://cs.stanford.edu/people/dorarad/gqa/download.html) and
[Visual Genome](https://homes.cs.washington.edu/~ranjay/visualgenome/api.html).

#### Re-creating the Hub copy

`tools/convert_zip_to_hf.py` is the converter that produced the Hub dataset from
the original `vlt5_dataset` archive, kept here for reproducibility.

### Extract your own CLIP features
Please refer to `feature_extraction` for more details.

### Video-text dataset
Please go to [VALUE](https://github.com/VALUE-Leaderboard/DataRelease) to download the ViT processed data.

## Run different approaches
The following scripts can run every approach with the best hyper-parameters.

### Image dataset

```bash
# Full fine-tuning
cd VL-T5/
bash scripts/image/full_finetuning.sh 1

# Single Adapter
cd VL-T5/
bash scripts/image/single_adapter.sh 1

# Multiple Adapters
cd VL-T5/
bash scripts/image/multiple_adapters.sh 1

# Hyperformer
cd VL-T5/
bash scripts/image/hyperformer.sh 1

# Single Compacter
cd VL-T5/
bash scripts/image/single_compacter.sh 1

# Multiple Compacters
cd VL-T5/
bash scripts/image/multiple_compacters.sh 1

# Single LoRA
cd VL-T5/
bash scripts/image/single_lora.sh 1

# Multiple LoRA
cd VL-T5/
bash scripts/image/multiple_lora.sh 1

# Single Prompt
cd VL-T5/
bash scripts/image/single_prompt.sh 1

# Multiple Prompts
cd VL-T5/
bash scripts/image/multiple_prompts.sh 1
```

### Video dataset

```bash
# Full fine-tuning
cd VL-T5/
bash scripts/video/full_finetuning.sh 1

# Single Adapter
cd VL-T5/
bash scripts/video/single_adapter.sh 1

# Single LoRA
cd VL-T5/
bash scripts/video/single_lora.sh 1

# Single Prompt
cd VL-T5/
bash scripts/video/single_prompt.sh 1

```


## Acknowledgement

This repo is adapted from [VLT5](https://github.com/j-min/VL-T5). I also borrow some codes from [CLIP](https://github.com/openai/CLIP), [CLIP-ViL](https://github.com/clip-vil/CLIP-ViL), [Compacter](https://github.com/ylsung/compacter), [Hyperformer](https://github.com/rabeehk/hyperformer) and [Prefix-tuning](https://github.com/XiangLi1999/PrefixTuning).


## Reference

Please cite our paper if you use our models in your project.

```bibtex
@inproceedings{sung2022vladapter,
  title     = {VL-Adapter: Parameter-Efficient Transfer Learning for Vision-and-Language Tasks},
  author    = {Yi-Lin Sung, Jaemin Cho, Mohit Bansal},
  booktitle = {CVPR},
  year      = {2022}
}
```