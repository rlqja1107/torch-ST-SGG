# Dataset: Visual Genome 50

We follow the same-preprocessing strategy used in [IE-Trans](https://arxiv.org/pdf/2203.11654.pdf). Please download the linked files to prepare the dataset.

* Download the raw image dataset followed by links [part1 (9 Gb)](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip), [part2 (5 Gb)](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip). After downloading the two files, put the datasets into **datasets/vg/50** directory. 



* Download the [image_data.json](https://drive.google.com/file/d/1gfpXlVJkQsVg7-psFlPqU9ZBo3xsuUHF/view?usp=share_link) and [meta-data(VG-50)](https://drive.google.com/file/d/1jWOrsxkRQ5Ov-5jdfG3jOjZMQktySf-k/view?usp=share_link) and put this data into **datasets/vg/50** directory.


You should check the dataset path in `maskrcnn_benchmark/config/paths_catalog.py`. Refer to the example of config as: 
```  python      
# maskrcnn_benchmark/config/paths_catalog.py => 129 line
"50VG_stanford_filtered_with_attribute": { # check the path associated with dataset  
    "img_dir": "vg/50/VG_100k",
    "roidb_file": "vg/50/VG-SGG-with-attri.h5",
    "dict_file": "vg/50/VG-SGG-dicts-with-attri.json",
    "image_file": "vg/50/image_data.json",
},
```  

## I-Trans Dataset

To reproduce the **I-Trans+ST-SGG** performance, we also provide the dataset adopted the I-Trans for each SGG model. Following the IE-Trans, we apply *k_i* parameter, which decide the amount of changing the annotated dataset, as 0.7. For more details, refer to [IE-Trans](https://github.com/waxnkw/IETrans-SGG.pytorch/blob/master/MODEL_ZOO.md) To run the I-Trans+ST-SGG, download the below pickle file and put them on `datasets/vg/50/{model_name}` directory.
  
* [Motif+I-Trans](https://drive.google.com/file/d/1WRaSADSdjujQzzEn4wqwE6_O2ynWnlCm/view?usp=sharing)  
* [VCTree+I-Trans](https://drive.google.com/file/d/1oJI-4FiqQL07VUC5a_lWFuwJ_nRMUMu5/view?usp=sharing)  
* [BGNN+I-Trans](https://drive.google.com/file/d/1xeQVIc2GjhlMkB6KH-12XT2e1pxZzFrr/view?usp=sharing)  
* [HetSGG+I-Trans](https://drive.google.com/file/d/1IEWE6aUwO40T9Oqs8mcbHUQQ-zZlBskt/view?usp=sharing)  




## Directory Structure for Dataset

```python
root  
├── datasets 
│   └── vg
│       └── 50 
|           │── Category_Type_Info.json
│           │── VG-SGG-with-attri.h5
│           │── VG-SGG-dicts-with-attri.json
│           ├── VGKB.json
│           ├── vg_sup_data
│           ├── vg_clip_logits.pk
│           ├── CCKB.json
│           ├── cc_clip_logits
│           └── VG_100k
│                 └── *.png
```

* *Category_Type_Info.json* is for HetSGG predictor.