# Pre-trained Model for VG-150

## Run the pre-trained model  

For evaluating the pre-trained model, please run the `run_shell/evaluation4pretrained_model.sh` code.  
Note that the `output directory` variable in shell code should be changed.

``` bash   
bash run_shell/evaluation4pretrained_model.sh  
```  

## **Motif**   
---  

### Vanilla  

Task | Link | result(.txt)
-- | -- | --
SGCls   | [link](https://drive.google.com/file/d/1ynO4vgaeCZYWcbYoe1EvyGucz6yGrd9I/view?usp=sharing) | [evaluation_res.txt](https://drive.google.com/file/d/1RTWR11vmEbTsJwkl2gXbvCIG3CrBnHvF/view?usp=sharing)
SGDet   | [link](https://drive.google.com/file/d/1qT-dmuP211W6hYJrru7FXngZH9jg2zlB/view?usp=sharing) | [evaluation_res.txt](https://drive.google.com/file/d/1b8tyQ0KCWFJe9dHHLo7orO3bPt_OY7Y8/view?usp=sharing)

### [Resampling](https://github.com/SHTUPLUS/PySGG)  

Task | Link | result(.txt)
-- | -- | --
SGCls   | [link](https://drive.google.com/file/d/1ZBMh-4yYfN81dxemysYYg5uacfmMvKTe/view?usp=sharing) | [evaluation_res.txt](https://drive.google.com/file/d/1NkfkXMAG287ZeagHSeYWZG6obpyQHg39/view?usp=sharing)
SGDet   | [link](https://drive.google.com/file/d/1O8Z6YWwT3nLJTsArP3d2-zUqi0hRbFee/view?usp=sharing) | [evaluation_res.txt](https://drive.google.com/file/d/14lJklpqj2KpElrc4qqA7zJWb2ZopBVc8/view?usp=sharing)

## **BGNN**  
---  
### [Resampling](https://github.com/SHTUPLUS/PySGG)  

Task | Link | result(.txt)
-- | -- | --
PredCls   | [link](https://drive.google.com/file/d/1MywMamIJZjCeVdXZrzDnbZZWeOcXTyYY/view?usp=sharing) | [evaluation_res.txt](https://drive.google.com/file/d/18iZcb3VN-WYbJCaEOIW2fj4msXPOULjb/view?usp=sharing)
SGCls   | [link](https://drive.google.com/file/d/12sQfBv-dtFH2Ie7TFbSuczkzf65kCB-f/view?usp=sharing) | [evaluation_res.txt](https://drive.google.com/file/d/1Qxug1jAOVg3UmB9bawNp8lGYGWf7V-1h/view?usp=sharing)
SGDet   | [link](https://drive.google.com/file/d/1O8Z6YWwT3nLJTsArP3d2-zUqi0hRbFee/view?usp=sharing) | [evaluation_res.txt](https://drive.google.com/file/d/1MAwDjYEMfrXXp2IcCieaEuoi7DTPG4rc/view?usp=sharing)

### [I-Trans](https://github.com/waxnkw/IETrans-SGG.pytorch/tree/master)  

Task | Link | result(.txt)
-- | -- | --
SGDet   | [link](https://drive.google.com/file/d/1SkkXelvXCLcqMtZS9p8rZ4pyhKnSOfah/view?usp=sharing) | [evaluation_res.txt](https://drive.google.com/file/d/1A7xloNEYxuZZTwC6kGXSzCLeMmbVjPgH/view?usp=sharing)

## **HetSGG**  
---  
### [Resampling](https://github.com/SHTUPLUS/PySGG)  

Task | Link | result(.txt)
-- | -- | --
SGCls   | [link](https://drive.google.com/file/d/1vvmRSKWr-hJhpOHYxRRRIyqafwbFy99q/view?usp=sharing) | [evaluation_res.txt](https://drive.google.com/file/d/1Vv4TmXjq9ieHXFSdaZfdz91Le90SrV0U/view?usp=sharing)
SGDet   | [link](https://drive.google.com/file/d/1dadL5xi33qgqpdJdQ2mgaozixqOZUum8/view?usp=sharing) |[evaluation_res.txt](https://drive.google.com/file/d/1_xGRTjRAxRmuMX5nmvMYBZe35mfVkh_a/view?usp=sharing)

### [I-Trans](https://github.com/waxnkw/IETrans-SGG.pytorch/tree/master)  

Task | Link | result(.txt)
-- | -- | --
SGDet   | [link](https://drive.google.com/file/d/14lJklpqj2KpElrc4qqA7zJWb2ZopBVc8/view?usp=sharing) |[evaluation_res.txt](https://drive.google.com/file/d/1Yfn96aLmtj8BiPhLnfER0a8IO0FYDCAU/view?usp=sharing)

## **Notice**  

The performance of pre-trained models can slightly be different with performance described in paper since the pre-processing of test set is different between [IE-Trans](https://github.com/waxnkw/IETrans-SGG.pytorch/blob/master/DATASET.md) and [BGNN](https://github.com/SHTUPLUS/PySGG/blob/main/DATASET.md). Therefore, if you want to reproduce the performance, you need to evaluate it on BGNN repository. 

Instead, we provide the detailed evaluation result for each model.
