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
SGCls   | [link]() | [evaluation_res.txt](yes)
SGDet   | [link](https://drive.google.com/file/d/1qT-dmuP211W6hYJrru7FXngZH9jg2zlB/view?usp=sharing) | [evaluation_res.txt](yes)

### [Resampling](https://github.com/SHTUPLUS/PySGG)  

Task | Link | result(.txt)
-- | -- | --
SGCls   | [link](https://drive.google.com/file/d/1ZBMh-4yYfN81dxemysYYg5uacfmMvKTe/view?usp=sharing) | [evaluation_res.txt](yes)
SGDet   | [link](https://drive.google.com/file/d/1O8Z6YWwT3nLJTsArP3d2-zUqi0hRbFee/view?usp=sharing) | [evaluation_res.txt](yes)

## **BGNN**  
---  
### [Resampling](https://github.com/SHTUPLUS/PySGG)  

Task | Link | result(.txt)
-- | -- | --
PredCls   | [link](https://drive.google.com/file/d/1MywMamIJZjCeVdXZrzDnbZZWeOcXTyYY/view?usp=sharing) | [evaluation_res.txt](yes)
SGCls   | [link](https://drive.google.com/file/d/12sQfBv-dtFH2Ie7TFbSuczkzf65kCB-f/view?usp=sharing) | [evaluation_res.txt](yes)
SGDet   | [link](https://drive.google.com/file/d/1O8Z6YWwT3nLJTsArP3d2-zUqi0hRbFee/view?usp=sharing) | [evaluation_res.txt](yes)

### [I-Trans](https://github.com/waxnkw/IETrans-SGG.pytorch/tree/master)  

Task | Link | result(.txt)
-- | -- | --
SGDet   | [link](https://drive.google.com/file/d/1SkkXelvXCLcqMtZS9p8rZ4pyhKnSOfah/view?usp=sharing) | [evaluation_res.txt](yes)

## **HetSGG**  
---  
### [Resampling](https://github.com/SHTUPLUS/PySGG)  

Task | Link | result(.txt)
-- | -- | --
SGCls   | [link](https://drive.google.com/file/d/1vvmRSKWr-hJhpOHYxRRRIyqafwbFy99q/view?usp=sharing) | [evaluation_res.txt](yes)
SGDet   | [link](https://drive.google.com/file/d/1dadL5xi33qgqpdJdQ2mgaozixqOZUum8/view?usp=sharing) |[evaluation_res.txt](yes)

### [I-Trans](https://github.com/waxnkw/IETrans-SGG.pytorch/tree/master)  

Task | Link | result(.txt)
-- | -- | --
SGDet   | [link](s) |[evaluation_res.txt](yes)

## **Notice**  

The performance of pre-trained models can slightly be different with performance described in paper since the pre-processing of test set is different between [IE-Trans](https://github.com/waxnkw/IETrans-SGG.pytorch/blob/master/DATASET.md) and [BGNN](https://github.com/SHTUPLUS/PySGG/blob/main/DATASET.md). Therefore, if you want to reproduce the performance, you need to evaluate it on BGNN repository. 

Instead, we provide the detailed evaluation result for each model.
