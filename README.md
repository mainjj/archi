# Total Architecture

## Predict
```shell
python .\predict.py -r .\models\yolov7.pt -c  .\models\model_path.txt -l .\models\thermal_models\log_model.pickle -s .\data\test_images\ -p .\runs\ -n exp
```
## Train
### make dataset
```shell
python .\predict.py -r .\models\yolov7.pt -c  .\models\model_path.txt -s .\data\test_images\ --task save
```
### labeling dataset
```shell
python .\labeling.py -d .\runs\exp\save_for_labeling.csv -sp .\runs\exp\
```
### train logistic regressor
```shell
python .\logistic_train.py -d .\runs\exp\train_data.csv -i 1000 -p .\runs\exp\ -n logi
```