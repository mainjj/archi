# Total Architecture

## Predict
```shell
python .\predict.py -r .\pretrained_models\thermal_models\person.pt -c  .\pretrained_models\model_path.txt -l .\pretrained_models\thermal_models\log_model.pickle -s .\data\test_images\ -p .\runs\ -n exp
```
## Train
### make dataset
```shell
python .\predict.py -r .\pretrained_models\thermal_models\person.pt -c  .\pretrained_models\model_path.txt -s .\data\test_images\ --task save
```
### labeling dataset
```shell
python .\labeling.py -d .\runs\exp\save_for_labeling.csv -sp .\runs\exp\
```
### train logistic regressor
```shell
python .\logistic_train.py -d .\runs\exp\train_data.csv -p .\runs\exp\ -n logi
```