# CNN-LSTM CTC model for handwritten character recognition
## Usage
```
pipenv run python run.py [-h]
            [--train-data=<train>]
            [--dev-data=<train>]
            [--test-data=<test>]
            [--initial-model=<model>]
            [--train-batch-size=<N>]
            [--test-batch-size=<M>]
            [--epochs=<K>]
            [--learning-rate=<R>]
            [--learning-rate-factor=<F>]
            [--gpu=<G>]
            --model-dir=<model>
            --target=<target>

Options:
    -h, --help                  show this help message and exit
    --train-data=<train>        training data in json format
    --dev-data=<dev>            validation data in json format
    --test-data=<test>          test data in json format
    --initial-model=<model>     model path for initializing
    --train-batch-size=<N>      batch size for training [default: 16]
    --test-batch-size=<M>       batch size for test [default: 1]
    --epochs=<K>                number of training epoch [default: 30]
    --learning-rate=<R>         learning rate [default: 0.0001]
    --learning-rate-factor=<F>  attenuation rate of learning rate [default: 0.5]
    --gpu=<G>                   whether to use GPU or not [default: False]
    --model-dir=<dir>           output direcrory
    --traget=<target>           target labels in joson format
```
### example
An example for training with number images in [ETL6](http://etlcdb.db.aist.go.jp/?lang=ja)
```
pipenv run python run.py \
        --train-data example/number/number_train.json \
        --dev-data example/number/number_dev.json\
        --test-data example/number/number_test.json\
        --model-dir examplenumber \
        --target example/number/label.json
```
- Trained model: [example/CnnLstmModel_checkpoint_best.pth](example/CnnLstmModel_checkpoint_best.pth)
- Test result: [example/test_result.json](example/test_result.json)
