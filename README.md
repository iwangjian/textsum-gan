# textsum-gan

Tensorflow re-implementation of "[Generative Adversarial Network for Abstractive Text Summarization](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16238/16492)" (AAAI-18).

## Dependencies
* Python3 (tested on Python 3.6)
* Tensorflow >= 1.4 (tested on Tensorflow 1.4.1)
* numpy
* tqdm
* sklearn
* rouge
* pyrouge

You can use the python package manager of your choice (pip/conda) to install the dependencies. The code is tested on Ubuntu 16.04 operating system.

## Quick Start
* Dataset

    Please follow the instructions [here](https://github.com/abisee/cnn-dailymail) for downloading and preprocessing the CNN/DailyMail dataset. After that, store data files ```train.bin```, ```val.bin```, ```test.bin``` and vocabulary file ```vocab``` into specified data directory, e.g. ```./data/```

* Pretrain generator
```
python3 main.py --mode=pretrain --data_path=./data/train.bin --vocab_path=./data/vocab --log_root=./log --restore_best_model=False
```
In this period, you could pretrain generator for some steps and stop it, then restore the model for training:
```
python3 main.py --mode=pretrain --data_path=./data/train.bin --vocab_path=./data/vocab --log_root=./log --restore_best_model=True
```
(**NOTE:** Set ```restore_best_model``` as ```True``` this step)

* Generate pos/neg samples for pretraining discriminator

First, you could decode training data by pretrained generator as negative samples:
```
python3 main.py --mode=decode --data_path=./data/train.bin --vocab_path=./data/vocab --log_root=./log --single_pass=True
```
Second, we provide a simple script ```gen_sample.py``` to generate ```.npz``` file containing both positive and negative samples:

```
python3 gen_sample.py --data_dir=./data --decode_dir=./log/decode_xxxx --vocab_path=./data/vocab
```
After that, ```discriminator_train_data.npz``` is generated in ```data_dir```.

* Train the full model
    
```
python3 main.py --mode=train --data_path=./data/train.bin --vocab_path=./data/vocab --log_root=./log --pretrain_dis_data_path=./data/discriminator_train_data.npz --restore_best_model=False
```

* Decode
    
```
python3 main.py --mode=decode --data_path=./data/test.bin --vocab_path=./data/vocab --log_root=./log --single_pass=True
```

## References
[1] [Generative Adversarial Network for Abstractive Text Summarization](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16238/16492)" (AAAI-18) 

[2] https://github.com/abisee/pointer-generator 

[3] https://github.com/LantaoYu/SeqGAN 
