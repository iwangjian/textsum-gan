# GAN for Text Summarization

Tensorflow re-implementation of "[Generative Adversarial Network for Abstractive Text Summarization](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16238/16492)" (AAAI-18).

## Requirements
* Python3 (tested on Python 3.6)
* Tensorflow >= 1.4 (tested on Tensorflow 1.4.1)
* numpy
* tqdm
* sklearn
* rouge
* pyrouge

You can use the python package manager of your choice (pip/conda) to install the dependencies. The code is tested on Ubuntu 16.04 operating system.

## Quickstart
* Dataset

    Please follow the instructions [here](https://github.com/abisee/cnn-dailymail) for downloading and preprocessing the CNN/DailyMail dataset. After that, store data files ```train.bin```, ```val.bin```, ```test.bin``` and vocabulary file ```vocab``` into specified data directory, e.g. ```./data/```

* Prepare negative samples for discriminator 

    You can download the generated data ```discriminator_train_data.npz``` for discriminator from [dropbox](https://www.dropbox.com/s/i1otqkrsgup63pt/discriminator_train_data.npz?dl=0). Meanwhile, you can follow the instructions below to prepare negative samples by yourself.
    
    Firstly, pretrain generator for some steps:
    ```
    python3 main.py --mode=pretrain --data_path=./data/train.bin --vocab_path=./data/vocab --log_root=./log --restore_best_model=False
    ```
    After pretraining some steps, stop it, then restore the model for training (**NOTE:** Set ```restore_best_model``` as ```True``` this step):
    ```
    python3 main.py --mode=pretrain --data_path=./data/train.bin --vocab_path=./data/vocab --log_root=./log --restore_best_model=True
    ```
    Secondly, decode training data using pretrained generator:
    ```
    python3 main.py --mode=decode --data_path=./data/train.bin --vocab_path=./data/vocab --log_root=./log --single_pass=True
    ```
    Finally, generate ```.npz``` file containing both positive and negative samples:
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
[1] ["Generative Adversarial Network for Abstractive Text Summarization"](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16238/16492) (AAAI-18) <br />
[2] https://github.com/abisee/pointer-generator <br />
[3] https://github.com/LantaoYu/SeqGAN 
