# GAN for Text Summarization

Tensorflow re-implementation of [Generative Adversarial Network for Abstractive Text Summarization](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16238/16492).

## Requirements
* Python3
* Tensorflow >= 1.4 (tested on Tensorflow 1.4.1)
* numpy
* tqdm
* sklearn
* rouge
* pyrouge

You can use the python package manager of your choice (pip/conda) to install the dependencies. The code is tested on Ubuntu 16.04 operating system.

## Quickstart
* Dataset

    Please follow the instructions [here](https://github.com/abisee/cnn-dailymail) for downloading and preprocessing the CNN/DailyMail dataset. After that, store data files ```train.bin```, ```val.bin```, ```test.bin``` and vocabulary file ```vocab``` into specified data directory, e.g., ```./data/```.

* Prepare negative samples for discriminator

    You can download the prepared data ```discriminator_train_data.npz``` for discriminator from [dropbox](https://www.dropbox.com/s/i1otqkrsgup63pt/discriminator_train_data.npz?dl=0) and store into specified data directory, e.g., ```./data/```. 

* Train the full model
    
    ```
    python3 main.py --mode=train --data_path=./data/train.bin --vocab_path=./data/vocab --log_root=./log --pretrain_dis_data_path=./data/discriminator_train_data.npz --restore_best_model=False
    ```

* Decode
    
    ```
    python3 main.py --mode=decode --data_path=./data/test.bin --vocab_path=./data/vocab --log_root=./log --single_pass=True
    ```

## References

- [1] [Generative Adversarial Network for Abstractive Text Summarization](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16238/16492) (AAAI-18)
- [2] https://github.com/abisee/pointer-generator  
- [3] https://github.com/LantaoYu/SeqGAN 
