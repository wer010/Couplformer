# This is the code implement for Couplformer

The code is based on Pytorch

## Model Training

run train.py to train a model, you can train model on single GPU by it, usually use for CIFAR dataset.
the config for training a 6 layers Couplformer with 128 embedding dimension on CIFAR100 is listed below 
> /home/lanhai/Projects/Compact-Transformers-main/data
--dataset
cifar100
--model_type
kct_sd
--num_layers
6
--num_heads
64
--mlp_ratio
3
--embedding_dim
128
--eval_every
1
--train_batch_size
100

ddp_train.py is for DistributedDataParallel on Imagenet-1K.