#**********************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************
#python -m torch.distributed.launch --nproc_per_node=8 --master_port 30012 ddp_train.py  --epoch 200  --train_batch_size 96 --eval_batch_size 32 --model_type vit_base_16  --eval_every 10 --gradient_accumulation_steps 1  --learning_rate 0.0001 --loss_scale 0 --max_grad_norm 1.0 --num_workers 4  --warmup_epoch 20 --warmup_lr 0.001 --weight_decay 0.3
#python train.py /home/lanhai/Projects/Compact-Transformers-main/data --dataset cifar10 --model_type vic_10 --epoch 200 --embedding_dim 256 --eval_every 1 --train_batch_size 100
#python train.py /home/lanhai/Projects/Compact-Transformers-main/data --dataset cifar10 --model_type vic_8 --epoch 200 --embedding_dim 256 --eval_every 1 --train_batch_size 100
#python train.py /home/lanhai/Projects/Compact-Transformers-main/data --dataset cifar10 --model_type vic_6 --epoch 200 --embedding_dim 256 --eval_every 1 --train_batch_size 100
#python train.py /home/lanhai/Projects/Compact-Transformers-main/data --dataset cifar10 --model_type vic_10 --epoch 200 --embedding_dim 128 --eval_every 1 --train_batch_size 100
#python train.py /home/lanhai/Projects/Compact-Transformers-main/data --dataset cifar10 --model_type vic_8 --epoch 200 --embedding_dim 128 --eval_every 1 --train_batch_size 100
#python train.py /home/lanhai/Projects/Compact-Transformers-main/data --dataset cifar10 --model_type vic_6 --epoch 200 --embedding_dim 128 --eval_every 1 --train_batch_size 100
#python train.py /home/lanhai/Projects/Compact-Transformers-main/data --dataset cifar10 --model_type vic_10 --epoch 200 --embedding_dim 96 --eval_every 1 --train_batch_size 100
#python train.py /home/lanhai/Projects/Compact-Transformers-main/data --dataset cifar10 --model_type vic_8 --epoch 200 --embedding_dim 96 --eval_every 1 --train_batch_size 100
#python train.py /home/lanhai/Projects/Compact-Transformers-main/data --dataset cifar10 --model_type vic_6 --epoch 200 --embedding_dim 96 --eval_every 1 --train_batch_size 100
#python train.py /home/lanhai/Projects/Compact-Transformers-main/data --dataset cifar10 --model_type cct_8 --epoch 200 --eval_every 1 --train_batch_size 100
#python train.py /home/lanhai/Projects/Compact-Transformers-main/data --dataset cifar10 --model_type cvt_8 --epoch 200 --eval_every 1 --train_batch_size 100
#python train.py /home/lanhai/Projects/Compact-Transformers-main/data --dataset cifar10 --model_type swin_t --epoch 200 --eval_every 1 --train_batch_size 100
#python train.py /home/lanhai/Projects/Compact-Transformers-main/data --dataset cifar100 --model_type swin_t --epoch 200 --eval_every 1 --train_batch_size 100
#python train.py /home/lanhai/Projects/Compact-Transformers-main/data --dataset cifar10 --model_type swin_s --epoch 200 --eval_every 1 --train_batch_size 100
#python train.py /home/lanhai/Projects/Compact-Transformers-main/data --dataset cifar100 --model_type swin_s --epoch 200 --eval_every 1 --train_batch_size 100
#python train.py /home/lanhai/Projects/Compact-Transformers-main/data --dataset cifar10 --model_type cvt_8 --epoch 200 --eval_every 1 --train_batch_size 128
#python train.py /home/lanhai/Projects/Compact-Transformers-main/data --dataset cifar10 --model_type kct_8 --epoch 200 --eval_every 1 --train_batch_size 128
#python train.py /home/lanhai/Projects/Compact-Transformers-main/data --dataset cifar10 --model_type cct_8 --epoch 200 --eval_every 1 --train_batch_size 128
#python train.py /home/lanhai/Projects/Compact-Transformers-main/data --dataset cifar10 --model_type vic_8 --num_heads 8 --epoch 200 --eval_every 1 --train_batch_size 128
#python train.py /home/lanhai/Projects/Compact-Transformers-main/data --dataset cifar10 --model_type vic_8 --num_heads 16 --epoch 200 --eval_every 1 --train_batch_size 128
#python train.py /home/lanhai/Projects/Compact-Transformers-main/data --dataset cifar10 --model_type vic_8 --num_heads 32 --epoch 200 --eval_every 1 --train_batch_size 128
#python train.py /home/lanhai/Projects/Compact-Transformers-main/data --dataset cifar10 --model_type vic_8 --num_heads 64 --epoch 200 --eval_every 1 --train_batch_size 128
python train.py /home/lanhai/Projects/Compact-Transformers-main/data --dataset cifar10 --model_type vic_8 --num_heads 256 --epoch 200 --eval_every 1 --train_batch_size 128
