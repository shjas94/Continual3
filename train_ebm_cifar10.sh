python train_EBM.py --dataset='cifar10' --num_classes=10 --num_tasks=5 --batch_size=32 --test_batch_size=20 \
--model='resnet_18' --norm='continualnorm' --lr=1e-04 --criterion='nll_energy' --epoch=5 --memory_option='bin_based' \
--img_size=32 --memory_size=20 --learning_mode='offline' --run_name='test_offline_bin_memory_20_batch_32_lr_1e-_epoch_5'