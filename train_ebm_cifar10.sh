python train_EBM.py --dataset='cifar10' --num_classes=10 --num_tasks=5 --batch_size=20 --test_batch_size=20 \
--model='resnet_18' --optimizer='sgd' --norm='continualnorm' --lr=0.01 --criterion='nll_energy' --weight_decay=1e-03 --epoch=1 --memory_option='bin_based' \
--img_size=32 --memory_size=20 --learning_mode='online' --run_name='nll_memory_20_10_27_final' --lam=2.0