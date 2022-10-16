python train_EBM.py --dataset='cifar10' --num_classes=10 --num_tasks=5 --batch_size=10 --test_batch_size=20 \
--model='resnet_18' --norm='continualnorm' --lr=1e-04 --criterion='nll_energy' --epoch=20 --memory_option='bin_based' \
--img_size=32 --memory_size=50 --run_name='resnet_18_3'