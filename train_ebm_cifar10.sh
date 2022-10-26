python train_EBM.py --dataset='cifar10' --num_classes=10 --num_tasks=5 --batch_size=20 --test_batch_size=20 \
--model='resnet_18' --norm='continualnorm' --lr=1e-04 --criterion='contrastive_divergence' --weight_decay=0.0 --epoch=1 --memory_option='bin_based' \
--img_size=32 --memory_size=20 --learning_mode='online' --run_name='contrastive_divergence_memory_20' --lam=1e-02