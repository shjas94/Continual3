python train_EBM.py --dataset='tiny_imagenet' --batch_size=20 --test_batch_size=20 --norm='continualnorm' \
--img_size=64 --num_classes=200 --model='resnet_18' --num_tasks=100 --run_name='tiny_imagenet_after_loss_change_final_changed_energy_calc' --epoch=1 --lr=1e-3 --criterion='nll_energy' \
--learning_mode='online' --weight_decay=0.0 --memory_size=20 --memory_option='bin_based'