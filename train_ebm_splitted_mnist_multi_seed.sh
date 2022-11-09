memory_size=2000
for ((seed=0; seed<5; seed+=1)); do
    python train_EBM.py --dataset='splitted_mnist' --num_classes=10 --num_tasks=5 --batch_size=32 --test_batch_size=32 \
    --model='resnet_18' --norm='continualnorm' --optimizer='adam' --lr=1e-04 --criterion='contrastive_divergence' --weight_decay=0.0 --epoch=1 --memory_option='low_energy' \
    --img_size=28 --memory_size="$memory_size" --learning_mode='online' --run_name="contrastive_divergence_memory_($memory_size)_with_aug_lr_5e-05_alpha_2.0_beta_2.0_seed_($seed)_with_rep_loss" --alpha=1.0 --beta=1.0 --seed="$seed" --num_channels=1
done
python calculate_mean_acc_forg.py --memory_size="$memory_size" --dataset="splitted_mnist" 