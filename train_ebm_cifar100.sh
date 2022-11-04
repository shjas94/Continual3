for memory_size in 2000 500 200; do
    for ((seed=0; seed<15; seed+=1)); do
        python train_EBM.py --dataset='cifar100' --num_classes=100 --num_tasks=20 --batch_size=32 --test_batch_size=32 \
    --model='resnet_18' --norm='continualnorm' --optimizer='sgd' --lr=0.1 --criterion='contrastive_divergence' --weight_decay=1e-05 --epoch=1 --memory_option='bin_based' \
    --img_size=32 --memory_size="$memory_size" --learning_mode='online' --run_name="contrastive_divergence_(memory_size_$memory_size)_(lr_5e-04)_(lam_0.0)_(SEED $seed)" --lam=0.0 --seed="$seed"
    done
done