python train_EBM.py --dataset=tiny_imagenet --batch_size=8 --test_batch_size=8 --norm='none' \
--img_size=64 --num_classes=200 --model=resnet_18 --num_tasks=20 --run_name=tiny_imagenet --epoch=40 --lr=1e-4 --criterion='nll_energy'