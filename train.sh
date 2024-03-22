python -m torch.distributed.launch --nproc_per_node=2 /workspace/torch-hmsm/train.py --maxdisp 128 \
--lr 1e-3 --epochs 120 --lrepochs 10,20,30,40,50,60,70,80,90,100,110:2 \
--batch_size 1 --test_batch_size 1 \
--logdir ./workspace/checkpoints/hmsmnet \
--summary_freq 1 \
--save_freq 10 \
--log_freq 100 \
--trainlist /workspace/CFNet/filenames/train_whu_dis.md.txt \
--testlist /workspace/CFNet/filenames/val_whu_dis.md.txt 
