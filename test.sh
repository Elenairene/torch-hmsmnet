python /workspace/torch-hmsm/test.py --maxdisp 128 \
--batch_size 1 --test_batch_size 1 \
--summary_freq 1 \
--save_freq 100 \
--log_freq 100 \
--loadckpt /workspace/workspace/checkpoints/hmsmnet/checkpoint_best.ckpt \
--testlist /workspace/CFNet/filenames/train_whu_dis.md.txt 