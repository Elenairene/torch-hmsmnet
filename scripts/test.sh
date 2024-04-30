python ../test.py --maxdisp 128 \
--batch_size 1 --test_batch_size 1 \
--summary_freq 1 \
--save_freq 100 \
--log_freq 100 \
--savepath ../save/ \
--loadckpt ../checkpoints/hmsmnet/checkpoint_best.ckpt \
--testlist ../filenames/test_whu_dis.txt 