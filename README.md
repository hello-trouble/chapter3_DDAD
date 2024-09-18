# chapter3_DDAD
code for chapter3 ：Dual-route Domain Adaptation for Coal Mine Images Dehazing Based on Enhancement


#train command: 

CUDA_VISIBLE_DEVICES=1,0 python train_2stage0507.py --model  dehaze0507 --dataroot  /home/Qyx/datasets/coal_datasets_0608   --dataset_mode  alignedclahe  --name widthclahe_0507  --batchSize 16   --fineSize 256     --lr 0.0001   --niter 160 --niter_decay 160  --display_freq 160 --print_freq 160  --no_html  --unlabel_decay 0.99 --save_epoch_freq 1   --lambda_syn_Dehazing  50    --lambda_syn_vgg 5  --lambda_rs_Dehazing_DC 0 --lambda_rs_Dehazing_TV  0.001 --lambda_recon 10   --gpu_ids 0,1   --lambda_gan_un 1    --global_imageD  --lambda_real_Dehazing 10 

