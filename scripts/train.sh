# Train for video mode
CUDA_VISIBLE_DEVICES=0 python train.py --dataroot /path --name ROMA_name --dataset_mode unaligned_double --no_flip --local_nums 64 --display_env ROMA_env --model roma --side_length 7 --lambda_spatial 5.0 --lambda_global 5.0 --lambda_motion 1.0 --atten_layers 1,3,5 --lr 0.00001

# Train for image mode
CUDA_VISIBLE_DEVICES=0 python train.py --dataroot /path --name ROMA_name --dataset_mode unaligned --local_nums 64 --display_env ROMA_env --model roma --side_length 7 --lambda_spatial 5.0 --lambda_global 5.0 --atten_layers 1,3,5 --lr 0.00001