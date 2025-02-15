CUDA_DEVICE=3

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python DP/SNLI/trainer.py --lora_r 64 --epsilon 3

