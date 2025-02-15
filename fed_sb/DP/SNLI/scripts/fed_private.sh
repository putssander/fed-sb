CUDA_DEVICE=0

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python DP/SNLI/fed_trainer.py --lora_r 64 --epsilon 3
