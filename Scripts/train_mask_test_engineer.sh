export CUDA_VISIBLE_DEVICES=3
accelerate launch --config_file "/home/qhn/Codes/Projects/KnowledgeDist/Configs/accelerate_config.yaml" \
    "/home/qhn/Codes/Projects/KnowledgeDist/Trainers/train_mask_test_engineer.py" \