accelerate launch --config_file "/home/qhn/Codes/Projects/KnowledgeDist/Configs/accelerate_config.yaml" \
    "/home/qhn/Codes/Projects/KnowledgeDist/train.py" \
    --max_seq_length 256 \
    --log_path "/home/qhn/Codes/Projects/KnowledgeDist/Logs/6.18-1.log" \
    --num_training_steps 200