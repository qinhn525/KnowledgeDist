accelerate launch --config_file "/home/qhn/Codes/Projects/KnowledgeDist/Configs/accelerate_config.yaml" \
    "/home/qhn/Codes/Projects/KnowledgeDist/train_seq.py" \
    --max_seq_length 256 \
    --log_path "/home/qhn/Codes/Projects/KnowledgeDist/Logs/train_seq.log" \
    --epoch 5