set -x

torchrun -m verl.trainer.sft_trainer \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.messages_key=messages \
    data.micro_batch_size_per_gpu=2 \
    model.path=Qwen/Qwen2.5-0.5B-Instruct \
    trainer.project_name=gsm8k-sft \
    trainer.experiment_name=gsm8k-sft-Qwen2.5-0.5B-Instruct \
    trainer.total_epochs=4 \
    trainer.logger='["console","wandb"]'