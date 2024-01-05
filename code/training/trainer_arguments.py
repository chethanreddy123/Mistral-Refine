# trainer_arguments.py
from transformers import GPTQConfig, TrainingArguments

def get_training_arguments(output_dir, per_device_train_batch_size, gradient_accumulation_steps,
                           optim, learning_rate, lr_scheduler_type, save_strategy, logging_steps,
                           num_train_epochs, max_steps, fp16, push_to_hub):
    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim=optim,
        learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler_type,
        save_strategy=save_strategy,
        logging_steps=logging_steps,
        num_train_epochs=num_train_epochs,
        max_steps=max_steps,
        fp16=fp16,
        push_to_hub=push_to_hub
    )