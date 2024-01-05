from transformers import TrainingArguments
from model.mistral_prep import MistralPrep
from data_processing.data_formatter import DataFormatter
from trl import SFTTrainer

class MistralTrainer:
    def __init__(self, dataset_path, model_name_or_path, r, alpha):
        ImportingMistralPrep = MistralPrep(model_name_or_path = model_name_or_path, r = r,alpha = alpha)
        self.model, self.tokenizer, self.peft_config   =  ImportingMistralPrep.prep_model()
        self.formatted_data = DataFormatter.prepare_dataset(dataset_path) 
        self.output_dir =  ImportingMistralPrep.output_dir
        self.per_device_train_batch_size = ImportingMistralPrep.per_device_train_batch_size
        self.gradient_accumulation_steps = ImportingMistralPrep.gradient_accumulation_steps
        self.optim = ImportingMistralPrep.optim
        self.learning_rate = ImportingMistralPrep.learning_rate
        self.lr_scheduler_type = ImportingMistralPrep.lr_scheduler_type
        self.save_strategy = ImportingMistralPrep.save_strategy
        self.logging_steps = ImportingMistralPrep.logging_steps
        self.num_train_epochs = ImportingMistralPrep.num_train_epochs
        self.max_steps = ImportingMistralPrep.max_steps
        self.fp16 = ImportingMistralPrep.fp16

    def train(self):
        training_arguments = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=self.per_device_train_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            optim=self.optim,
            learning_rate=self.learning_rate,
            lr_scheduler_type=self.lr_scheduler_type,
            save_strategy=self.save_strategy,
            logging_steps=self.logging_steps,
            num_train_epochs=self.num_train_epochs,
            max_steps=self.max_steps,
            fp16=self.fp16,
            push_to_hub=False
        )

        trainer = SFTTrainer(
            model=self.model,
            train_dataset=self.formatted_data,
            peft_config=self.peft_config,
            dataset_text_field="text",
            args=training_arguments,
            tokenizer=self.tokenizer,
            packing=False,
            max_seq_length=1024
        )

        trainer.train()
