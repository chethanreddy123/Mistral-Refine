from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig



class MistralPrep:
    def __init__(
        self, 
        dataset_path, 
        model_name_or_path, 
        r, 
        alpha, 
        lora_dropout=0.1, 
        bias=None,  # Changed "None" (string) to None (NoneType)
        task_type="CASUAL_LM", 
        target_modules=["q_proj", "v_proj"],
        output_dir="mistral-finetuned", 
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1, 
        optim="paged_adamw_32bit",
        learning_rate=2e-4, 
        lr_scheduler_type="cosine",
        save_strategy="epoch", 
        logging_steps=100, 
        num_train_epochs=1,
        max_steps=250, 
        fp16=True, 
        push_to_hub=False
    ):
        self.dataset_path = dataset_path
        self.model_name_or_path = model_name_or_path
        self.r = r
        self.alpha = alpha
        self.lora_dropout = lora_dropout
        self.bias = bias
        self.task_type = task_type
        self.target_modules = target_modules
        self.output_dir = output_dir
        self.per_device_train_batch_size = per_device_train_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.optim = optim
        self.learning_rate = learning_rate
        self.lr_scheduler_type = lr_scheduler_type
        self.save_strategy = save_strategy
        self.logging_steps = logging_steps
        self.num_train_epochs = num_train_epochs
        self.max_steps = max_steps
        self.fp16 = fp16
        self.push_to_hub = push_to_hub

    # Add other methods here as needed:
    def prep_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        tokenizer.pad_token = tokenizer.eos_token
        quantization_config_loading = GPTQConfig(bits=4, disable_exllama=True, tokenizer=tokenizer)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            quantization_config=quantization_config_loading,
            device_map="auto"
        )

        model.config.use_cache = False
        model.config.pretraining_tp = 1
        model.gradient_checkpointing_enable()

        model = prepare_model_for_kbit_training(model)

        peft_config = LoraConfig(
            r=self.r, lora_alpha=self.alpha, lora_dropout=self.lora_dropout, bias=self.bias, task_type=self.task_type,
            target_modules=self.target_modules
        )

        model = get_peft_model(model, peft_config)

        return model, tokenizer, peft_config

        