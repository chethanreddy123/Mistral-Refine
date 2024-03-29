from model.mistral_trainer import MistralTrainer

def main():
    dataset_path = "code/test.csv"
    model_name_or_path = "TheBloke/Mistral-7B-Instruct-v0.1-GPTQ"
    r = 8
    alpha = 8
    trainer = MistralTrainer(dataset_path, model_name_or_path, r, alpha)
    trainer.train()


if __name__ == "__main__":
    main()