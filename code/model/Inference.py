import argparse
from peft import AutoPeftModelForCausalLM, TaskType
from transformers import GenerationConfig, AutoTokenizer, AutoModelForCausalLM
import torch

class InvoiceExtractor:
    def __init__(self, model_path, tokenizer_path):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            return_dict=True,
            torch_dtype=torch.float16,
            device_map="cuda"
        )

    def generate_invoice(self, raw_text):
        prompt = """###Human: Given raw text, identify and extract following details in JSON format...
                    (your prompt remains the same)
                    Raw text: {raw_text}
                    \n\n###Assistant: """
        prompt_template = prompt.format(raw_text=raw_text)
        inputs = self.tokenizer(prompt_template, return_tensors="pt").to("cuda")

        generation_config = GenerationConfig(
            do_sample=True,
            top_k=1,
            temperature=0.1,
            max_new_tokens=1024,
            pad_token_id=self.tokenizer.eos_token_id
        )

        # Assuming your peft model and task type are correctly defined
        # You might need to adapt this part based on your specific peft model and task type
        output = self.model.generate(**inputs, **generation_config)

        return self.tokenizer.decode(output[0], skip_special_tokens=True)

class InvoiceData:
    def __init__(self, extracted_data):
        self.extracted_data = extracted_data

    # Add methods or properties to process or manipulate the extracted data as needed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract invoice details from raw text.")
    parser.add_argument("raw_data", type=str, help="Raw text input for invoice extraction.")
    args = parser.parse_args()

    # Update these paths accordingly
    model_path = "checkpoint-50"
    tokenizer_path = "checkpoint-50"

    # Initialize the InvoiceExtractor
    invoice_extractor = InvoiceExtractor(model_path, tokenizer_path)

    # Get raw_data from command line argument
    raw_data = args.raw_data

    # Generate invoice
    generated_invoice = invoice_extractor.generate_invoice(raw_data)

    # Initialize the InvoiceData object with the generated data
    invoice_data = InvoiceData(generated_invoice)

    # Now you can use the invoice_data object to process or manipulate the extracted data further
    print("Extracted Invoice Data:")
    print(invoice_data.extracted_data)
