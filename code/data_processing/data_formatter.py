import pandas as pd
from datasets import Dataset

# TODO: We need to add the KOR prompt for NER usecase
# TODO: We need to remove the Summarization for the dialogue usecase

class DataFormatter:
    def __init__(self, dataset_path : str):
        # Initialize any attributes if needed
        self.dataset_path = dataset_path

    def prepare_dataset(self):
        data_df = pd.read_csv(self.dataset_path)
        data_df["text"] = data_df[["ocr_text", "response"]].apply(
            lambda x: "###Human: Summarize this following dialogue: " + str(x["ocr_text"]) + "\n###Assistant: " + str(x["response"]), axis=1
        )
        formatted_data = Dataset.from_pandas(data_df)
        return formatted_data
