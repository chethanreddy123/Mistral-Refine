import pandas as pd
from datasets import Dataset

# TODO: We need to add the KOR prompt for NER usecase
# TODO: We need to remove the Summarization for the dialogue usecase

class DataFormatter:
    def __init__(self, dataset_path : str):
        # Initialize any attributes if needed
        self.dataset_path = dataset_path

    def prepare_dataset(self):
        data_df = pd.read_csv("training_data_1.csv")
        data_df["text"] = data_df[["ocr_text", "response"]].apply(
            lambda x: """###Human: Given raw text, identify and extract following details in JSON format. Make corrections wherever possible. Convert raw text to english before starting with extraction
                1.) document type (invoice, prescription or lab report)- string
                2.) provider name - string
                3.) doctor name - string
                4.) customer name - string
                5.) age -  string
                6.) gender - string
                7.) invoice number - string
                8.) invoice date - string in dd-mm-yyyy format (fix incorrect date as per 2023)
                9.) bill total amount - string (fix as per raw text if incorrect)
                10.) doctor degree - string
                11.) doctor speciality - string
                12.) provider address - string
                13.) provider phone number - string
                14.) zipcode - list of string
                15.) city - string
                16.) state - string
                17.) bill items and their corresponding amount - JSON
                18.) Maternity benefits - boolean (true if raw text contains anything related to maternity else false)
                19.) provider category (hospital, clinic, lab or doctor) - string
                20.) medical invoice - boolean (true if raw text is a pharmacy bill)
                Wherever detail is not present return 'Not found'

                sample input = phoenix multi speciality clinic\nshop no. 3, akkale orchid, infront of renuka mata mandir, keshav nagar\nmundhwa, pune-411036\nemail : dr.mukeshmahajan@gmail.com | web : phoenixmultispecialityclinic.com\nmob: 8010360823 | 89992 89478\nname of patient :\nbeena vadgama.\nkeshar\ndate: 8//03/2020\nno :\n846\nno\ntreatment\namount\nblood test\ncbc\n5001.\ncrp\ntotal\n500\ndr. muk shmiahajan\m.s. md ., dnb.\namount in words\n041049\nstamp & signature\npagebreak\n

                sample output = {
                    "document_type": "invoice",
                    "provider": "phoenix multi speciality clinic",
                    "doctor": "mukesh mahajan",
                    "customer": "beena vadgama",
                    "age": "Not found",
                    "gender": "Not found",
                    "invoice" : "846",
                    "date" : "08-03-2023",
                    "amount" : "500",
                    "degree" : "ms,md,dnb",
                    "speciality": "Not found",
                    "address": "shop no. 3, akkale orchid, infront of renuka mata mandir, keshav nagar\nmundhwa, pune-411036",
                    "phone": "8010360823",
                    "zipcode" : ["411036"],
                    "city": "Pune",
                    "state": "Maharashtra",
                    "bill_items": { 
                        "Blood Test": "",
                        "CBC": "500",
                        "CRP" : ""
                    },
                    "maternity_benefits": false,
                    "provider_category": "Clinic",
                    "is_medical_invoice": false
                }
                Raw text: """ + str(x["ocr_text"]) + "\n###Assistant: " + str(x["response"]), axis=1
        )
        formatted_data = Dataset.from_pandas(data_df)
        return formatted_data