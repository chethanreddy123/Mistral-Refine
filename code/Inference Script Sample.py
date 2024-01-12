#!/usr/bin/env python
# coding: utf-8

# In[2]:


from peft import AutoPeftModelForCausalLM,TaskType
from transformers import GenerationConfig
from transformers import AutoTokenizer
import torch
tokenizer = AutoTokenizer.from_pretrained("/content/Mistral/code/mistral-finetuned/checkpoint-50")
prompt="""
###Human: Given raw text, identify and extract following details in JSON format. Make corrections wherever possible. Convert raw text to english before starting with extraction
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
                Raw text: {raw_text}
###Assistant: """

prompt_template=prompt.format(raw_text="")
inputs = tokenizer(prompt_template, return_tensors="pt").to("cuda")
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "checkpoint-300",
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map="cuda"
    )

generation_config = GenerationConfig(
    do_sample=True,
    top_k=1,
    temperature=0.1,
    max_new_tokens=25,
    pad_token_id=tokenizer.eos_token_id
)

import time
st_time = time.time()
outputs = model.generate(**inputs, generation_config=generation_config)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
print(time.time()-st_time)


# In[ ]:



