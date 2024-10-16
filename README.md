# Privacy_Different_Task

Due to high volumn of our dataset, we only provide part of images from the full dataset. However, we provide the full database and text data from our full dataset.


## Memory Set
In the folder ```Memory Set```, we provide our memory set that be used to inject private information into MLLMs by fine-tuning. ```database_full.xlsx``` contains the database information and ```final_data_full.json``` contains the labels for fine-tuning MLLMs. We also provide two sample codes that using our dataset to do supervised fine-tune with Idefics2 and Xgen3 on our ```Memory Set```.

## Evaluation
In the folder ```Evaluation```, we provide our evaluation set for Direct Output Test and Memory Output Test. In the folder ```Real Image```, we provide examples for our real-world images and its corresponding database and text data. Again we provide codes that uses Xgen3 to generate responses to 5 different tasks with the evaluation set for Direct Output Test and Memory Output Test, respectively.

