# Source-Aware Training Enables Knowledge Attribution in Language Models

📝 [paper](https://arxiv.org/abs/2404.01019) 🤗 [Data(BioCite)](https://huggingface.co/datasets/mkhalifa/BioCite/tree/main/qa)

We explore **Source-aware Training** to enable LLMs to cite their pretraining data. Source-aware training involves (i) training the LLM to associate unique source document identifiers with the knowledge in each document, followed by (ii) an instruction-tuning to teach the LLM to cite a supporting pretraining source when prompted. We demonstrate that our training recipe can enable faithful attribution to the pretraining data without a substantial impact on the model's quality compared to standard pretraining. Our results also highlight the importance of data augmentation in achieving attribution.

<p align="center">
<img src="https://github.com/mukhal/intrinsic-source-citation/assets/5109053/9f4d582e-5b92-4715-88ab-97d20f82ee04" alt="image" width="500" height="250">
</p>

## Pretraining Corpus
We conduct our experiments on **BioCite** a synthetic corpus of fake biography information. Each document in BioCite is constructed by sampling multiple facts from different biographies. Each document ID is constructed as a concatenation of 3-letter prefix of each last name in the document. 
<p align="center">
<img src="https://github.com/mukhal/intrinsic-source-citation/assets/5109053/86beaa3f-088a-4f21-bed5-de2dfa319e5e" alt="image" width="600" height="230">
</p>

You can access our synthetic pretraining dataset on 🤗 [here](https://huggingface.co/datasets/mkhalifa/BioCite/tree/main/qa)


## Code
Coming Soon
