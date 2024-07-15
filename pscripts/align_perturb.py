
import datasets
from tqdm import tqdm

path_to_perturb = '/net/nfs/allennlp/muhammadk/processed-data/wikipedia/actual_date_random_tokens_1M/perturbed_10000/'

path_to_new_data = '/net/nfs/allennlp/muhammadk/processed-data/wikipedia/page_title_para_id_10M'


## read perturbed data and create a dict mapping 'text' to 'perturbed'
perturbed_data = datasets.load_from_disk(path_to_perturb)

perturbed_dict = {}
for i in range(len(perturbed_data)):
    text = perturbed_data[i]['text']
    perturbed_dict[perturbed_data[i]['text']] = perturbed_data[i]['perturb']


## read new data and create a dict mapping 'text' to 'url'
new_data = datasets.load_from_disk(path_to_new_data)
new_data = new_data.select(range(50000))

new_dict = {}
for i in tqdm(range(len(new_data))):
    text = new_data[i]['text']
    url = new_data[i]['url']
    new_dict[text] = url

## create a new perturbed dataset with the new url and perturbed text

new_perturbed_data = {
    'text': [],
    'perturb': [],
    'url': []
}

for text in perturbed_dict:
    perturbed_text = perturbed_dict[text]
    if text not in new_dict:
        print("text not found in new data: ", text)
        continue

    url = new_dict[text]
    new_perturbed_data['text'].append(text)
    new_perturbed_data['perturb'].append(perturbed_text)
    new_perturbed_data['url'].append(url)

print("got {} perturbed examples".format(len(new_perturbed_data['text'])))
## save the new perturbed data
output_path = path_to_new_data + '/perturbed_10000'

## convert new_perturbed_data to list of dicts 
new_perturbed_data = datasets.Dataset.from_dict(new_perturbed_data)
new_perturbed_data.save_to_disk(output_path)
