from tqdm import tqdm
import logging, sys, argparse
import datasets
import random
from nltk.tokenize import sent_tokenize
from pathlib import Path

from datasets import disable_caching 
disable_caching()


def create_path(path):
    Path(path).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    logging.basicConfig(
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.ERROR,
    )
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/net/nfs/allennlp/muhammadk/processed-data/wikipedia/actual_date_random_tokens_1M/')
    parser.add_argument('--out_path', type=str, default=None)
    parser.add_argument('--augment_type', type=str, default='permute', choices=['permute'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_sample_per_doc', type=int, default=4)
    parser.add_argument('--keep_first_sentence', action='store_true')

    args = parser.parse_args()
    
    ## load only the first args.n samples
    data = datasets.load_from_disk(args.data_path)


    ### augment the dataset 
    if args.augment_type == "permute":
        #### shuffle the sentences in each doc args.n_sample_per_doc times

        #### create a new dataset
        def get_permuted_doc(doc):
            sents = sent_tokenize(doc)
            if args.keep_first_sentence:
                remaining_sents = sents[1:]
                random.shuffle(remaining_sents)
                sents = [sents[0]] + remaining_sents
            else:
                random.shuffle(sents)
            
            return " ".join(sents)
        
        def get_permuted_obj(obj):
            obj['text'] = get_permuted_doc(obj['text'])
            return obj
        
        all_datasets = [data]

        for _ in tqdm(range(args.n_sample_per_doc), desc="Augmenting data"):
            all_datasets.append(data.map(get_permuted_obj))
        
        aug_data = datasets.concatenate_datasets(all_datasets)
        print("Augmented dataset size = {}".format(len(aug_data)))

    if args.out_path is not None:
        ### create the out path if it doesn't exist
        create_path(args.out_path)
        aug_data.save_to_disk(args.out_path)



