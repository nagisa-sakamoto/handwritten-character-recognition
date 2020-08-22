import Levenshtein
from data.dataset import ImageDataset, ImageDataLoader
import torch.nn.functional as F
import re
from tqdm import tqdm
import json
import os

def evaluate(model, test_data, targets, output_dir, batch_size=1, verbose=False):
    model.eval()
    test_loader = ImageDataLoader(ImageDataset(test_data, targets), batch_size=batch_size)
    total_distance = 0
    total_length = 0
    results = []
    consecutive_char = re.compile(r'(.)\1+')
    for batch_idx, (input_data, reference, input_lengths, reference_lengths) in tqdm(enumerate(test_loader)): 
        split_refs = []
        offset = 0
        for size in reference_lengths:
            split_ref = reference[offset:offset + size].to('cpu').detach().numpy().copy()
            split_refs.append(split_ref)
            offset += size

        output = model(input_data, input_lengths)
        output = F.softmax(output, dim=-1)
        output = output.to('cpu').detach().numpy().copy()
        max_index = output.argmax(axis=2)

        for i in range(batch_size):
            pred = model.get_target_seq(max_index[i])
            pred = consecutive_char.sub('\\1', pred).replace('_', '')
            ref = model.get_target_seq(split_refs[i])
            if pred:
                distance = Levenshtein.distance(pred, ref)
            else:
                distance = len(ref)
            total_distance += distance
            total_length += len(ref)
            results.append({'ref':ref, 'pred':pred, 'accuracy': 1-(distance/len(ref))})
            if verbose:
                print(f'Ref : {ref}')
                print(f'Pred: {pred}')
                print()
    print(f'Testset accuracy: {1-(total_distance/total_length)}')
    results.append({'Ave. of accuray':1-(total_distance/total_length)})
    output_file = os.path.join(output_dir, 'test_result.json')
    with open(output_file, 'w') as fo:
        json.dump(results, fo, ensure_ascii=False, indent=4)
    

