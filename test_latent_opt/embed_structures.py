# test_latent_opt/embed_petase_strucures.py
'''
* Author: Evan Komp
* Created: 6/25/2024
* Company: Bottle Institute @ National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT

Embed input structures with ESM3's structure VAE
'''
import os

import pandas as pd
import numpy as np
import datasets
import torch
import tqdm

from esm.utils.structure.protein_chain import ProteinChain
from esm.pretrained import load_local_model
from esm.utils.constants.models import ESM3_STRUCTURE_ENCODER_V0
from esm.models.esm3 import ESM3
from esm.sdk.api import (
    ESMProtein,
    ESMProteinTensor,
)
# from huggingface_hub import login

import dvc.api

PARAMS = dvc.api.params_show()

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, filemode='w', filename=os.path.join('logs', 'embed_structures.log'))

def main():

    # read label data
    label_data = pd.read_csv(os.path.join('data', 'label_data.csv'))
    assert "target" in label_data.columns, "label_data.csv must contain a 'target' column"
    assert "sequence" in label_data.columns, "label_data.csv must contain a 'sequence' column"
    assert "id" in label_data.columns, "label_data.csv must contain a 'id' column"

    # get the pdb filenames
    structures_available = os.listdir(os.path.join('data', 'structures'))
    structure_file_map = {}
    for _, row in label_data.iterrows():
        if row['id'] + '.pdb' not in structures_available:
            logger.warning(f"{row['id']} does not have a structure file")
            continue
        else:
            structure_file_map[row['id']] = os.path.join('data', 'structures', row['id'] + '.pdb')

            # check that the structure matches the sequence
            p = ProteinChain.from_pdb(structure_file_map[row['id']])
            if p.sequence != row['sequence']:
                if 'X' in row['sequence']:
                    logger.warning(f"{row['id']} contains unknown AAs, dropping.")
                    del structure_file_map[row['id']]
                else:
                    print(p.sequence, row['sequence'])
                    raise ValueError(f"Sequence in label_data.csv does not match sequence in {structure_file_map[row['id']]}")
    logger.info(f"Found {len(structure_file_map)} structures")

    # filter the labele data by only good structures
    good_structures = label_data['id'].apply(lambda i: i in structure_file_map)
    label_data = label_data[good_structures]
    label_data.to_csv(os.path.join('data', 'clean_data.csv'))
    
    # load the model
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    if PARAMS['esm']['embeddings'] == 'structure_vae':
    
        encoder = load_local_model(ESM3_STRUCTURE_ENCODER_V0, device=device)
        logger.info(f"Embedding on device {device}")

        # build a dataset by looping through sequences
        def generator():
            for _, row in tqdm.tqdm(list(label_data.iterrows())):
                # if row['id'] not in structure_file_map:
                #     continue
                p = ProteinChain.from_pdb(structure_file_map[row['id']])
                coordinates, plddt, residue_index = p.to_structure_encoder_inputs()
                coordinates = coordinates.to(device)
                residue_index = residue_index.to(device)
                z, tokens = encoder.encode(coordinates, residue_index=residue_index)
                # convert to base dtypes
                z = z.cpu().detach().numpy().astype(float)
                tokens = tokens.cpu().detach().numpy().astype(int)

                # remove tensors from cuda memory
                del coordinates
                del residue_index

                yield {
                    'id': row['id'],
                    'sequence': row['sequence'],
                    'z': z,
                    'tokens': tokens,
                    'y': float(row['target']),
                    'split': int(row['cross_val_split'])
                }
    elif PARAMS['esm']['embeddings'] == 'esm':
        # here we use the embeddings from the whole transformer
        model = ESM3.from_pretrained("esm3_sm_open_v1", device=torch.device(device))

        def generator():
            for _, row in tqdm.tqdm(list(label_data.iterrows())):
                # if row['id'] not in structure_file_map:
                #     continue
                p = ProteinChain.from_pdb(structure_file_map[row['id']])
                p = ESMProtein.from_protein_chain(p, with_annotations=False)
                prompt = model.encode(p)
                outs = model.forward(

                    sequence_tokens=prompt.sequence.unsqueeze(0),
                    structure_tokens=prompt.structure.unsqueeze(0),
                    # ss8_tokens=prompt.secondary_structure.unsqueeze(0),
                    # sasa_tokens=prompt.sasa.unsqueeze(0),
                    structure_coords=prompt.coordinates.unsqueeze(0),
                )
                embeddings = outs.embeddings.cpu().detach().numpy().astype(float)
                tokens = None
                print(f"Id {row['id']} complete")
                yield {
                    'id': row['id'],
                    'sequence': row['sequence'],
                    'z': embeddings,
                    'tokens': tokens,
                    'y': float(row['target']),
                    'split': int(row['cross_val_split'])}
                                     

    ds = datasets.Dataset.from_generator(generator)

    # convert to data dict over splits
    unique_splits = [str(i) for i in list(set(ds['split']))]
    data_dict = {}
    for s in unique_splits:
        data_dict[s] = ds.filter(lambda x: x['split'] == int(s))
    ds = datasets.DatasetDict(data_dict)
    logger.info(f"Split data into {ds}")

    # save to disk
    ds.save_to_disk(os.path.join('data', 'embedded_data'))

if __name__ == '__main__':
    main()
