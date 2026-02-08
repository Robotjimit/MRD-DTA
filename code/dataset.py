from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data as DATA
from tqdm import tqdm
import pandas as pd
import torch
import os


class DTA_Dataset(InMemoryDataset):
    def __init__(self, root, path, smiles_emb, target_emb, smiles_len, target_len, mode):
        super(DTA_Dataset, self).__init__(root)
        self.path = path
        self.mode = mode
        self.data = []

        df = pd.read_csv(path)
        sm_id = pd.read_csv("kiba/sm_id.csv")
        self.sm_id = {sm_id.loc[i, 'smiles']: sm_id.loc[i, 'id'] for i in range(len(sm_id))}

        self.process(df, smiles_emb, target_emb, smiles_len, target_len)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['process.pt']

    def download(self):
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self, df, smiles_emb, target_emb, smiles_len, target_len):
        if 'uniprot' in df.columns:
            data_type = "davis"
            df['id'] = df['uniprot']
        elif 'pdbid' in df.columns:
            data_type = "pdbbind"
            df['id'] = df['pdbid']
        else:
            data_type = "kiba"
            df['id'] = df['target_key']

        for i in tqdm(range(len(df))):
            sm = df.loc[i, 'compound_iso_smiles']
            seq = df.loc[i, 'target_sequence']
            label = df.loc[i, 'affinity']
            id = df.loc[i, 'uniprot']

            smiles = smiles_emb[sm]
            protein = target_emb[seq]
            smiles_lengths = smiles_len[sm]
            protein_lengths = target_len[seq]
            # s_data = torch.load(f'pdbbind/pyg_ligand/pyg_ligand/{id}.pt')
            # t_data = torch.load(f'pdbbind/pyg_protein/{id}.pt')
            if self.mode == 'casf':
                s_data = torch.load(f'./casf/pyg/{id}_ligand.pt')
                t_data = torch.load(f'./casf/pyg/{id}_protein.pt')
            else:
                # s_data = torch.load(f'./{data_type}/pyg/{sm}.pt')
                t_data = torch.load(f'./{data_type}/pyg/{id}.pt')
                if data_type == "davis":
                    s_data = torch.load(f'./{data_type}/pyg/{sm}.pt')
                else:
                    s_data = torch.load(f'./{data_type}/pyg/{self.sm_id[sm]}.pt')
            t_data.smiles = smiles
            t_data.protein = protein
            # print(protein.shape)
            if protein.shape[0] != 1000:
                # 假设 DataFrame 中有 pdbid 或 ligand_id 列
                print(f"[Warning] Protein length != {1000}, actual: {protein.shape[0]}, ID: {df.loc[i,'pdbid']}")
            t_data.smiles_lengths = smiles_lengths
            t_data.protein_lengths = protein_lengths
            t_data.y = label

            self.data.append((s_data, t_data))

        if self.pre_filter is not None:
            self.data = [data for data in self.data if self.pre_filter(data)]

        if self.pre_transform is not None:
            self.data = [self.pre_transform(data) for data in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
