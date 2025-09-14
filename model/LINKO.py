from typing import Tuple, List, Dict, Optional
import functools
import torch
from torch import nn
from pyhealth.datasets import BaseEHRDataset
from pyhealth.models import BaseModel, TransformerLayer
from pyhealth.tokenizer import Tokenizer
from pyhealth.medcode import InnerMap
import pandas as pd
import torch.nn.functional as F
import numpy as np
from collections import Counter
from tqdm import tqdm
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, HypergraphConv
from openai import OpenAI
import random



class Mega(BaseModel):
    def __init__(self, dataset: List[BaseEHRDataset], train_dataset: List[BaseEHRDataset], feature_keys: List[str], label_key: str, mode: str,
                 embedding_dim=128, dropout = 0.5, nheads=1, nlayers=1,
                 G_dropout = 0.5, n_G_heads=1, n_G_layers=1,threshold3=0.00, threshold2=0.02, threshold1=0.12,
                 llm_model = 'text-embedding-3-small', gpt_embd_path='../saved_files/gpt_code_emb/tx-emb-3-small/',
                 n_hap_layers=1, n_hap_heads=1, hap_dropout = 0.5,
                 ds_size_ratio='', device='cuda', seed=None, **kwargs):
        super().__init__(dataset, feature_keys, label_key, mode)

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.device1 = device
        self.ds_size_ratio = ds_size_ratio
        self.train_dataset = train_dataset
        # Set seed for CUDA operations
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        # Any BaseModel should have these attributes, as functions like add_feature_transform_layer uses them
        self.feat_tokenizers1 = {}
        self.feat_tokenizers2 = {}
        self.feat_tokenizers3 = {}
        self.dataset = dataset

        self.embeddings1 = nn.ModuleDict()
        self.embeddings2 = nn.ModuleDict()
        self.embeddings3 = nn.ModuleDict()
        # self.embeddings1_gpt = nn.ModuleDict()
        # self.embeddings2_gpt = nn.ModuleDict()
        # self.embeddings3_gpt = nn.ModuleDict()

        self.linear_layers = nn.ModuleDict()
        self.label_tokenizer = self.get_label_tokenizer()
        self.embedding_dim = embedding_dim

        self.threshold1 = threshold1
        self.threshold2 = threshold2
        self.threshold3 = threshold3

        self._ontology_tables()

        #importing/generating conditional_prob_matrix
        if os.path.isfile(f'../saved_files/conditional_prob_matrix{ds_size_ratio}.csv'):
            # Load the DataFrame if the file exists
            self.conditional_prob_matrix_l3 = pd.read_csv(f'../saved_files/conditional_prob_matrix{ds_size_ratio}.csv')
            self.conditional_prob_matrix_l3.drop(['Unnamed: 0'], axis=1, inplace=True)
            self.conditional_prob_matrix_l3.index = self.conditional_prob_matrix_l3.columns
            print(f"conditional_prob_matrix loaded successfully.")
        else:
            print('file not found, generating co-occurence matrix ...')
            self.get_co_occurrence()
            print(f"conditional_prob_matrix generated successfully.")

        # importing/generating conditional_prob_matrix for parents
        if os.path.isfile(f'../saved_files/conditional_prob_matrix1{ds_size_ratio}.csv') and os.path.isfile(
                '../saved_files/conditional_prob_matrix2{ds_size_ratio}.csv'):
            # Load the DataFrame if the file exists
            self.conditional_prob_matrix_l1 = pd.read_csv(f'../saved_files/conditional_prob_matrix1{ds_size_ratio}.csv')
            self.conditional_prob_matrix_l1.drop(['Unnamed: 0'], axis=1, inplace=True)
            self.conditional_prob_matrix_l1.index = self.conditional_prob_matrix_l1.columns
            self.conditional_prob_matrix_l2 = pd.read_csv(f'../saved_files/conditional_prob_matrix2{ds_size_ratio}.csv')
            self.conditional_prob_matrix_l2.drop(['Unnamed: 0'], axis=1, inplace=True)
            self.conditional_prob_matrix_l2.index = self.conditional_prob_matrix_l2.columns
            print(f"conditional_prob_matrix for parents loaded successfully.")
        else:
            print('files not found, generating co-occurence matrix for parents...')
            self.get_co_occurrence_for_parents()
            print(f"conditional_prob_matrix for parents generated successfully.")



        #llm
        api_key = "***************************************"
        self.client = OpenAI(api_key=api_key)
        self.llm_model = llm_model
        self.gpt_embd_path = gpt_embd_path

        if os.path.isfile(self.gpt_embd_path+ f'dx1_gpt_emb{ds_size_ratio}.npy'):

            self.dx1_gpt_emb = torch.tensor(np.load(self.gpt_embd_path + f'dx1_gpt_emb{ds_size_ratio}.npy'), dtype=torch.float32)
            self.dx2_gpt_emb = torch.tensor(np.load(self.gpt_embd_path + f'dx2_gpt_emb{ds_size_ratio}.npy'), dtype=torch.float32)
            self.dx3_gpt_emb = torch.tensor(np.load(self.gpt_embd_path + f'dx3_gpt_emb{ds_size_ratio}.npy'), dtype=torch.float32)

            self.rx1_gpt_emb = torch.tensor(np.load(self.gpt_embd_path + f'rx1_gpt_emb{ds_size_ratio}.npy'), dtype=torch.float32)
            self.rx2_gpt_emb = torch.tensor(np.load(self.gpt_embd_path + f'rx2_gpt_emb{ds_size_ratio}.npy'), dtype=torch.float32)
            self.rx3_gpt_emb = torch.tensor(np.load(self.gpt_embd_path + f'rx3_gpt_emb{ds_size_ratio}.npy'), dtype=torch.float32)

            self.px1_gpt_emb = torch.tensor(np.load(self.gpt_embd_path + f'px1_gpt_emb{ds_size_ratio}.npy'), dtype=torch.float32)
            self.px2_gpt_emb = torch.tensor(np.load(self.gpt_embd_path + f'px2_gpt_emb{ds_size_ratio}.npy'), dtype=torch.float32)
            self.px3_gpt_emb = torch.tensor(np.load(self.gpt_embd_path + f'px3_gpt_emb{ds_size_ratio}.npy'), dtype=torch.float32)

        else:
            self.creat_llm_emb()


        #print(self.embeddings1['conditions'].weight[:2].device)
        #print(self.dx1_gpt_emb.device)
        self.GPT_Embedding1_weights = {'conditions': torch.cat([torch.randn(2, embedding_dim), self.dx1_gpt_emb], dim=0),
                               'drugs': torch.cat([torch.randn(2, embedding_dim), self.rx1_gpt_emb], dim=0),
                               'procedures': torch.cat([torch.randn(2, embedding_dim), self.px1_gpt_emb], dim=0)}

        self.GPT_Embedding2_weights = {'conditions': torch.cat([torch.randn(2, embedding_dim), self.dx2_gpt_emb], dim=0),
                               'drugs': torch.cat([torch.randn(2, embedding_dim), self.rx2_gpt_emb], dim=0),
                               'procedures': torch.cat([torch.randn(2, embedding_dim), self.px2_gpt_emb], dim=0)}

        self.GPT_Embedding3_weights = {'conditions': torch.cat([torch.randn(2, embedding_dim), self.dx3_gpt_emb], dim=0),
                               'drugs': torch.cat([torch.randn(2, embedding_dim), self.rx3_gpt_emb], dim=0),
                               'procedures': torch.cat([torch.randn(2, embedding_dim), self.px3_gpt_emb], dim=0)}



        self.code_attention_energy = nn.ModuleDict({key:nn.Linear(2*embedding_dim,1) for key in feature_keys})
        self.softmax = nn.Softmax(dim=-2)



        # self.add_feature_transform_layer will create a transformation layer for each feature
        for feature_key in self.feature_keys:
            input_info = self.dataset.input_info[feature_key]
            self._my_add_feature_transform_layer(
                feature_key, input_info, special_tokens=["<pad>", "<unk>"]
            )

        self._token_id_ontology_tables()

        self.get_hyper_edges()
        self.n_G_layers = n_G_layers
        self.HyperG3 = HypergraphConv(embedding_dim, embedding_dim, use_attention=True, attention_mode= 'node' , heads=n_G_heads, concat=False, dropout=G_dropout)

        self.adj1 = torch.tensor((self.conditional_prob_matrix_l1 > self.threshold1).astype('int32').values)
        self.adj1.fill_diagonal_(0)
        # adj = torch.triu(adj, diagonal=1)  # Consider only upper triangular part to avoid duplicate edges
        # adj = adj + adj.T  # Make the adjacency matrix symmetric
        # Convert the adjacency matrix to edge_index
        self.edge_index1 = self.adj1.nonzero(as_tuple=False).t().contiguous().to(device)
        self.GATLayer1 = GATConv(embedding_dim, embedding_dim, heads=n_G_heads, concat=False, dropout=G_dropout)

        self.adj2 = torch.tensor((self.conditional_prob_matrix_l2 > self.threshold2).astype('int32').values)
        self.adj2.fill_diagonal_(0)
        # adj = torch.triu(adj, diagonal=1)  # Consider only upper triangular part to avoid duplicate edges
        # adj = adj + adj.T  # Make the adjacency matrix symmetric
        # Convert the adjacency matrix to edge_index
        self.edge_index2 = self.adj2.nonzero(as_tuple=False).t().contiguous().to(device)
        self.GATLayer2 = GATConv(embedding_dim, embedding_dim, heads=n_G_heads, concat=False, dropout=G_dropout)

        if n_G_layers == 2:
            self.HyperG3_2 = HypergraphConv(embedding_dim, embedding_dim, use_attention=True, attention_mode= 'node' , heads=n_G_heads, concat=False, dropout=G_dropout)
            self.GATLayer2_2 = GATConv(embedding_dim, embedding_dim, heads=n_G_heads, concat=False, dropout=G_dropout)
            self.GATLayer1_2 = GATConv(embedding_dim, embedding_dim, heads=n_G_heads, concat=False, dropout=G_dropout)


        #bot_up_hap GAT layers
        self.n_hap_layers = n_hap_layers
        self.n_hap_heads = n_hap_heads
        self.hap_dropout = hap_dropout
        self.GATLayer_hap1 = nn.ModuleDict()
        self.GATLayer_hap2 = nn.ModuleDict()
        if n_hap_layers == 2:
            self.GATLayer_hap1_2 = nn.ModuleDict()
            self.GATLayer_hap2_2 = nn.ModuleDict()

        for feature_key in self.feature_keys:
            self.GATLayer_hap1[feature_key] = GATConv(embedding_dim, embedding_dim, heads=n_hap_heads, concat=False, dropout=hap_dropout)
            self.GATLayer_hap2[feature_key] = GATConv(embedding_dim, embedding_dim, heads=n_hap_heads, concat=False, dropout=hap_dropout)
            if n_hap_layers==2:
                self.GATLayer_hap1_2[feature_key] = GATConv(embedding_dim, embedding_dim, heads=n_hap_heads, concat=False,
                                                          dropout=hap_dropout)
                self.GATLayer_hap2_2[feature_key] = GATConv(embedding_dim, embedding_dim, heads=n_hap_heads, concat=False,
                                                          dropout=hap_dropout)


        #bot_up_hap
        self.edge_list_dx2 = self.get_edge_list_for_hap(table=self.dx_table, parent_level ='l2').contiguous().to(device)
        self.edge_list_dx1 = self.get_edge_list_for_hap(table=self.dx_table, parent_level='l1').contiguous().to(device)

        self.edge_list_rx2 = self.get_edge_list_for_hap(table=self.rx_table, parent_level ='l2').contiguous().to(device)
        self.edge_list_rx1 = self.get_edge_list_for_hap(table=self.rx_table, parent_level='l1').contiguous().to(device)

        self.edge_list_px2 = self.get_edge_list_for_hap(table=self.px_table, parent_level ='l2').contiguous().to(device)
        self.edge_list_px1 = self.get_edge_list_for_hap(table=self.px_table, parent_level='l1').contiguous().to(device)


        self.transformer = nn.ModuleDict()
        for feature_key in feature_keys:
            self.transformer[feature_key] = TransformerLayer(
                feature_size=embedding_dim, heads = nheads, dropout = dropout, num_layers = nlayers, **kwargs
            )

        # final output layer
        output_size = self.get_output_size(self.label_tokenizer)
        self.fc = nn.Linear(len(self.feature_keys) * embedding_dim, output_size)

        self.linear = nn.Linear(embedding_dim*2, embedding_dim)
        self.dropout = nn.Dropout(dropout)


    def _ontology_tables(self):
        icd9 = InnerMap.load("ICD9CM")
        icd9proc = InnerMap.load("ICD9PROC")
        atc = InnerMap.load("ATC")

        dx_parents1, dx_parents2 = [], []
        dx_parents3 = self.dataset.get_all_tokens('conditions')
        for code in dx_parents3:
            dx_parents = icd9.get_ancestors(code)
            dx_parents1.append(dx_parents[-1])
            dx_parents2.append(dx_parents[-2])

        self.dx_table = pd.DataFrame()
        self.dx_table['l1'] = dx_parents1
        self.dx_table['l2'] = dx_parents2
        self.dx_table['l3'] = dx_parents3


        rx_parents1, rx_parents2 = [], []
        rx_parents3 = self.dataset.get_all_tokens('drugs')
        for code in rx_parents3:
            rx_parents = atc.get_ancestors(code)
            rx_parents1.append(rx_parents[-1])
            rx_parents2.append(rx_parents[-2])

        self.rx_table = pd.DataFrame()
        self.rx_table['l1'] = rx_parents1
        self.rx_table['l2'] = rx_parents2
        self.rx_table['l3'] = rx_parents3

        px_parents1, px_parents2 = [], []
        px_parents3 = self.dataset.get_all_tokens('procedures')
        for code in px_parents3:
            px_parents = icd9proc.get_ancestors(code)
            px_parents1.append(px_parents[-1])
            px_parents2.append(px_parents[-2])

        self.px_table = pd.DataFrame()
        self.px_table['l1'] = px_parents1
        self.px_table['l2'] = px_parents2
        self.px_table['l3'] = px_parents3

    def _token_id_ontology_tables(self):
        self.dx_table_token_id = pd.DataFrame()
        self.dx_table_token_id['l1'] = [0, 1] + self.feat_tokenizers1['conditions'].convert_tokens_to_indices(self.dx_table['l1'].values.tolist())
        self.dx_table_token_id['l2'] = [0, 1] + self.feat_tokenizers2['conditions'].convert_tokens_to_indices(self.dx_table['l2'].values.tolist())
        self.dx_table_token_id['l3'] = [0, 1] + self.feat_tokenizers3['conditions'].convert_tokens_to_indices(self.dx_table['l3'].values.tolist())

        self.px_table_token_id = pd.DataFrame()
        self.px_table_token_id['l1'] = [0, 1] + self.feat_tokenizers1['procedures'].convert_tokens_to_indices(self.px_table['l1'].values.tolist())
        self.px_table_token_id['l2'] = [0, 1] + self.feat_tokenizers2['procedures'].convert_tokens_to_indices(self.px_table['l2'].values.tolist())
        self.px_table_token_id['l3'] = [0, 1] + self.feat_tokenizers3['procedures'].convert_tokens_to_indices(self.px_table['l3'].values.tolist())

        self.rx_table_token_id = pd.DataFrame()
        self.rx_table_token_id['l1'] = [0, 1] + self.feat_tokenizers1['drugs'].convert_tokens_to_indices(self.rx_table['l1'].values.tolist())
        self.rx_table_token_id['l2'] = [0, 1] + self.feat_tokenizers2['drugs'].convert_tokens_to_indices(self.rx_table['l2'].values.tolist())
        self.rx_table_token_id['l3'] = [0, 1] + self.feat_tokenizers3['drugs'].convert_tokens_to_indices(self.rx_table['l3'].values.tolist())


    def _my_add_feature_transform_layer(self, feature_key: str, info, special_tokens=None):

        if feature_key=='conditions':
            tokens1 = self.dx_table['l1'].unique().tolist()
            tokens2 = self.dx_table['l2'].unique().tolist()
            tokens3 = self.dx_table['l3'].unique().tolist()
        elif feature_key=='procedures':
            tokens1 = self.px_table['l1'].unique().tolist()
            tokens2 = self.px_table['l2'].unique().tolist()
            tokens3 = self.px_table['l3'].unique().tolist()
        elif feature_key=='drugs':
            tokens1 = self.rx_table['l1'].unique().tolist()
            tokens2 = self.rx_table['l2'].unique().tolist()
            tokens3 = self.rx_table['l3'].unique().tolist()


        if info["type"] == str:
            # feature tokenizer
            if special_tokens is None:
                special_tokens = ["<pad>", "<unk>"]
            tokenizer1 = Tokenizer(
                #tokens=self.datasets[0].get_all_tokens(key=feature_key),
                tokens=tokens1,
                special_tokens=special_tokens,
            )
            tokenizer2 = Tokenizer(
                tokens=tokens2,
                special_tokens=special_tokens,
            )
            tokenizer3 = Tokenizer(
                tokens=tokens3,
                special_tokens=special_tokens,
            )
            self.feat_tokenizers1[feature_key] = tokenizer1
            self.feat_tokenizers2[feature_key] = tokenizer2
            self.feat_tokenizers3[feature_key] = tokenizer3
            # feature embedding

            self.embeddings1[feature_key] = nn.Embedding(
                tokenizer1.get_vocabulary_size(),
                self.embedding_dim,
                padding_idx=tokenizer1.get_padding_index(),
            )
            self.embeddings2[feature_key] = nn.Embedding(
                tokenizer2.get_vocabulary_size(),
                self.embedding_dim,
                padding_idx=tokenizer2.get_padding_index(),
            )
            self.embeddings3[feature_key] = nn.Embedding(
                tokenizer3.get_vocabulary_size(),
                self.embedding_dim,
                padding_idx=tokenizer3.get_padding_index(),
            )

            # self.embeddings1_gpt[feature_key] = nn.Embedding.from_pretrained(self.GPT_Embedding1_weights[feature_key], freeze=False)
            # self.embeddings2_gpt[feature_key] = nn.Embedding.from_pretrained(self.GPT_Embedding2_weights[feature_key], freeze=False)
            # self.embeddings3_gpt[feature_key] = nn.Embedding.from_pretrained(self.GPT_Embedding3_weights[feature_key], freeze=False)


            self.embeddings1[feature_key] = nn.Embedding.from_pretrained(self.GPT_Embedding1_weights[feature_key], freeze=False)
            self.embeddings2[feature_key] = nn.Embedding.from_pretrained(self.GPT_Embedding2_weights[feature_key], freeze=False)
            self.embeddings3[feature_key] = nn.Embedding.from_pretrained(self.GPT_Embedding3_weights[feature_key], freeze=False)

        elif info["type"] in [float, int]:
            self.linear_layers[feature_key] = nn.Linear(info["len"], self.embedding_dim)
        else:
            raise ValueError("Unsupported feature type: {}".format(info["type"]))

    def _get_gpt_embedding(self, text, model="text-embedding-3-small", dimensions=None):
        # avialable models: "text-embedding-3-large" ,  "text-embedding-3-small", "text-embedding-ada-002"
        return self.client.embeddings.create(input=[text], model=model, dimensions=dimensions).data[0].embedding

    def _get_llm_emb(self, codes, code_type, level):

        if code_type=='dx':
            onto = InnerMap.load("ICD9CM")
            code_type = 'ICD-9 diagnosis'
            #code_type = 'ICD-9 dx'
        elif code_type=='rx':
            onto = InnerMap.load("ATC")
            code_type = 'ATC prescription'
            #code_type = 'ATC rx'
        else:
            onto = InnerMap.load("ICD9PROC")
            code_type = 'ICD-9 procedure'
            #code_type = 'ICD-9 px'

        gpt_code_emb_lst = []
        for code in codes:
            if level==3:
                parents = onto.get_ancestors(code)
                parent_code1, parent_code2 = parents[-1], parents[-2]

                code_concept_name = onto.lookup(code)
                parent1_concept_name = onto.lookup(parent_code1)
                parent2_concept_name = onto.lookup(parent_code2)


                text = f"{code_type} code {code} represents {code_concept_name}. It is a specific medical concept under the broader categories of {parent_code2} ({parent2_concept_name}) and {parent_code1} ({parent1_concept_name})."

            elif level==2:
                parents = onto.get_ancestors(code)
                parent_code1 = parents[-1]

                code_concept_name = onto.lookup(code)
                parent1_concept_name = onto.lookup(parent_code1)

                text = f"{code_type} code {code} represents {code_concept_name}. It is a specific medical concept under the broader categoriey of {parent_code1} ({parent1_concept_name})."

            elif level==1:
                code_concept_name = onto.lookup(code)
                text = f'{code_type} code {code} represents {code_concept_name}. which a general medical concept'
                # text = f'{code_type} code:{code_concept_name}'

            else:
                "Error in level input!"

            print(text)
            gpt_code_emb = self._get_gpt_embedding(text, model=self.llm_model, dimensions=self.embedding_dim)
            gpt_code_emb_lst.append(gpt_code_emb)

        return np.array(gpt_code_emb_lst)

    def creat_llm_emb(self):

        print(f'creating llm embedding, llm_model:{self.llm_model}')
        dx1 = self.dx_table['l1'].unique().tolist()
        dx2 = self.dx_table['l2'].unique().tolist()
        dx3 = self.dx_table['l3'].unique().tolist()
        dx1_gpt_emb = self._get_llm_emb(codes=dx1, code_type='dx')
        np.save(self.gpt_embd_path + f'dx1_gpt_emb{self.ds_size_ratio}.npy', dx1_gpt_emb)
        self.dx1_gpt_emb = torch.tensor(dx1_gpt_emb, dtype=torch.float64).to(self.device)


        dx2_gpt_emb = self._get_llm_emb(codes=dx2, code_type='dx')
        np.save(self.gpt_embd_path + f'dx2_gpt_emb{self.ds_size_ratio}.npy', dx2_gpt_emb)
        self.dx2_gpt_emb = torch.tensor(dx2_gpt_emb, dtype=torch.float64).to(self.device)

        dx3_gpt_emb = self._get_llm_emb(codes=dx3, code_type='dx')
        np.save(self.gpt_embd_path + f'dx3_gpt_emb{self.ds_size_ratio}.npy', dx3_gpt_emb)
        self.dx3_gpt_emb = torch.tensor(dx3_gpt_emb, dtype=torch.float64).to(self.device)

        rx1 = self.rx_table['l1'].unique().tolist()
        rx2 = self.rx_table['l2'].unique().tolist()
        rx3 = self.rx_table['l3'].unique().tolist()

        rx1_gpt_emb = self._get_llm_emb(codes=rx1, code_type='rx')
        np.save(self.gpt_embd_path + f'rx1_gpt_emb{self.ds_size_ratio}.npy', rx1_gpt_emb)
        self.rx1_gpt_emb = torch.tensor(rx1_gpt_emb, dtype=torch.float64).to(self.device)

        rx2_gpt_emb = self._get_llm_emb(codes=rx2, code_type='rx')
        np.save(self.gpt_embd_path + f'rx2_gpt_emb{self.ds_size_ratio}.npy', rx2_gpt_emb)
        self.rx2_gpt_emb = torch.tensor(rx2_gpt_emb, dtype=torch.float64).to(self.device)

        rx3_gpt_emb = self._get_llm_emb(codes=rx3, code_type='rx')
        np.save(self.gpt_embd_path + f'rx3_gpt_emb{self.ds_size_ratio}.npy', rx3_gpt_emb)
        self.rx3_gpt_emb = torch.tensor(rx3_gpt_emb, dtype=torch.float64).to(self.device)

        px1 = self.px_table['l1'].unique().tolist()
        px2 = self.px_table['l2'].unique().tolist()
        px3 = self.px_table['l3'].unique().tolist()

        px1_gpt_emb = self._get_llm_emb(codes=px1, code_type='px')
        np.save(self.gpt_embd_path+ f'px1_gpt_emb{self.ds_size_ratio}.npy', px1_gpt_emb)
        self.px1_gpt_emb = torch.tensor(px1_gpt_emb, dtype=torch.float64).to(self.device)

        px2_gpt_emb = self._get_llm_emb(codes=px2, code_type='px')
        np.save(self.gpt_embd_path+ f'px2_gpt_emb{self.ds_size_ratio}.npy', px2_gpt_emb)
        self.px2_gpt_emb = torch.tensor(px2_gpt_emb, dtype=torch.float64).to(self.device)

        px3_gpt_emb = self._get_llm_emb(codes=px3, code_type='px')
        np.save(self.gpt_embd_path+ f'px3_gpt_emb{self.ds_size_ratio}.npy', px3_gpt_emb)
        self.px3_gpt_emb = torch.tensor(px3_gpt_emb, dtype=torch.float64).to(self.device)

        return


    def get_co_occurrence(self):

        data = self.train_dataset.samples

        # Flatten the data into a DataFrame
        rows = []
        for patient in tqdm(data):
            patient_id = patient['patient_id']
            for v_idx in range(len(patient['conditions'])):
                rows.append({'patient_id': patient_id, 'codes': patient['conditions'][v_idx] + ['p' + code for code in patient['procedures'][v_idx]] + patient['drugs'][v_idx]})

        df = pd.DataFrame(rows)

        # Initialize a counter for co-occurrences
        co_occurrence_counts = Counter()

        # Count co-occurrences
        for visit in df['codes']:
            for i in range(len(visit)):
                for j in range(i + 1, len(visit)):
                    co_occurrence_counts[(visit[i], visit[j])] += 1
                    co_occurrence_counts[(visit[j], visit[i])] += 1

        # Create a DataFrame from the co-occurrence counts
        codes = list(set(code for visit in df['codes'] for code in visit))
        co_occurrence_matrix = pd.DataFrame(0, index=codes, columns=codes)



        for (code1, code2), count in co_occurrence_counts.items():
            co_occurrence_matrix.at[code1, code2] = count

        self.co_occurrence_matrix_l3 = co_occurrence_matrix

        # Initialize the conditional probability matrix
        conditional_prob_matrix = pd.DataFrame(0.0, index=codes, columns=codes)

        # Calculate the total counts for each code
        total_counts = co_occurrence_matrix.sum(axis=1)

        # Calculate the conditional probabilities
        for code1 in tqdm(codes):
            for code2 in codes:
                if total_counts[code1] > 0:
                    conditional_prob_matrix.at[code1, code2] = co_occurrence_matrix.at[code1, code2] / total_counts[code1]

        codes_sorted = self.dx_table['l3'].unique().tolist() + self.rx_table['l3'].unique().tolist() + ['p' + code for code in self.px_table[ 'l3'].unique().tolist()]
        conditional_prob_matrix_reordered = conditional_prob_matrix.reindex(index=codes_sorted, columns=codes_sorted)

        #conditional_prob_matrix for level3
        conditional_prob_matrix_reordered.to_csv(f'../saved_files/conditional_prob_matrix{self.ds_size_ratio}.csv')

        self.conditional_prob_matrix_l3 = conditional_prob_matrix_reordered

        return


    def get_co_occurrence_for_parents(self):
        """now creating conditional_prob_matrix for other levels"""

        co_occurrence_matrix = self.co_occurrence_matrix_l3

        parent_codes_l1_sorted = self.dx_table['l1'].unique().tolist() + self.rx_table['l1'].unique().tolist() + ['p' + code for code in self.px_table[ 'l1'].unique().tolist()]
        parent_codes_l2_sorted = self.dx_table['l2'].unique().tolist() + self.rx_table['l2'].unique().tolist() + ['p' + code for code in self.px_table['l2'].unique().tolist()]

        icd9_to_parent_l1 = dict(zip(self.dx_table['l3'].values.tolist() + self.rx_table['l3'].values.tolist() + ['p' + code for code in self.px_table['l3'].values.tolist()], self.dx_table['l1'].values.tolist() + self.rx_table['l1'].values.tolist() + ['p' + code for code in self.px_table['l1'].values.tolist()]))
        icd9_to_parent_l2 = dict(zip(self.dx_table['l3'].values.tolist() + self.rx_table['l3'].values.tolist() + ['p' + code for code in self.px_table['l3'].values.tolist()], self.dx_table['l2'].values.tolist() + self.rx_table['l2'].values.tolist() + ['p' + code for code in self.px_table['l2'].values.tolist()]))

        co_occurrence_matrix_l1 = pd.DataFrame(0.0, index=parent_codes_l1_sorted, columns=parent_codes_l1_sorted, dtype=float)
        co_occurrence_matrix_l2 = pd.DataFrame(0.0, index=parent_codes_l2_sorted, columns=parent_codes_l2_sorted, dtype=float)

        # Create a dictionary to keep track of the count of child codes per parent code
        #parent_count_l1 = pd.DataFrame(0, index=parent_codes_l1_sorted, columns=parent_codes_l1_sorted, dtype=int)
        #parent_count_l2 = pd.DataFrame(0, index=parent_codes_l2_sorted, columns=parent_codes_l2_sorted, dtype=int)

        # Iterate through the conditional probability DataFrame to sum probabilities based on parent codes
        #print(conditional_prob_matrix.index)
        for child_row in tqdm(co_occurrence_matrix.index):
            for child_col in co_occurrence_matrix.columns:
                parent_row1 = icd9_to_parent_l1[child_row]
                parent_col1 = icd9_to_parent_l1[child_col]
                co_occurrence_matrix_l1.loc[parent_row1, parent_col1] += co_occurrence_matrix.loc[child_row, child_col]
                #parent_count_l1.loc[parent_row1, parent_col1] += 1

                parent_row2 = icd9_to_parent_l2[child_row]
                parent_col2 = icd9_to_parent_l2[child_col]
                co_occurrence_matrix_l2.loc[parent_row2, parent_col2] += co_occurrence_matrix.loc[child_row, child_col]
                #parent_count_l2.loc[parent_row2, parent_col2] += 1

        self.co_occurrence_matrix_l1 = co_occurrence_matrix_l1
        self.co_occurrence_matrix_l2 = co_occurrence_matrix_l2

        # Initialize the conditional probability matrix
        conditional_prob_matrix1 = pd.DataFrame(0.0, index=parent_codes_l1_sorted, columns=parent_codes_l1_sorted)
        conditional_prob_matrix2 = pd.DataFrame(0.0, index=parent_codes_l2_sorted, columns=parent_codes_l2_sorted)

        # Calculate the total counts for each code
        total_counts1 = co_occurrence_matrix_l1.sum(axis=1)
        total_counts2 = co_occurrence_matrix_l2.sum(axis=1)

        # Calculate the conditional probabilities for level 1
        for code1 in parent_codes_l1_sorted:
            for code2 in parent_codes_l1_sorted:
                if total_counts1[code1] > 0:
                    conditional_prob_matrix1.at[code1, code2] = co_occurrence_matrix_l1.at[code1, code2] / total_counts1[code1]


        # Calculate the conditional probabilities for level 2
        for code1 in parent_codes_l2_sorted:
            for code2 in parent_codes_l2_sorted:
                if total_counts2[code1] > 0:
                    conditional_prob_matrix2.at[code1, code2] = co_occurrence_matrix_l2.at[code1, code2] / total_counts2[code1]


        conditional_prob_matrix1.to_csv(f'../saved_files/conditional_prob_matrix1{self.ds_size_ratio}.csv')
        conditional_prob_matrix2.to_csv(f'../saved_files/conditional_prob_matrix2{self.ds_size_ratio}.csv')

        self.conditional_prob_matrix_l1 = conditional_prob_matrix1
        self.conditional_prob_matrix_l2 = conditional_prob_matrix2


        return

    def get_hyper_edges(self):

        print('creating hyperedge_index')

        data = self.dataset.samples
        hyper_G = {}
        hyper_edge = []
        hyper_node = []

        nodes3 = self.dx_table['l3'].tolist() + ['p' + code for code in self.px_table['l3']] + self.rx_table['l3'].tolist()
        dict_nodes3 = dict(zip(nodes3, range(len(nodes3))))

        for sample in tqdm(data):
            patient_id = sample['patient_id']
            visit_index_lists = sample['visit_index_list']
            for v_id in range(len(visit_index_lists)):
                hyperedge = str(patient_id) + f'@{visit_index_lists[v_id][0]}'
                if hyperedge not in hyper_G:
                    nodes_name = sample['conditions'][v_id] + ['p' + code for code in sample['procedures'][v_id]] + sample['drugs'][v_id]
                    nodes_ids = [dict_nodes3[node] for node in nodes_name]
                    hyper_G[hyperedge] = nodes_ids

        for node in nodes3:
            if node not in hyper_G:
                hyper_G[node] = [dict_nodes3[node]]

        hyperedge_list = hyper_G.keys()
        hyperedge_id_list = np.arange(len(hyperedge_list))
        hyper_G = dict(zip(hyperedge_id_list, hyper_G.values()))

        self.heyper_edge_emb = nn.Embedding(len(hyperedge_list), self.embedding_dim)

        for hyper_G_id in hyper_G:
            hyper_node.extend(hyper_G[hyper_G_id])
            hyper_edge.extend(len(hyper_G[hyper_G_id]) * [hyper_G_id])

        hyperedge_index_list = [hyper_node, hyper_edge]

        self.hyperedge_index_tensor = torch.tensor(hyperedge_index_list).contiguous().to(self.device1)
        print('hyperedge_index created')


        return



    def Onto_GAT(self):

        """compute GAT based on cooccurrence on each level of ontology taking to acount all code types"""

        # print('Onto_GAT started!')
        edge_index3 = self.hyperedge_index_tensor
        dx_emb3 = self.embeddings3['conditions'].weight
        rx_emb3 = self.embeddings3['drugs'].weight
        px_emb3 = self.embeddings3['procedures'].weight
        emb_input3 = torch.cat([dx_emb3, rx_emb3, px_emb3])
        emb_output3 = self.HyperG3(x=emb_input3, hyperedge_index=edge_index3, hyperedge_attr=self.heyper_edge_emb.weight)
        if self.n_G_layers==2:
            emb_output3 = self.HyperG3_2(x=emb_output3, hyperedge_index=edge_index3, hyperedge_attr=self.heyper_edge_emb.weight)

        dx_emb_new3 = emb_output3[0:len(dx_emb3)]
        rx_emb_new3 = emb_output3[len(dx_emb3): len(rx_emb3) + len(dx_emb3)]
        px_emb_new3 = emb_output3[len(rx_emb3) + len(dx_emb3):]

        edge_index1 = self.edge_index1
        dx_emb1 = self.embeddings1['conditions'].weight
        rx_emb1 = self.embeddings1['drugs'].weight
        px_emb1 = self.embeddings1['procedures'].weight
        emb_input1 = torch.cat([dx_emb1, rx_emb1, px_emb1])
        emb_output1 = self.GATLayer1(emb_input1, edge_index1)
        if self.n_G_layers==2:
            emb_output1 = self.GATLayer1_2(emb_output1, edge_index1)


        dx_emb_new1 = emb_output1[0:len(dx_emb1)]
        rx_emb_new1 = emb_output1[len(dx_emb1): len(rx_emb1) + len(dx_emb1)]
        px_emb_new1 = emb_output1[len(rx_emb1) + len(dx_emb1):]


        edge_index2 = self.edge_index2
        dx_emb2 = self.embeddings2['conditions'].weight
        rx_emb2 = self.embeddings2['drugs'].weight
        px_emb2 = self.embeddings2['procedures'].weight
        emb_input2 = torch.cat([dx_emb2, rx_emb2, px_emb2])
        emb_output2 = self.GATLayer2(emb_input2, edge_index2)
        if self.n_G_layers==2:
            emb_output2 = self.GATLayer2_2(emb_output2, edge_index2)

        dx_emb_new2 = emb_output2[0:len(dx_emb2)]
        rx_emb_new2 = emb_output2[len(dx_emb2): len(rx_emb2) + len(dx_emb2)]
        px_emb_new2 = emb_output2[len(rx_emb2) + len(dx_emb2):]


        new_embeddings = {'l1':(dx_emb_new1, rx_emb_new1, px_emb_new1), 'l2':(dx_emb_new2, rx_emb_new2, px_emb_new2), 'l3':(dx_emb_new3, rx_emb_new3, px_emb_new3)}

        return new_embeddings

    def get_edge_list_for_hap(self, table, parent_level):
        if parent_level == 'l2':
            child_level = 'l3'
        elif parent_level == 'l1':
            child_level = 'l2'

        nodes = table[parent_level].unique().tolist() + table[child_level].unique().tolist()
        nodes_indices = np.arange(len(nodes)).tolist()

        code_id_map = dict(zip(nodes, nodes_indices))

        edges_tuple = []
        for child in table[child_level].unique():
            parent = table[table[child_level] == child][parent_level].values[0]
            child_id = code_id_map[child]
            parent_id = code_id_map[parent]
            edges_tuple.append((child_id, parent_id))

        child_list, parent_list = zip(*edges_tuple)
        edge_list = [child_list, parent_list]

        return torch.tensor(edge_list)

    def bottom_up_hap(self, embeddings):
        dx_emb1, rx_emb1, px_emb1 = embeddings['l1']
        dx_emb2, rx_emb2, px_emb2 = embeddings['l2']
        dx_emb3, rx_emb3, px_emb3 = embeddings['l3']

        edge_list_dx2 = self.edge_list_dx2
        edge_list_dx1 = self.edge_list_dx1

        edge_list_rx2 = self.edge_list_rx2
        edge_list_rx1 = self.edge_list_rx1

        edge_list_px2 = self.edge_list_px2
        edge_list_px1 = self.edge_list_px1


        concat_dx_2 = torch.cat([dx_emb2[2:], dx_emb3[2:]])
        new_concat_dx_2 = self.GATLayer_hap2['conditions'](concat_dx_2 ,edge_list_dx2)
        if self.n_hap_layers ==2:
            new_concat_dx_2 = self.GATLayer_hap2_2['conditions'](new_concat_dx_2, edge_list_dx2)
        new_dx_emb2 = torch.cat([dx_emb2[:2], new_concat_dx_2[0:len(dx_emb2[2:])]])

        concat_dx_1 = torch.cat([dx_emb1[2:], new_dx_emb2[2:]])
        new_concat_dx_1 = self.GATLayer_hap1['conditions'](concat_dx_1 ,edge_list_dx1)
        if self.n_hap_layers == 2:
            new_concat_dx_1 = self.GATLayer_hap1_2['conditions'](new_concat_dx_1, edge_list_dx1)
        new_dx_emb1 = torch.cat([dx_emb1[:2], new_concat_dx_1[0: len(dx_emb1[2:])]])


        concat_rx_2 = torch.cat([rx_emb2[2:], rx_emb3[2:]])
        new_concat_rx_2 = self.GATLayer_hap2['drugs'](concat_rx_2 ,edge_list_rx2)
        if self.n_hap_layers == 2:
            new_concat_rx_2 = self.GATLayer_hap2_2['drugs'](new_concat_rx_2, edge_list_rx2)
        new_rx_emb2 = torch.cat([rx_emb2[:2], new_concat_rx_2[0:len(rx_emb2[2:])]])

        concat_rx_1 = torch.cat([rx_emb1[2:], new_rx_emb2[2:]])
        new_concat_rx_1 = self.GATLayer_hap1['drugs'](concat_rx_1 ,edge_list_rx1)
        if self.n_hap_layers == 2:
            new_concat_rx_1 = self.GATLayer_hap1_2['drugs'](new_concat_rx_1, edge_list_rx1)
        new_rx_emb1 = torch.cat([rx_emb1[:2], new_concat_rx_1[0:len(rx_emb1[2:])]])

        concat_px_2 = torch.cat([px_emb2[2:], px_emb3[2:]])
        new_concat_px_2 = self.GATLayer_hap2['procedures'](concat_px_2 ,edge_list_px2)
        if self.n_hap_layers == 2:
            new_concat_px_2 = self.GATLayer_hap2_2['procedures'](new_concat_px_2, edge_list_px2)
        new_px_emb2 = torch.cat([px_emb2[:2], new_concat_px_2[0:len(px_emb2[2:])]])

        concat_px_1 = torch.cat([px_emb1[2:], new_px_emb2[2:]])
        new_concat_px_1 = self.GATLayer_hap1['procedures'](concat_px_1 ,edge_list_px1)
        if self.n_hap_layers == 2:
            new_concat_px_1 = self.GATLayer_hap1_2['procedures'](new_concat_px_1, edge_list_px1)
        new_px_emb1 = torch.cat([px_emb1[:2], new_concat_px_1[0:len(px_emb1[2:])]])


        new_embeddings = {'l1': (new_dx_emb1, new_rx_emb1, new_px_emb1),
                          'l2': (new_dx_emb2, new_rx_emb2, new_px_emb2),
                          'l3': (dx_emb3, rx_emb3, px_emb3)}
        return new_embeddings


    def _gram(self, new_embeddings):

        dx_emb_new1, rx_emb_new1, px_emb_new1 = new_embeddings['l1']
        dx_emb_new2, rx_emb_new2, px_emb_new2 = new_embeddings['l2']
        dx_emb_new3, rx_emb_new3, px_emb_new3 = new_embeddings['l3']

        embeddings1_extended_dx = dx_emb_new1[self.dx_table_token_id['l1'].values]
        embeddings2_extended_dx = dx_emb_new2[self.dx_table_token_id['l2'].values]

        stacked_embedding_dx = torch.cat([embeddings1_extended_dx.unsqueeze(-2),
                                          embeddings2_extended_dx.unsqueeze(-2),
                                          self.embeddings3['conditions'].weight.unsqueeze(-2)], dim=-2) # (code_level3_num, 3, code_embed)
        #embedding_dx3_extend = self.embeddings3['conditions'].weight.unsqueeze(-2).repeat(1, 3, 1) # (code_level3_num, 3, code_embed)
        embedding_dx3_extend = dx_emb_new3.unsqueeze(-2).repeat(1, 3, 1)  # (code_level3_num, 3, code_embed)
        embedding_dx_cat = torch.cat([embedding_dx3_extend, stacked_embedding_dx], dim=-1) # (code_level3_num, 3, 2*code_embed)

        energy_dx = F.relu(self.code_attention_energy['conditions'](embedding_dx_cat)) # (code_level3_num, 3, 1)
        attention_dx = self.softmax(energy_dx) # (code_level3_num, 3, 1)


        # attention_gram_dx:        (code_level3_num, 3, 1) abc
        # stacked_embedding_dx:     (code_level3_num, 3, code_embed) abd
        # G_dx:                         (code_level3_num, 1, code_embed) acd
        G_dx = torch.einsum("abc,abd->acd", attention_dx, stacked_embedding_dx).squeeze(-2)
        self.G_dx = G_dx
        # (code_level3_num, code_embed)


        embeddings1_extended_rx = rx_emb_new1[self.rx_table_token_id['l1'].values]
        embeddings2_extended_rx = rx_emb_new2[self.rx_table_token_id['l2'].values]

        stacked_embedding_rx = torch.cat([embeddings1_extended_rx.unsqueeze(-2),
                                          embeddings2_extended_rx.unsqueeze(-2),
                                          self.embeddings3['drugs'].weight.unsqueeze(-2)], dim=-2) # (code_level3_num, 3, code_embed)
        #embedding_rx3_extend = self.embeddings3['drugs'].weight.unsqueeze(-2).repeat(1, 3, 1) # (code_level3_num, 3, code_embed)
        embedding_rx3_extend = rx_emb_new3.unsqueeze(-2).repeat(1, 3,1)  # (code_level3_num, 3, code_embed)
        embedding_rx_cat = torch.cat([embedding_rx3_extend, stacked_embedding_rx], dim=-1) # (code_level3_num, 3, 2*code_embed)

        energy_rx = F.relu(self.code_attention_energy['drugs'](embedding_rx_cat)) # (code_level3_num, 3, 1)
        attention_rx = self.softmax(energy_rx) # (code_level3_num, 3, 1)


        # attention_gram_dx:        (code_level3_num, 3, 1) abc
        # stacked_embedding_dx:     (code_level3_num, 3, code_embed) abd
        # G_dx:                         (code_level3_num, 1, code_embed) acd
        G_rx = torch.einsum("abc,abd->acd", attention_rx, stacked_embedding_rx).squeeze(-2)
        self.G_rx = G_rx
        # (code_level3_num, code_embed)



        embeddings1_extended_px = px_emb_new1[self.px_table_token_id['l1'].values]
        embeddings2_extended_px = px_emb_new2[self.px_table_token_id['l2'].values]

        stacked_embedding_px = torch.cat([embeddings1_extended_px.unsqueeze(-2),
                                          embeddings2_extended_px.unsqueeze(-2),
                                          self.embeddings3['procedures'].weight.unsqueeze(-2)], dim=-2) # (code_level3_num, 3, code_embed)

        #embedding_px3_extend = self.embeddings3['procedures'].weight.unsqueeze(-2).repeat(1, 3, 1) # (code_level3_num, 3, code_embed)
        embedding_px3_extend = px_emb_new3.unsqueeze(-2).repeat(1, 3,1)  # (code_level3_num, 3, code_embed)
        embedding_px_cat = torch.cat([embedding_px3_extend, stacked_embedding_px], dim=-1) # (code_level3_num, 3, 2*code_embed)

        energy_px = F.relu(self.code_attention_energy['procedures'](embedding_px_cat)) # (code_level3_num, 3, 1)
        attention_px = self.softmax(energy_px) # (code_level3_num, 3, 1)

        # attention_gram_dx:        (code_level3_num, 3, 1) abc
        # stacked_embedding_dx:     (code_level3_num, 3, code_embed) abd
        # G_dx:                         (code_level3_num, 1, code_embed) acd
        G_px = torch.einsum("abc,abd->acd", attention_px, stacked_embedding_px).squeeze(-2)
        self.G_px = G_px
        # (code_level3_num, code_embed)

        G = {'conditions':self.G_dx, 'drugs':self.G_rx, 'procedures':self.G_px}
        self.G = G

        return G

    def CustomEmbeddingLookup(self, embedding_matrix, input_indices):
        # input_indices shape: (num_batch, num_visits, num_codes)
        batch_size, num_visits, num_codes = input_indices.shape

        # Reshape input_indices to 2D tensor for gathering
        input_indices_flat = input_indices.view(-1, num_codes)

        # Fetch embeddings; output shape: (num_batch * num_visits, num_codes, code_dim)
        embedded = embedding_matrix[input_indices_flat]

        # Reshape back to (num_batch, num_visits, num_codes, code_dim)
        embedded = embedded.view(batch_size, num_visits, num_codes, -1)

        return embedded


    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        patient_emb = []
        for feature_key in self.feature_keys:

            input_info = self.dataset.input_info[feature_key]

            # each patient's feature is represented by [[code1, code2],[code3]]
            assert input_info["dim"] == 3 and input_info["type"] == str

            input = self.feat_tokenizers3[feature_key].batch_encode_3d(
                kwargs[feature_key]
            )

            # (patient, visit, event)
            input = torch.tensor(input, dtype=torch.long, device=self.device)

            # (patient, visit, event, embedding_dim)
            new_embeddings = self.Onto_GAT()
            new_embeddings = self.bottom_up_hap(new_embeddings)
            G = self._gram(new_embeddings)
            x = self.CustomEmbeddingLookup(G[feature_key], input)

            # (patient, visit, embedding_dim)
            x = torch.sum(x, dim=2)
            pad_idx = self.feat_tokenizers3[feature_key].vocabulary("<pad>")

            # (patient, visit)
            mask = torch.any(input != pad_idx, dim=2)

            _, x = self.transformer[feature_key](x, mask)
            patient_emb.append(x)



        # (patient, features * hidden_dim)
        patient_emb = torch.cat(patient_emb, dim=1)
        # (patient, label_size)
        logits = self.fc(patient_emb)
        # obtain y_true, loss, y_prob
        y_true = self.prepare_labels(kwargs[self.label_key], self.label_tokenizer)
        loss = self.get_loss_function()(logits, y_true)
        y_prob = self.prepare_y_prob(logits)
        return {"loss": loss, "y_prob": y_prob, "y_true": y_true}
