from torch.utils.data import Dataset, DataLoader
from typing import Dict, Callable, Tuple, Union, List, Optional
from pyhealth.datasets.sample_dataset import SampleEHRDataset
from tqdm import tqdm
import pickle

import os
from collections.abc import Iterable
from typing import Union
import pandas as pd
from pyhealth.medcode import InnerMap, CrossMap
class ICD10toICD9():
    """
    Maps icd10 codes to icd9.

    Source of mapping: https://www.nber.org/research/data/icd-9-cm-and-icd-10-cm-and-icd-10-pcs-crosswalk-or-general-equivalence-mappings
    """
    def __init__(self, dx=True):
        super().__init__()

        if dx:
            #self.filename = "../saved_files/icd_maping/icd10cmtoicd9gem.csv"
            self.filename = '../saved_files/icd_maping/2018_I10gem.txt'
            self.df = pd.read_csv(self.filename)
            self.df = pd.read_csv(self.filename, delim_whitespace=True, header=None)
            self.df.columns = ['icd10cm', 'icd9cm', 'flag']
        else:
            self.filename = '../saved_files/icd_maping/gem_pcsi9.txt'
            self.df = pd.read_csv(self.filename, delim_whitespace=True, header=None)
            self.df.columns = ['icd10cm', 'icd9cm', 'flag']

        self._setup()

    def _setup(self):
        self.icd10_to_icd9 = self._parse_file()


    def _map_single(self, icd10code : str):
        return self.icd10_to_icd9.get(icd10code)

    def map(self, icd10code : Union[str, Iterable]) -> Union[str, Iterable]:
        """
        Given an icd10 code, returns the corresponding icd9 code.

        Parameters
        ----------

        code : str | Iterable
            icd10 code

        Returns:
            icd9 code or None when the mapping is not possible
        """

        if isinstance(icd10code, str):
            return self._map_single(icd10code)

        elif isinstance(icd10code, Iterable):
            return [self._map_single(c) for c in icd10code]

        return None


    def _parse_file(self):

        mapping = {}
        for icd10_code, icd9_code in zip(self.df['icd10cm'], self.df['icd9cm']):
            icd10, icd9 = str(icd10_code).strip(), str(icd9_code).strip()
            mapping[icd10] = icd9

        return mapping



def collate_fn_dict_cl(batch):
    out= (
          {key: [d[0][key] for d in batch] for key in batch[0][0]},
          {key: [d[1][key] for d in batch] for key in batch[0][0]}
          )

    return out

def collate_fn_dict_levels(batch):
    out= (
          {key: [d[0][key] for d in batch] for key in batch[0][0]},
          {key: [d[1][key] for d in batch] for key in batch[0][0]},
          {key: [d[2][key] for d in batch] for key in batch[0][0]}
          )

    return out




def get_dataloader_cl(dataset, batch_size, shuffle=False):

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn_dict_cl,
    )

    return dataloader

def get_dataloader_levels(dataset, batch_size, shuffle=False):

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn_dict_levels,
    )

    return dataloader

class MultiDataset(Dataset):
    def __init__(self, dataset):
        self.dataset1 = dataset[0]
        self.dataset2 = dataset[1]


        # Calculate total length
        self.total_length = len(dataset[0])

    def __len__(self):
        return self.total_length

    def __getitem__(self, index):

        return (self.dataset1[index],self.dataset2[index])

class MultiDataset_levels(Dataset):
    def __init__(self, dataset):
        self.dataset1 = dataset[0]
        self.dataset2 = dataset[1]
        self.dataset3 = dataset[2]


        # Calculate total length
        self.total_length = len(dataset[0])

    def __len__(self):
        return self.total_length

    def __getitem__(self, index):

        return (self.dataset1[index],self.dataset2[index], self.dataset3[index])





def customized_set_task_levels(
    dataset,
    task_fn: Callable,
    ccs_label=False,
    task_name: Optional[str] = None
) -> SampleEHRDataset:
    """Processes the base dataset to generate the task-specific sample dataset.

    This function should be called by the user after the base dataset is
    initialized. It will iterate through all patients in the base dataset
    and call `task_fn` which should be implemented by the specific task.

    Args:
        task_fn: a function that takes a single patient and returns a
            list of samples (each sample is a dict with patient_id, visit_id,
            and other task-specific attributes as key). The samples will be
            concatenated to form the sample dataset.
        task_name: the name of the task. If None, the name of the task
            function will be used.

    Returns:
        sample_dataset: the task-specific sample dataset.

    Note:
        In `task_fn`, a patient may be converted to multiple samples, e.g.,
            a patient with three visits may be converted to three samples
            ([visit 1], [visit 1, visit 2], [visit 1, visit 2, visit 3]).
            Patients can also be excluded from the task dataset by returning
            an empty list.
    """
    samples1 = []
    samples2 = []
    samples3 = []

    icd9cm = InnerMap.load("ICD9CM")
    icd9cmproc = InnerMap.load("ICD9PROC")
    atc = InnerMap.load('ATC')
    CCS_mapping = CrossMap("ICD9CM", "CCSCM")


    if ccs_label:
        path = 'CCS'
    else:
        path = 'ICD'


    if os.path.isfile(f'../saved_files/samples/levels/{path}/samples1.pkl'):
        with open(f'../saved_files/samples/levels/{path}/samples1.pkl', 'rb') as f:
            samples1 = pickle.load(f)
        with open(f'../saved_files/samples/levels/{path}/samples2.pkl', 'rb') as f:
            samples2 = pickle.load(f)
        with open(f'../saved_files/samples/levels/{path}/samples3.pkl', 'rb') as f:
            samples3 = pickle.load(f)
        print('-- samples loaded!')

    else:
        print('-- generating samples ...')
        for patient_id, patient in tqdm(dataset.patients.items(), desc=f"Generating samples for {task_name}"):
            sample1, sample2, sample3 = task_fn(patient, icd9cm, icd9cmproc, atc, CCS_mapping, ccs_label=ccs_label)
            samples1.extend(sample1)
            samples2.extend(sample2)
            samples3.extend(sample3)


        with open(f'../saved_files/samples/levels/{path}/samples1.pkl', 'wb') as f:
            pickle.dump(samples1, f)
        with open(f'../saved_files/samples/levels/{path}/samples2.pkl', 'wb') as f:
            pickle.dump(samples2, f)
        with open(f'../saved_files/samples/levels/{path}/samples3.pkl', 'wb') as f:
            pickle.dump(samples3, f)


    sample_dataset1 = SampleEHRDataset(
        samples=samples1,
        code_vocs=dataset.code_vocs,
        dataset_name=dataset.dataset_name,
        task_name=task_name,
    )

    sample_dataset2 = SampleEHRDataset(
        samples=samples2,
        code_vocs=dataset.code_vocs,
        dataset_name=dataset.dataset_name,
        task_name=task_name,
    )

    sample_dataset3 = SampleEHRDataset(
        samples=samples3,
        code_vocs=dataset.code_vocs,
        dataset_name=dataset.dataset_name,
        task_name=task_name,
    )



    return sample_dataset1, sample_dataset2, sample_dataset3





def customized_set_task_cl(
    dataset,
    task_fn: Callable,
    ccs_label=False,
    task_name: Optional[str] = None
) -> SampleEHRDataset:
    """Processes the base dataset to generate the task-specific sample dataset.

    This function should be called by the user after the base dataset is
    initialized. It will iterate through all patients in the base dataset
    and call `task_fn` which should be implemented by the specific task.

    Args:
        task_fn: a function that takes a single patient and returns a
            list of samples (each sample is a dict with patient_id, visit_id,
            and other task-specific attributes as key). The samples will be
            concatenated to form the sample dataset.
        task_name: the name of the task. If None, the name of the task
            function will be used.

    Returns:
        sample_dataset: the task-specific sample dataset.

    Note:
        In `task_fn`, a patient may be converted to multiple samples, e.g.,
            a patient with three visits may be converted to three samples
            ([visit 1], [visit 1, visit 2], [visit 1, visit 2, visit 3]).
            Patients can also be excluded from the task dataset by returning
            an empty list.
    """
    samples = []
    samples_fake = []

    condition_onto = pd.read_csv('../saved_files/ontology_tables/condition_onto.csv')
    procedure_onto = pd.read_csv('../saved_files/ontology_tables/procedure_onto.csv')
    drug_onto = pd.read_csv('../saved_files/ontology_tables/drug_onto.csv')

    onto_tables = {'conditions':condition_onto, 'drugs':drug_onto, 'procedures':procedure_onto}

    if ccs_label:
        path = 'CL_ccs'
    else:
        path = 'CL'


    if os.path.isfile(f'../saved_files/samples/{path}/samples.pkl'):
        with open(f'../saved_files/samples/{path}/samples.pkl', 'rb') as f:
            samples = pickle.load(f)
        with open(f'../saved_files/samples/{path}/samples_fake.pkl', 'rb') as f:
            samples_fake = pickle.load(f)
        print('-- samples loaded!')
    else:
        print('-- generating samples ...')
        for patient_id, patient in tqdm(dataset.patients.items(), desc=f"Generating samples for {task_name}"):
            sample, sample_fake = task_fn(patient, onto_tables, ccs_label)
            samples.extend(sample)
            samples_fake.extend(sample_fake)

        with open(f'../saved_files/samples/{path}/samples.pkl', 'wb') as f:
            pickle.dump(samples, f)
        with open(f'../saved_files/samples/{path}/samples_fake.pkl', 'wb') as f:
            pickle.dump(samples_fake, f)


    sample_dataset = SampleEHRDataset(
        samples=samples,
        code_vocs=dataset.code_vocs,
        dataset_name=dataset.dataset_name,
        task_name=task_name,
    )

    sample_dataset_fake = SampleEHRDataset(
        samples=samples_fake,
        code_vocs=dataset.code_vocs,
        dataset_name=dataset.dataset_name,
        task_name=task_name,
    )

    return sample_dataset, sample_dataset_fake



def customized_set_task(
    dataset,
    task_fn: Callable,
    ccs_label=False,
    task_name: Optional[str] = None
) -> SampleEHRDataset:
    """Processes the base dataset to generate the task-specific sample dataset.

    This function should be called by the user after the base dataset is
    initialized. It will iterate through all patients in the base dataset
    and call `task_fn` which should be implemented by the specific task.

    Args:
        task_fn: a function that takes a single patient and returns a
            list of samples (each sample is a dict with patient_id, visit_id,
            and other task-specific attributes as key). The samples will be
            concatenated to form the sample dataset.
        task_name: the name of the task. If None, the name of the task
            function will be used.

    Returns:
        sample_dataset: the task-specific sample dataset.

    Note:
        In `task_fn`, a patient may be converted to multiple samples, e.g.,
            a patient with three visits may be converted to three samples
            ([visit 1], [visit 1, visit 2], [visit 1, visit 2, visit 3]).
            Patients can also be excluded from the task dataset by returning
            an empty list.
    """
    samples = []
    mapping = CrossMap("ICD9CM", "CCSCM")


    if ccs_label:
        path = 'CL_ccs'
    else:
        path = 'CL'


    if os.path.isfile(f'../saved_files/samples/{path}/samples.pkl'):
        with open(f'../saved_files/samples/{path}/samples.pkl', 'rb') as f:
            samples = pickle.load(f)
        print('-- samples loaded!')
    else:
        print('-- generating samples ...')
        for patient_id, patient in tqdm(dataset.patients.items(), desc=f"Generating samples for {task_name}"):
            sample = task_fn(patient, ccs_label, mapping)
            samples.extend(sample)

        with open(f'../saved_files/samples/{path}/samples.pkl', 'wb') as f:
            pickle.dump(samples, f)


    sample_dataset = SampleEHRDataset(
        samples=samples,
        code_vocs=dataset.code_vocs,
        dataset_name=dataset.dataset_name,
        task_name=task_name,
    )


    return sample_dataset


import random
def select_random_subset(original_list, ratio=0.5, seed=None):
    """
    Select a random subset of the original list based on the given ratio.

    Parameters:
    original_list (list): The list from which to select the subset.
    ratio (float): The ratio of the subset length to the original list length (default is 0.5).
    seed (int, optional): The seed for the random number generator for reproducibility.

    Returns:
    list: A randomly selected subset of the original list.
    """
    if seed is not None:
        random.seed(seed)

    subset_length = int(len(original_list) * ratio)
    random_subset = random.sample(original_list, subset_length)

    return random_subset








def customized_set_task_mimic4(
    dataset,
    task_fn: Callable,
    ccs_label=False,
    task_name: Optional[str] = None,
    ds_size_ratio=1,
        seed=None
) -> SampleEHRDataset:

    """Processes the base dataset to generate the task-specific sample dataset.

    This function should be called by the user after the base dataset is
    initialized. It will iterate through all patients in the base dataset
    and call `task_fn` which should be implemented by the specific task.

    Args:
        task_fn: a function that takes a single patient and returns a
            list of samples (each sample is a dict with patient_id, visit_id,
            and other task-specific attributes as key). The samples will be
            concatenated to form the sample dataset.
        task_name: the name of the task. If None, the name of the task
            function will be used.

    Returns:
        sample_dataset: the task-specific sample dataset.

    Note:
        In `task_fn`, a patient may be converted to multiple samples, e.g.,
            a patient with three visits may be converted to three samples
            ([visit 1], [visit 1, visit 2], [visit 1, visit 2, visit 3]).
            Patients can also be excluded from the task dataset by returning
            an empty list.
    """

    samples = []
    ccs_samples = []

    if ccs_label:
        path = '_ccs'
    else:
        path = ''
    path = path + f'_{ds_size_ratio}'

    if os.path.isfile(f'../saved_files/mimic4_samples/samples{path}.pkl'):
        with open(f'../saved_files/mimic4_samples/samples{path}.pkl', 'rb') as f:
            final_samples = pickle.load(f)
        print('--- samples loaded!')

        sample_dataset = SampleEHRDataset(
            samples=final_samples,
            code_vocs=dataset.code_vocs,
            dataset_name=dataset.dataset_name,
            task_name=task_name,
        )

        return sample_dataset
    else:
        print('--- generating samples ...')


        dx_mapper = ICD10toICD9(dx=True)
        px_mapper = ICD10toICD9(dx=False)
        # initialize an InnerMap
        icd9proc = InnerMap.load("ICD9PROC")
        icd9cm = InnerMap.load("ICD9CM")
        ccs_mapping = CrossMap("ICD9CM", "CCSCM")

        for patient_id, patient in tqdm(dataset.patients.items(), desc=f"Generating samples for {task_name}"):
            sample, ccs_sample = task_fn(patient=patient, dx_mapper= dx_mapper, px_mapper= px_mapper,
                                         icd9proc=icd9proc, icd9cm=icd9cm, ccs_mapping=ccs_mapping)
            samples.extend(sample)
            ccs_samples.extend(ccs_sample)


        if ds_size_ratio != 1:
            final_samples = select_random_subset(samples, ratio=ds_size_ratio, seed=seed)
            final_samples_ccs = select_random_subset(ccs_samples, ratio=ds_size_ratio, seed=seed)
        else:
            final_samples = samples
            final_samples_ccs = ccs_samples

        with open(f'../saved_files/mimic4_samples/samples_{ds_size_ratio}.pkl', 'wb') as f:
            pickle.dump(final_samples, f)
        with open(f'../saved_files/mimic4_samples/samples_ccs_{ds_size_ratio}.pkl', 'wb') as f:
            pickle.dump(final_samples_ccs, f)


        if ccs_label:
            final_samples = final_samples_ccs
        else:
            final_samples = final_samples


        sample_dataset = SampleEHRDataset(
            samples=final_samples,
            code_vocs=dataset.code_vocs,
            dataset_name=dataset.dataset_name,
            task_name=task_name,
        )

        return sample_dataset




def customized_set_task_mimic3(
    dataset,
    task_fn: Callable,
    ccs_label=False,
    task_name: Optional[str] = None,
    ds_size_ratio=1,
        seed=None
) -> SampleEHRDataset:

    """Processes the base dataset to generate the task-specific sample dataset.

    This function should be called by the user after the base dataset is
    initialized. It will iterate through all patients in the base dataset
    and call `task_fn` which should be implemented by the specific task.

    Args:
        task_fn: a function that takes a single patient and returns a
            list of samples (each sample is a dict with patient_id, visit_id,
            and other task-specific attributes as key). The samples will be
            concatenated to form the sample dataset.
        task_name: the name of the task. If None, the name of the task
            function will be used.

    Returns:
        sample_dataset: the task-specific sample dataset.

    Note:
        In `task_fn`, a patient may be converted to multiple samples, e.g.,
            a patient with three visits may be converted to three samples
            ([visit 1], [visit 1, visit 2], [visit 1, visit 2, visit 3]).
            Patients can also be excluded from the task dataset by returning
            an empty list.
    """

    samples = []
    ccs_samples = []

    if ccs_label:
        path = '_ccs'
    else:
        path = ''
    path = path + f'_{ds_size_ratio}'

    if os.path.isfile(f'../saved_files/mimic3_samples/samples{path}.pkl'):
        with open(f'../saved_files/mimic3_samples/samples{path}.pkl', 'rb') as f:
            final_samples = pickle.load(f)
        print('--- samples loaded!')

        sample_dataset = SampleEHRDataset(
            samples=final_samples,
            code_vocs=dataset.code_vocs,
            dataset_name=dataset.dataset_name,
            task_name=task_name,
        )

        return sample_dataset
    else:
        print('--- generating samples ...')
        ccs_mapping = CrossMap("ICD9CM", "CCSCM")

        for patient_id, patient in tqdm(dataset.patients.items(), desc=f"Generating samples for {task_name}"):
            sample, ccs_sample = task_fn(patient=patient, ccs_mapping=ccs_mapping)
            samples.extend(sample)
            ccs_samples.extend(ccs_sample)


        if ds_size_ratio != 1:
            final_samples = select_random_subset(samples, ratio=ds_size_ratio, seed=seed)
            final_samples_ccs = select_random_subset(ccs_samples, ratio=ds_size_ratio, seed=seed)
        else:
            final_samples = samples
            final_samples_ccs = ccs_samples

        with open(f'../saved_files/mimic3_samples/samples_{ds_size_ratio}.pkl', 'wb') as f:
            pickle.dump(final_samples, f)
        with open(f'../saved_files/mimic3_samples/samples_ccs_{ds_size_ratio}.pkl', 'wb') as f:
            pickle.dump(final_samples_ccs, f)


        if ccs_label:
            final_samples = final_samples_ccs
        else:
            final_samples = final_samples


        sample_dataset = SampleEHRDataset(
            samples=final_samples,
            code_vocs=dataset.code_vocs,
            dataset_name=dataset.dataset_name,
            task_name=task_name,
        )

        return sample_dataset


def customized_set_task_levels_mimic3(
        dataset,
        task_fn: Callable,
        ccs_label=False,
        task_name: Optional[str] = None,
        ds_size_ratio=1,
        seed=None
) -> SampleEHRDataset:
    """Processes the base dataset to generate the task-specific sample dataset.

    This function should be called by the user after the base dataset is
    initialized. It will iterate through all patients in the base dataset
    and call `task_fn` which should be implemented by the specific task.

    Args:
        task_fn: a function that takes a single patient and returns a
            list of samples (each sample is a dict with patient_id, visit_id,
            and other task-specific attributes as key). The samples will be
            concatenated to form the sample dataset.
        task_name: the name of the task. If None, the name of the task
            function will be used.

    Returns:
        sample_dataset: the task-specific sample dataset.

    Note:
        In `task_fn`, a patient may be converted to multiple samples, e.g.,
            a patient with three visits may be converted to three samples
            ([visit 1], [visit 1, visit 2], [visit 1, visit 2, visit 3]).
            Patients can also be excluded from the task dataset by returning
            an empty list.
    """

    samples1 = []
    samples2 = []
    samples3 = []

    icd9cm = InnerMap.load("ICD9CM")
    icd9cmproc = InnerMap.load("ICD9PROC")
    atc = InnerMap.load('ATC')
    CCS_mapping = CrossMap("ICD9CM", "CCSCM")

    if ccs_label:
        path = 'CCS'
    else:
        path = 'ICD'

    extention = f'_{ds_size_ratio}'

    if os.path.isfile(f'../saved_files/samples/levels/{path}/samples1{extention}.pkl'):
        with open(f'../saved_files/samples/levels/{path}/samples1{extention}.pkl', 'rb') as f:
            final_samples1 = pickle.load(f)
        with open(f'../saved_files/samples/levels/{path}/samples2{extention}.pkl', 'rb') as f:
            final_samples2 = pickle.load(f)
        with open(f'../saved_files/samples/levels/{path}/samples3{extention}.pkl', 'rb') as f:
            final_samples3 = pickle.load(f)
        print('-- samples loaded!')

        sample_dataset1 = SampleEHRDataset(
            samples=final_samples1,
            code_vocs=dataset.code_vocs,
            dataset_name=dataset.dataset_name,
            task_name=task_name,
        )

        sample_dataset2 = SampleEHRDataset(
            samples=final_samples2,
            code_vocs=dataset.code_vocs,
            dataset_name=dataset.dataset_name,
            task_name=task_name,
        )

        sample_dataset3 = SampleEHRDataset(
            samples=final_samples3,
            code_vocs=dataset.code_vocs,
            dataset_name=dataset.dataset_name,
            task_name=task_name,
        )
        return sample_dataset1, sample_dataset2, sample_dataset3


    else:
        print('-- generating samples ...')
        for patient_id, patient in tqdm(dataset.patients.items(), desc=f"Generating samples for {task_name}"):
            sample1, sample2, sample3 = task_fn(patient, icd9cm, icd9cmproc, atc, CCS_mapping, ccs_label=ccs_label)
            samples1.extend(sample1)
            samples2.extend(sample2)
            samples3.extend(sample3)

        if ds_size_ratio != 1:
            final_samples1 = select_random_subset(samples1, ratio=ds_size_ratio, seed=seed)
            final_samples2 = select_random_subset(samples2, ratio=ds_size_ratio, seed=seed)
            final_samples3 = select_random_subset(samples3, ratio=ds_size_ratio, seed=seed)
        else:
            final_samples1 = samples1
            final_samples2 = samples2
            final_samples3 = samples3

        with open(f'../saved_files/samples/levels/{path}/samples1{extention}.pkl', 'wb') as f:
            pickle.dump(final_samples1, f)
        with open(f'../saved_files/samples/levels/{path}/samples2{extention}.pkl', 'wb') as f:
            pickle.dump(final_samples2, f)
        with open(f'../saved_files/samples/levels/{path}/samples3{extention}.pkl', 'wb') as f:
            pickle.dump(final_samples3, f)

    sample_dataset1 = SampleEHRDataset(
        samples=final_samples1,
        code_vocs=dataset.code_vocs,
        dataset_name=dataset.dataset_name,
        task_name=task_name,
    )

    sample_dataset2 = SampleEHRDataset(
        samples=final_samples2,
        code_vocs=dataset.code_vocs,
        dataset_name=dataset.dataset_name,
        task_name=task_name,
    )

    sample_dataset3 = SampleEHRDataset(
        samples=final_samples3,
        code_vocs=dataset.code_vocs,
        dataset_name=dataset.dataset_name,
        task_name=task_name,
    )
    return sample_dataset1, sample_dataset2, sample_dataset3


def customized_set_task_levels_mimic4(
        dataset,
        task_fn: Callable,
        ccs_label=False,
        task_name: Optional[str] = None,
        ds_size_ratio=1,
        seed=None
) -> SampleEHRDataset:
    """Processes the base dataset to generate the task-specific sample dataset.

    This function should be called by the user after the base dataset is
    initialized. It will iterate through all patients in the base dataset
    and call `task_fn` which should be implemented by the specific task.

    Args:
        task_fn: a function that takes a single patient and returns a
            list of samples (each sample is a dict with patient_id, visit_id,
            and other task-specific attributes as key). The samples will be
            concatenated to form the sample dataset.
        task_name: the name of the task. If None, the name of the task
            function will be used.

    Returns:
        sample_dataset: the task-specific sample dataset.

    Note:
        In `task_fn`, a patient may be converted to multiple samples, e.g.,
            a patient with three visits may be converted to three samples
            ([visit 1], [visit 1, visit 2], [visit 1, visit 2, visit 3]).
            Patients can also be excluded from the task dataset by returning
            an empty list.
    """

    samples1 = []
    samples2 = []
    samples3 = []

    icd9cm = InnerMap.load("ICD9CM")
    icd9cmproc = InnerMap.load("ICD9PROC")
    atc = InnerMap.load('ATC')
    CCS_mapping = CrossMap("ICD9CM", "CCSCM")
    dx_mapper = ICD10toICD9(dx=True)
    px_mapper = ICD10toICD9(dx=False)

    if ccs_label:
        path = 'CCS'
    else:
        path = 'ICD'

    extention = f'_{ds_size_ratio}'

    if os.path.isfile(f'../saved_files/samples/levels_mimic4/{path}/samples1{extention}.pkl'):
        with open(f'../saved_files/samples/levels_mimic4/{path}/samples1{extention}.pkl', 'rb') as f:
            final_samples1 = pickle.load(f)
        with open(f'../saved_files/samples/levels_mimic4/{path}/samples2{extention}.pkl', 'rb') as f:
            final_samples2 = pickle.load(f)
        with open(f'../saved_files/samples/levels_mimic4/{path}/samples3{extention}.pkl', 'rb') as f:
            final_samples3 = pickle.load(f)
        print('-- samples loaded!')

        print(f'len(final_samples1):{len(final_samples1)}')
        print(f'len(final_samples2):{len(final_samples2)}')
        print(f'len(final_samples3):{len(final_samples3)}')

        sample_dataset1 = SampleEHRDataset(
            samples=final_samples1,
            code_vocs=dataset.code_vocs,
            dataset_name=dataset.dataset_name,
            task_name=task_name,
        )

        sample_dataset2 = SampleEHRDataset(
            samples=final_samples2,
            code_vocs=dataset.code_vocs,
            dataset_name=dataset.dataset_name,
            task_name=task_name,
        )

        sample_dataset3 = SampleEHRDataset(
            samples=final_samples3,
            code_vocs=dataset.code_vocs,
            dataset_name=dataset.dataset_name,
            task_name=task_name,
        )
        return sample_dataset1, sample_dataset2, sample_dataset3

    else:
        print('-- generating samples ...')
        for patient_id, patient in tqdm(dataset.patients.items(), desc=f"Generating samples for {task_name}"):
            sample1, sample2, sample3 = task_fn(patient, dx_mapper, px_mapper, icd9cm, icd9cmproc, atc, CCS_mapping, ccs_label=ccs_label)
            samples1.extend(sample1)
            samples2.extend(sample2)
            samples3.extend(sample3)

        if ds_size_ratio != 1:
            print(f'sampling with ds_size_ratio={ds_size_ratio}!')
            final_samples1 = select_random_subset(samples1, ratio=ds_size_ratio, seed=seed)
            final_samples2 = select_random_subset(samples2, ratio=ds_size_ratio, seed=seed)
            final_samples3 = select_random_subset(samples3, ratio=ds_size_ratio, seed=seed)
        else:
            print('whole dataset used!')
            final_samples1 = samples1
            final_samples2 = samples2
            final_samples3 = samples3

        print(f'len(final_samples1):{len(final_samples1)}')
        print(f'len(final_samples2):{len(final_samples2)}')
        print(f'len(final_samples3):{len(final_samples3)}')

        with open(f'../saved_files/samples/levels_mimic4/{path}/samples1{extention}.pkl', 'wb') as f:
            pickle.dump(final_samples1, f)
        with open(f'../saved_files/samples/levels_mimic4/{path}/samples2{extention}.pkl', 'wb') as f:
            pickle.dump(final_samples2, f)
        with open(f'../saved_files/samples/levels_mimic4/{path}/samples3{extention}.pkl', 'wb') as f:
            pickle.dump(final_samples3, f)

    sample_dataset1 = SampleEHRDataset(
        samples=final_samples1,
        code_vocs=dataset.code_vocs,
        dataset_name=dataset.dataset_name,
        task_name=task_name,
    )
    sample_dataset2 = SampleEHRDataset(
        samples=final_samples2,
        code_vocs=dataset.code_vocs,
        dataset_name=dataset.dataset_name,
        task_name=task_name,
    )
    sample_dataset3 = SampleEHRDataset(
        samples=final_samples3,
        code_vocs=dataset.code_vocs,
        dataset_name=dataset.dataset_name,
        task_name=task_name,
    )

    return sample_dataset1, sample_dataset2, sample_dataset3