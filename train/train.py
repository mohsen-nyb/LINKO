from preprocessing.MIMIC_preprocessing import preprocessing_seq_diag_pred
from tasks.diagnosis_prediction import sequential_diagnosis_prediction_mimic3
from model.LINKO import Mega
from pyhealth.trainer import Trainer
import torch
import numpy as np
from pyhealth.datasets import MIMIC3Dataset
from utils.eval_test import evaluate, get_group_labels1, calculate_confidence_interval
import os
from utils.data import customized_set_task_mimic3
import random


def nfold_experiment(mimic3sample, epochs , ds_size_ratio, print_results=True, record_results=True):

    data = mimic3sample.samples
    co_occurrence_counts, groups1 = get_group_labels1(data)

    seeds = [123, 321, 54, 65, 367]


    list_top_k = [3,5,7,10, 15, 20, 30]
    metrics_dict = {'roc_auc_samples': [], 'pr_auc_samples': [], 'f1_samples': []}

    for group_name in groups1.keys():
        metrics_dict[f'roc_auc_samples_{group_name}'] = []
        metrics_dict[f'pr_auc_samples_{group_name}'] = []


    for k in list_top_k:
        metrics_dict[f'acc_at_k={k}'] = []
        metrics_dict[f'hit_at_k={k}'] = []
        for group_name in groups1.keys():
            metrics_dict[f'Group_acc_at_k={k}@' + group_name] = []
            metrics_dict[f'Group_hit_at_k={k}@' + group_name] = []


    for seed in seeds:
        print(f'----------------------seed:{seed}-----------------------')

        torch.manual_seed(seed)
        np.random.seed(seed)
        #random.seed(seed)
        # Set seed for CUDA operations
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        train_loader, val_loader, test_loader = preprocessing_seq_diag_pred(
            mimic3sample, train_ratio=0.8, val_ratio=0.2, test_ratio=0, batch_size=252, print_stats=False, seed=seed
        )
        print('preprocessing done!')

        # Stage 3: define model
        device = "cuda:0"

        if ds_size_ratio==1.0:
            ds_size_ratio_model = ''
        else:
            ds_size_ratio_model = '_' + str(ds_size_ratio)

        model = Mega(
            dataset=mimic3sample,
            feature_keys=["conditions", "drugs", "procedures"],
            label_key="label",
            mode="multilabel",
            embedding_dim=128,dropout=0.5,nheads=1,nlayers=1,
            G_dropout=0.1,n_G_heads=4,n_G_layers=1,
            threshold3=0.00, threshold2=0.02, threshold1=0.00,
            n_hap_layers=1, n_hap_heads=2, hap_dropout=0.2,
            llm_model='text-embedding-3-small', gpt_embd_path='../saved_files/gpt_code_emb/tx-emb-3-small/include_all_parents2/', #gpt_embd_path='../saved_files/gpt_code_emb/tx-emb-3-small/' => so far best results
            ds_size_ratio=ds_size_ratio_model,device=device, seed=seed,
        )
        model.to(device)


        # if torch.cuda.device_count() > 1:
        #     print("Let's use", torch.cuda.device_count(), "GPUs!")
        #     model = torch.nn.DataParallel(model)

        model.to(device)

        # Stage 4: model training

        trainer = Trainer(model=model,
                          checkpoint_path=None,
                          metrics = ['roc_auc_samples', 'pr_auc_samples', 'f1_samples'],
                          enable_logging=True,
                          output_path=f"./output/OntoFAR_{ds_size_ratio}",
                          exp_name=f'EXP_:seed:{seed}',
                          device=device)

        trainer.train(
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            epochs=epochs,
            optimizer_class =  torch.optim.Adam,
            optimizer_params = {"lr": 1e-3},
            weight_decay=0.0,
            monitor="pr_auc_samples",
            monitor_criterion='max',
            load_best_model_at_last=True
        )


        all_metrics = [

            "pr_auc_samples",
            "roc_auc_samples",
            "f1_samples",
        ]

        y_true, y_prob, loss = trainer.inference(val_loader)

        result = evaluate(y_true, y_prob, co_occurrence_counts, groups1, list_top_k=list_top_k, all_metrics=all_metrics)

        print ('\n', result)

        metrics_dict['pr_auc_samples'].append(result['pr_auc_samples'])
        metrics_dict['roc_auc_samples'].append(result['roc_auc_samples'])
        metrics_dict['f1_samples'].append(result['f1_samples'])

        for group_name in groups1.keys():
            metrics_dict[f'roc_auc_samples_{group_name}'].append(result[f'roc_auc_samples_{group_name}'])
            metrics_dict[f'pr_auc_samples_{group_name}'].append(result[f'pr_auc_samples_{group_name}'])


        for k in list_top_k:
            metrics_dict[f'acc_at_k={k}'].append(result[f'acc_at_k={k}'])
            metrics_dict[f'hit_at_k={k}'].append(result[f'hit_at_k={k}'])
            for group_name in groups1.keys():
                metrics_dict[f'Group_acc_at_k={k}@' + group_name].append(result[f'Group_acc_at_k={k}@' + group_name])
                metrics_dict[f'Group_hit_at_k={k}@' + group_name].append(result[f'Group_hit_at_k={k}@' + group_name])

    if print_results:
        print()
        print('mean pr_auc_samples:', np.mean(metrics_dict['pr_auc_samples']))
        print('max pr_auc_samples:', np.max(metrics_dict['pr_auc_samples']))
        print('min pr_auc_samples:', np.min(metrics_dict['pr_auc_samples']))
        print('CI pr_auc_samples:', calculate_confidence_interval(metrics_dict['pr_auc_samples']))

        print()

        print('mean roc_auc_samples:', np.mean(metrics_dict['roc_auc_samples']))
        print('max roc_auc_samples:', np.max(metrics_dict['roc_auc_samples']))
        print('min roc_auc_samples:', np.min(metrics_dict['roc_auc_samples']))
        print('CI roc_auc_samples:', calculate_confidence_interval(metrics_dict['roc_auc_samples']))
        print()

        print('mean f1_samples:', np.mean(metrics_dict['f1_samples']))
        print('max f1_samples:', np.max(metrics_dict['f1_samples']))
        print('min f1_samples:', np.min(metrics_dict['f1_samples']))
        print('CI f1_samples:', calculate_confidence_interval(metrics_dict['f1_samples']))
        print()

        for group_name in groups1:
            print()
            print(f'mean pr_auc_samples_{group_name}:', np.mean(metrics_dict[f'pr_auc_samples_{group_name}']))
            print(f'max pr_auc_samples_{group_name}:', np.max(metrics_dict[f'pr_auc_samples_{group_name}']))
            print(f'min pr_auc_samples_{group_name}:', np.min(metrics_dict[f'pr_auc_samples_{group_name}']))
            print(f'CI pr_auc_samples_{group_name}:',
                  calculate_confidence_interval(metrics_dict[f'pr_auc_samples_{group_name}']))
            print()

            print(f'mean roc_auc_samples_{group_name}:', np.mean(metrics_dict[f'roc_auc_samples_{group_name}']))
            print(f'max roc_auc_samples_{group_name}:', np.max(metrics_dict[f'roc_auc_samples_{group_name}']))
            print(f'min roc_auc_samples_{group_name}:', np.min(metrics_dict[f'roc_auc_samples_{group_name}']))
            print(f'CI roc_auc_samples_{group_name}:',
                  calculate_confidence_interval(metrics_dict[f'roc_auc_samples_{group_name}']))
            print()

        for k in list_top_k:
            print('------------------------------------------')

            print(f'mean acc_at_k={k}:', np.mean(metrics_dict[f'acc_at_k={k}']))
            print(f'max acc_at_k={k}:', np.max(metrics_dict[f'acc_at_k={k}']))
            print(f'min acc_at_k={k}:', np.min(metrics_dict[f'acc_at_k={k}']))
            print(f'CI acc_at_k={k}:', calculate_confidence_interval(metrics_dict[f'acc_at_k={k}']))
            print()

            print(f'mean hit_at_k={k}:', np.mean(metrics_dict[f'hit_at_k={k}']))
            print(f'max hit_at_k={k}:', np.max(metrics_dict[f'hit_at_k={k}']))
            print(f'min hit_at_k={k}:', np.min(metrics_dict[f'hit_at_k={k}']))
            print(f'CI hit_at_k={k}:', calculate_confidence_interval(metrics_dict[f'hit_at_k={k}']))
            print()

            for group_name in groups1:
                print(f'mean Group_acc_at_k={k}@{group_name}:',
                      np.mean(metrics_dict[f'Group_acc_at_k={k}@' + group_name]))
                print(f'max Group_acc_at_k={k}@{group_name}:',
                      np.max(metrics_dict[f'Group_acc_at_k={k}@' + group_name]))
                print(f'min Group_acc_at_k={k}@{group_name}:',
                      np.min(metrics_dict[f'Group_acc_at_k={k}@' + group_name]))
                print(f'CI Group_acc_at_k={k}@{group_name}:',
                      calculate_confidence_interval(metrics_dict[f'Group_acc_at_k={k}@' + group_name]))
                print()

                print(f'mean Group_hit_at_k={k}@{group_name}:',
                      np.mean(metrics_dict[f'Group_hit_at_k={k}@' + group_name]))
                print(f'max Group_hit_at_k={k}@{group_name}:',
                      np.max(metrics_dict[f'Group_hit_at_k={k}@' + group_name]))
                print(f'min Group_hit_at_k={k}@{group_name}:',
                      np.min(metrics_dict[f'Group_hit_at_k={k}@' + group_name]))
                print(f'CI Group_hit_at_k={k}@{group_name}:',
                      calculate_confidence_interval(metrics_dict[f'Group_hit_at_k={k}@' + group_name]))
                print()

    if record_results:
        with open(f'results_prompting/metrics_results_BestModel_OntoFAR_{ds_size_ratio}.txt', 'w') as file:
            file.write('\n')
            file.write(f'mean pr_auc_samples: {np.mean(metrics_dict["pr_auc_samples"])}\n')
            file.write(f'max pr_auc_samples: {np.max(metrics_dict["pr_auc_samples"])}\n')
            file.write(f'min pr_auc_samples: {np.min(metrics_dict["pr_auc_samples"])}\n')
            file.write(f'CI pr_auc_samples: {calculate_confidence_interval(metrics_dict["pr_auc_samples"])}\n')
            file.write('\n')

            file.write(f'mean roc_auc_samples: {np.mean(metrics_dict["roc_auc_samples"])}\n')
            file.write(f'max roc_auc_samples: {np.max(metrics_dict["roc_auc_samples"])}\n')
            file.write(f'min roc_auc_samples: {np.min(metrics_dict["roc_auc_samples"])}\n')
            file.write(f'CI roc_auc_samples: {calculate_confidence_interval(metrics_dict["roc_auc_samples"])}\n')
            file.write('\n')

            file.write(f'mean f1_samples: {np.mean(metrics_dict["f1_samples"])}\n')
            file.write(f'max f1_samples: {np.max(metrics_dict["f1_samples"])}\n')
            file.write(f'min f1_samples: {np.min(metrics_dict["f1_samples"])}\n')
            file.write(f'CI f1_samples: {calculate_confidence_interval(metrics_dict["f1_samples"])}\n')
            file.write('\n')

            for group_name in groups1:
                file.write('\n')
                file.write(
                    f'mean pr_auc_samples_{group_name}: {np.mean(metrics_dict[f"pr_auc_samples_{group_name}"])}\n')
                file.write(
                    f'max pr_auc_samples_{group_name}: {np.max(metrics_dict[f"pr_auc_samples_{group_name}"])}\n')
                file.write(
                    f'min pr_auc_samples_{group_name}: {np.min(metrics_dict[f"pr_auc_samples_{group_name}"])}\n')
                file.write(
                    f'CI pr_auc_samples_{group_name}: {calculate_confidence_interval(metrics_dict[f"pr_auc_samples_{group_name}"])}\n')
                file.write('\n')

                file.write(
                    f'mean roc_auc_samples_{group_name}: {np.mean(metrics_dict[f"roc_auc_samples_{group_name}"])}\n')
                file.write(
                    f'max roc_auc_samples_{group_name}: {np.max(metrics_dict[f"roc_auc_samples_{group_name}"])}\n')
                file.write(
                    f'min roc_auc_samples_{group_name}: {np.min(metrics_dict[f"roc_auc_samples_{group_name}"])}\n')
                file.write(
                    f'CI roc_auc_samples_{group_name}: {calculate_confidence_interval(metrics_dict[f"roc_auc_samples_{group_name}"])}\n')
                file.write('\n')

            for k in list_top_k:
                file.write('------------------------------------------\n')

                file.write(f'mean acc_at_k={k}: {np.mean(metrics_dict[f"acc_at_k={k}"])}\n')
                file.write(f'max acc_at_k={k}: {np.max(metrics_dict[f"acc_at_k={k}"])}\n')
                file.write(f'min acc_at_k={k}: {np.min(metrics_dict[f"acc_at_k={k}"])}\n')
                file.write(f'CI acc_at_k={k}: {calculate_confidence_interval(metrics_dict[f"acc_at_k={k}"])}\n')
                file.write('\n')

                file.write(f'mean hit_at_k={k}: {np.mean(metrics_dict[f"hit_at_k={k}"])}\n')
                file.write(f'max hit_at_k={k}: {np.max(metrics_dict[f"hit_at_k={k}"])}\n')
                file.write(f'min hit_at_k={k}: {np.min(metrics_dict[f"hit_at_k={k}"])}\n')
                file.write(f'CI hit_at_k={k}: {calculate_confidence_interval(metrics_dict[f"hit_at_k={k}"])}\n')
                file.write('\n')

                for group_name in groups1:
                    file.write(
                        f'mean Group_acc_at_k={k}@{group_name}: {np.mean(metrics_dict[f"Group_acc_at_k={k}@" + group_name])}\n')
                    file.write(
                        f'max Group_acc_at_k={k}@{group_name}: {np.max(metrics_dict[f"Group_acc_at_k={k}@" + group_name])}\n')
                    file.write(
                        f'min Group_acc_at_k={k}@{group_name}: {np.min(metrics_dict[f"Group_acc_at_k={k}@" + group_name])}\n')
                    file.write(
                        f'CI Group_acc_at_k={k}@{group_name}: {calculate_confidence_interval(metrics_dict[f"Group_acc_at_k={k}@" + group_name])}\n')
                    file.write('\n')

                    file.write('------------------------------------------\n')

                    file.write(
                        f'mean Group_hit_at_k={k}@{group_name}: {np.mean(metrics_dict[f"Group_hit_at_k={k}@" + group_name])}\n')
                    file.write(
                        f'max Group_hit_at_k={k}@{group_name}: {np.max(metrics_dict[f"Group_hit_at_k={k}@" + group_name])}\n')
                    file.write(
                        f'min Group_hit_at_k={k}@{group_name}: {np.min(metrics_dict[f"Group_hit_at_k={k}@" + group_name])}\n')
                    file.write(
                        f'CI Group_hit_at_k={k}@{group_name}: {calculate_confidence_interval(metrics_dict[f"Group_hit_at_k={k}@" + group_name])}\n')
                    file.write('\n')

    return


mimic3_ds = MIMIC3Dataset(
    root="datasets/MIMIC_III/",
    tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
    # map all NDC codes to ATC 3-rd level codes in these tables
    code_mapping={
        "NDC": ("ATC", {"target_kwargs": {"level": 4}})},
)
print('--mimic-III loaded.')
mimic3sample = customized_set_task_mimic3(dataset=mimic3_ds,
                                          task_fn=sequential_diagnosis_prediction_mimic3,
                                          ccs_label=False,
                                          ds_size_ratio=1.0,
                                          seed=45)
print('--datasets created.')
print(mimic3sample.stat())
nfold_experiment(mimic3sample, epochs=230, ds_size_ratio=1.0)

