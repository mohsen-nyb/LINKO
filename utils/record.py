import numpy as np

def print_results(metrics_dict, list_top_k, groups):
    print()
    print('mean pr_auc_samples:', np.mean(metrics_dict['pr_auc_samples']))
    print('max pr_auc_samples:', np.max(metrics_dict['pr_auc_samples']))
    print('min pr_auc_samples:', np.min(metrics_dict['pr_auc_samples']))
    print()

    print('mean roc_auc_samples:', np.mean(metrics_dict['roc_auc_samples']))
    print('max roc_auc_samples:', np.max(metrics_dict['roc_auc_samples']))
    print('min roc_auc_samples:', np.min(metrics_dict['roc_auc_samples']))
    print()

    print('mean f1_samples:', np.mean(metrics_dict['f1_samples']))
    print('max f1_samples:', np.max(metrics_dict['f1_samples']))
    print('min f1_samples:', np.min(metrics_dict['f1_samples']))
    print()

    for k in list_top_k:
        print('------------------------------------------')

        print(f'mean acc_at_k={k}:', np.mean(metrics_dict[f'acc_at_k={k}']))
        print(f'max acc_at_k={k}:', np.max(metrics_dict[f'acc_at_k={k}']))
        print(f'min acc_at_k={k}:', np.min(metrics_dict[f'acc_at_k={k}']))
        print()

        for group_name in groups:

            print(f'mean Group_acc_at_k={k}@0-25:', np.mean(metrics_dict[f'Group_acc_at_k={k}@' + group_name]))
            print(f'max Group_acc_at_k={k}@0-25:', np.max(metrics_dict[f'Group_acc_at_k={k}@' + group_name]))
            print(f'min Group_acc_at_k={k}@0-25:', np.min(metrics_dict[f'Group_acc_at_k={k}@' + group_name]))
            print()


# Open a file in write mode
def record_results(name, metrics_dict, list_top_k, groups):
    with open(name, 'w') as file:
        file.write('\n')
        file.write(f'mean pr_auc_samples: {np.mean(metrics_dict["pr_auc_samples"])}\n')
        file.write(f'max pr_auc_samples: {np.max(metrics_dict["pr_auc_samples"])}\n')
        file.write(f'min pr_auc_samples: {np.min(metrics_dict["pr_auc_samples"])}\n')
        file.write('\n')

        file.write(f'mean roc_auc_samples: {np.mean(metrics_dict["roc_auc_samples"])}\n')
        file.write(f'max roc_auc_samples: {np.max(metrics_dict["roc_auc_samples"])}\n')
        file.write(f'min roc_auc_samples: {np.min(metrics_dict["roc_auc_samples"])}\n')
        file.write('\n')

        file.write(f'mean f1_samples: {np.mean(metrics_dict["f1_samples"])}\n')
        file.write(f'max f1_samples: {np.max(metrics_dict["f1_samples"])}\n')
        file.write(f'min f1_samples: {np.min(metrics_dict["f1_samples"])}\n')
        file.write('\n')

        for k in list_top_k:
            file.write('------------------------------------------\n')

            file.write(f'mean acc_at_k={k}: {np.mean(metrics_dict[f"acc_at_k={k}"])}\n')
            file.write(f'max acc_at_k={k}: {np.max(metrics_dict[f"acc_at_k={k}"])}\n')
            file.write(f'min acc_at_k={k}: {np.min(metrics_dict[f"acc_at_k={k}"])}\n')
            file.write('\n')

            for group_name in groups:

                file.write(f'mean Group_acc_at_k={k}@0-25: {np.mean(metrics_dict[f"Group_acc_at_k={k}@" + group_name])}\n')
                file.write(f'max Group_acc_at_k={k}@0-25: {np.max(metrics_dict[f"Group_acc_at_k={k}@" + group_name])}\n')
                file.write(f'min Group_acc_at_k={k}@0-25: {np.min(metrics_dict[f"Group_acc_at_k={k}@" + group_name])}\n')
                file.write('\n')

