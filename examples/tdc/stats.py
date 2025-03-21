import pandas as pd

def get_meta_info(df_train: pd.DataFrame,
                  df_valid: pd.DataFrame,
                  df_test: pd.DataFrame):
    """
    Get meta information about the datasets.
    """
    
    dataset_names = [c.lower() for c in df_train.columns if c != 'smi']

    df_meta = pd.DataFrame(index=dataset_names, columns=['training', 'validation', 'test'])
    
    for d in dataset_names:
        df_meta.loc[d, 'task']       = 'regression' if len(df_train[d].dropna().unique()) > 2 else 'binary_classification'
        df_meta.loc[d, 'training']   = df_train[d].notna().sum()
        df_meta.loc[d, 'validation'] = df_valid[d].notna().sum()
        df_meta.loc[d, 'test']       = df_test[d].notna().sum()

    df_meta.loc['TOTAL', 'training']   = df_train.notna().sum().sum()
    df_meta.loc['TOTAL', 'validation'] = df_valid.notna().sum().sum()
    df_meta.loc['TOTAL', 'test']       = df_test.notna().sum().sum()

    print("Molecule counts per dataset:")
    print(df_meta)

    def overlap_matrix(df, cols):
        mat = pd.DataFrame(index=cols, columns=cols, dtype=int)
        for i in cols:
            for j in cols:
                mat.loc[i, j] = ((df[i].notna()) & (df[j].notna())).sum()
        return mat

    train_overlap = overlap_matrix(df_train, dataset_names)
    valid_overlap = overlap_matrix(df_valid, dataset_names)
    test_overlap  = overlap_matrix(df_test,  dataset_names)

    print("\nTraining split overlap:")
    print(train_overlap)
    print("\nValidation split overlap:")
    print(valid_overlap)
    print("\nTest split overlap:")
    print(test_overlap)

    label_cols = df_train.columns.drop('smi')
    sparsity_train = df_train[label_cols].isna().sum().sum() / (df_train.shape[0] * len(label_cols))
    sparsity_valid = df_valid[label_cols].isna().sum().sum() / (df_valid.shape[0] * len(label_cols))
    sparsity_test  = df_test[label_cols].isna().sum().sum()  / (df_test.shape[0] * len(label_cols))

    print(f"Sparsity of training set: {sparsity_train:.2%}")
    print(f"Sparsity of validation set: {sparsity_valid:.2%}")
    print(f"Sparsity of test set: {sparsity_test:.2%}")

