import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# ------------------------------- helpers ---------------------------------

def remove_exact_prefix(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Remove an exact prefix from every column name."""
    df = df.copy()
    df.columns = df.columns.str.removeprefix(prefix)
    return df


def get_contrast_diff(df: pd.DataFrame, contrast_prefix: str) -> pd.DataFrame:
    """
    Given a contrast block with 'original_' and 'mirrored_' halves,
    return original - mirrored for non-shape features.
    """
    total_columns = df.shape[1]
    midpoint = total_columns // 2

    original_df = df.iloc[:, :midpoint].copy()
    mirrored_df = df.iloc[:, midpoint:].copy()

    # strip exact prefixes so names align before subtraction
    original_df = remove_exact_prefix(original_df, f"{contrast_prefix}_")
    original_df = remove_exact_prefix(original_df, "original_")
    mirrored_df = remove_exact_prefix(mirrored_df, f"{contrast_prefix}_")
    mirrored_df = remove_exact_prefix(mirrored_df, "mirrored_")

    diff = original_df - mirrored_df
    # drop all columns containing 'shape'
    diff.drop(columns=diff.filter(like="shape").columns, inplace=True)
    return diff


def get_diff_EGD(df: pd.DataFrame) -> pd.DataFrame:
    """
    EGD file has T1|T2|FLAIR blocks laid out consecutively.
    Build difference (original - mirrored) blocks for each contrast.
    """
    total_columns = df.shape[1]
    splitpoint = total_columns // 3

    df_t1 = df.iloc[:, :splitpoint]
    df_t2 = df.iloc[:, splitpoint : 2 * splitpoint]
    df_flair = df.iloc[:, 2 * splitpoint :]

    diff_t1 = get_contrast_diff(df_t1, "T1").add_prefix("T1_difference_")
    diff_t2 = get_contrast_diff(df_t2, "T2").add_prefix("T2_difference_")
    diff_flair = get_contrast_diff(df_flair, "FLAIR").add_prefix("FLAIR_difference_")

    return pd.concat([diff_t1, diff_t2, diff_flair], axis=1)


# ------------------------------- loaders ----------------------------------

def get_EGD() -> pd.DataFrame:
    """Load EGD features + clinical/labels and augment with difference + border features."""
    df_EGD = pd.read_csv("radiomics_embeddings/t1t2flair_mirror_radiomics_features_EGD.csv")
    cd_df = pd.read_excel("../tabular_data/Clinical_data.xlsx")
    labels_df = pd.read_excel("../tabular_data/Genetic_and_Histological_labels.xlsx")

    # unify id key
    df_EGD["Subject"] = df_EGD["Patient"].apply(lambda x: f"EGD-{int(x[-4:]):04d}")

    merged = (
        df_EGD.merge(cd_df, on="Subject", how="inner")
              .merge(labels_df, on="Subject", how="inner")
              .drop(columns=["Patient", "Subject"])
    )

    # build difference features from all non-metadata cols (last 5 are clinical/labels)
    feats_only = merged.iloc[:, :-5]
    difference_df = get_diff_EGD(feats_only)

    df_border_EGD = pd.read_csv("radiomics_embeddings/border_features_EGD.csv")

    merged = pd.concat([feats_only, difference_df, df_border_EGD, merged.iloc[:, -5:]], axis=1)

    # quick label stats
    label_pos_sum = merged.loc[merged["1p19q"] != -1, "1p19q"].sum()
    idh_pos_sum = merged.loc[merged["IDH"] != -1, "IDH"].sum()
    label_sum = (merged["1p19q"] != -1).sum()
    idh_sum = (merged["IDH"] != -1).sum()
    print(f"EGD: total 1p19q = {label_sum} | 1p19q+ = {label_pos_sum} | pos rate = {round(label_pos_sum/label_sum, 3)}")
    print(f"EGD: total idh = {idh_sum} | idh+ = {idh_pos_sum} | pos rate = {round(idh_pos_sum/idh_sum, 3)}")

    return merged


def get_PDGM(clahe: bool = False) -> pd.DataFrame:
    """Load PDGM features + labels (optionally CLAHE-preprocessed features)."""
    if clahe:
        df = pd.read_csv("radiomics_embeddings/clahe_wavelet_radiomics_features_PDGM.csv")
    else:
        df = pd.read_csv("radiomics_embeddings/wavelet_radiomics_features_PDGM.csv")

    labels_df = pd.read_excel("radiomics_embeddings/UCSF-PDGM-Genetic_and_Histological_labels_infered.xlsx")

    df["ID"] = df["Patient_ID"].apply(lambda x: f"UCSF-PDGM{int(x[-4:]):04d}")

    merged = (
        df.merge(labels_df, on="ID", how="inner")
          .drop(
              columns=[
                  "Patient_ID",
                  "Final pathologic diagnosis (WHO 2021)",
                  "MGMT status",
                  "MGMT index",
                  "1-dead 0-alive",
                  "OS",
                  "EOR",
                  "Biopsy prior to imaging",
                  "BraTS21 ID",
                  "BraTS21 Segmentation Cohort",
                  "BraTS21 MGMT Cohort",
                  "old_IDH",
                  "old_1p/19q",
                  "ID",
                  "1p/19q",
              ],
              errors="ignore",
          )
          .rename(
              columns={
                  "WHO CNS Grade": "Grade",
                  "1p19q_infered": "1p19q",
                  "Age at MRI": "Age",
              }
          )
    )

    # quick label stats
    label_pos_sum = merged.loc[merged["1p19q"] != -1, "1p19q"].sum()
    idh_pos_sum = merged.loc[merged["IDH"] != -1, "IDH"].sum()
    label_sum = (merged["1p19q"] != -1).sum()
    idh_sum = (merged["IDH"] != -1).sum()
    print(f"PDGM: total 1p19q = {label_sum} | 1p19q+ = {label_pos_sum} | pos rate = {round(label_pos_sum/label_sum, 3)}")
    print(f"PDGM: total idh = {idh_sum} | idh+ = {idh_pos_sum} | pos rate = {round(idh_pos_sum/idh_sum, 3)}")

    return merged


def get_LGG(clahe: bool = False) -> pd.DataFrame:
    """Load LGG features + labels/clinical (optionally CLAHE-preprocessed features)."""
    if clahe:
        df = pd.read_csv("radiomics_embeddings/clahe_radiomics_features_LGG.csv")
    else:
        df = pd.read_csv("radiomics_embeddings/radiomics_features_LGG.csv")

    labels_df = pd.read_excel("../tabular_data/TCIA_LGG_cases_159.xlsx")
    clinical_data_df = pd.read_excel("../tabular_data/TCIA_LGG_Clinical_data.xlsx")

    df = df.rename(columns={"Image": "Patient_ID"})
    labels_df = labels_df.rename(columns={"Filename": "Patient_ID"})

    merged = (
        df.merge(labels_df, on="Patient_ID", how="inner")
          .merge(clinical_data_df, on="Patient_ID", how="inner")
          .drop(columns=["Patient_ID"])
    )

    # recode, infer IDH, and tidy
    merged["1p/19q"] = merged["1p/19q"].map({"d/d": 1, "n/n": 0})
    merged["IDH"] = -1
    merged.loc[(merged["Type"] == "Oligoastrocytoma") & (merged["1p/19q"] == 1), "Type"] = "Oligodendroglioma"
    merged.loc[(merged["Type"] == "Oligoastrocytoma") & (merged["1p/19q"] == 0), "Type"] = "Astrocytoma"
    merged.loc[merged["Type"] == "Astrocytoma", "IDH"] = 1
    merged.loc[merged["Type"] == "Oligodendroglioma", "IDH"] = 1
    merged = merged.rename(columns={"1p/19q": "1p19q"}).drop(columns=["Type"])

    # quick label stats
    label_pos_sum = merged.loc[merged["1p19q"] != -1, "1p19q"].sum()
    idh_pos_sum = merged.loc[merged["IDH"] != -1, "IDH"].sum()
    label_sum = (merged["1p19q"] != -1).sum()
    idh_sum = (merged["IDH"] != -1).sum()
    print(f"LGG: total 1p19q = {label_sum} | 1p19q+ = {label_pos_sum} | pos rate = {round(label_pos_sum/label_sum, 3)}")
    print(f"LGG: total idh = {idh_sum} | idh+ = {idh_pos_sum} | pos rate = {round(idh_pos_sum/idh_sum, 3)}")

    return merged


# ------------------------------ cleaning ----------------------------------

def sort_non_radiomic_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Move core clinical/label columns to the end for convenience."""
    cols = ["Sex", "Age", "IDH", "Label", "Grade", "Class"]
    last = df[cols]
    remaining = df.drop(columns=cols)
    return pd.concat([remaining, last], axis=1)


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Basic clinical cleaning + derived Class label."""
    df = df.copy()

    # Sex: impute mode (excluding -1) then map to {F:0, M:1}
    most_common_sex = df.loc[df["Sex"] != -1, "Sex"].mode()[0]
    df["Sex"] = df["Sex"].replace(-1, most_common_sex).map({"F": 0, "M": 1})

    # Age: impute mean of known ages
    mean_age = df.loc[df["Age"] != -1, "Age"].mean()
    df["Age"] = df["Age"].replace(-1, mean_age)

    # Rename 1p19q → Label for convenience
    df = df.rename(columns={"1p19q": "Label"})

    # Build a coarse tumor class
    conditions = [
        df["Grade"] == 4,
        ((df["Label"] == 0) & (df["IDH"] == 1) & (df["Grade"] < 4))
        | ((df["IDH"] == 0) & (df["Grade"] < 4)),
        (df["Label"] == 1) & (df["IDH"] == 1),
    ]
    choices = ["Glioblastoma or other", "Astrocytoma", "Oligodendroglioma"]
    df["Class"] = np.select(conditions, choices, default="Unknown")

    return sort_non_radiomic_columns(df)


# ------------------------------ main entry --------------------------------

def get_data(
    experiment: str,
    test_size: float,
    random_state: int,
    clahe: bool = False,
):
    """
    Load + clean data per experiment string.
      - 'EGD_IDH' or 'EGD_COD' → EGD only
      - 'PDGM'                 → PDGM only
      - 'LGG'                  → LGG only
      - 'PDGM_TEST_EGD'        → train on PDGM, test on EGD
      - 'TRAIN_EGD_PDGM'       → train=EGD(+split) + PDGM, test=held-out EGD split
    """
    if experiment in ("EGD_IDH", "EGD_COD"):
        merged_df = clean_df(get_EGD())

    elif experiment == "PDGM":
        merged_df = clean_df(get_PDGM(clahe=clahe))

    elif experiment == "LGG":
        merged_df = clean_df(get_LGG(clahe=clahe))

    elif experiment in ("PDGM_TEST_EGD", "TRAIN_EGD_PDGM"):
        merged_EGD = clean_df(get_EGD())
        merged_PDGM = clean_df(get_PDGM(clahe=clahe))

    else:
        raise ValueError(f"Unknown experiment: {experiment}")

    # ------------------------- splitting strategies -------------------------
    if experiment in ("EGD_IDH", "EGD_COD", "PDGM", "LGG"):
        train_df, test_df = train_test_split(
            merged_df,
            test_size=test_size,
            random_state=random_state,
            stratify=merged_df["Label"],
        )

    elif experiment == "PDGM_TEST_EGD":
        train_df = merged_PDGM
        test_df = merged_EGD

    elif experiment == "TRAIN_EGD_PDGM":
        train_df, test_df = train_test_split(
            merged_EGD,
            test_size=test_size,
            random_state=random_state,
            stratify=merged_EGD["Label"],
        )
        # add all PDGM cases to the training side
        train_df = pd.concat([train_df, merged_PDGM], ignore_index=True, axis=0)

    return train_df, test_df
