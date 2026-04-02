"""Step 1: Filter PrimeKG — keep clinically relevant edges only."""

import argparse
from pathlib import Path

import pandas as pd

EXCLUDE_RELATIONS = {
    'anatomy_protein_present',
    'anatomy_protein_absent',
    'protein_protein',
    'bioprocess_bioprocess',
    'cellcomp_cellcomp',
    'molfunc_molfunc',
    'anatomy_anatomy',
    'phenotype_phenotype',
    'pathway_pathway',
    'exposure_exposure',
    'exposure_molfunc',
    'exposure_cellcomp',
}

KEEP_RELATIONS = {
    # TIER 1: Clinical core
    'disease_phenotype_positive',
    'disease_phenotype_negative',
    'indication',
    'contraindication',
    'off-label use',
    'drug_protein',
    'drug_effect',
    'disease_protein',
    'disease_disease',
    'exposure_disease',
    # TIER 2: Biomedical science
    'bioprocess_protein',
    'molfunc_protein',
    'pathway_protein',
    'cellcomp_protein',
    # TIER 3: Specialty
    'phenotype_protein',
    'exposure_protein',
    'exposure_bioprocess',
}


def filter_kg(kg_path: str, output_path: str) -> pd.DataFrame:
    """Filter PrimeKG to keep only clinically relevant edges.

    1. Keep edges in KEEP_RELATIONS (~845K rows)
    2. For drug_drug (2.67M): keep only pairs where both drugs share
       at least 1 common indicated disease (~15-20K pairs)
    """
    print(f"Loading {kg_path}...")
    kg = pd.read_csv(kg_path, low_memory=False)
    print(f"  Total edges: {len(kg):,}")

    # Keep standard relations
    filtered = kg[kg['relation'].isin(KEEP_RELATIONS)].copy()
    print(f"  After KEEP_RELATIONS filter: {len(filtered):,}")

    # DDI filter: drug_drug edges where both drugs share >= 1 indicated disease
    drug_drug = kg[kg['relation'] == 'drug_drug']
    indication = kg[kg['relation'] == 'indication']
    drug_diseases = indication.groupby('x_index')['y_index'].apply(set).to_dict()

    mask = drug_drug.apply(
        lambda r: bool(
            drug_diseases.get(r['x_index'], set())
            & drug_diseases.get(r['y_index'], set())
        ),
        axis=1,
    )
    ddi_filtered = drug_drug[mask]
    print(f"  DDI edges kept: {len(ddi_filtered):,}")

    filtered = pd.concat([filtered, ddi_filtered], ignore_index=True)
    print(f"  Total filtered edges: {len(filtered):,}")

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    filtered.to_parquet(output_path, index=False)
    print(f"  Saved to {output_path}")

    return filtered


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Filter PrimeKG')
    parser.add_argument('--kg-path', default='PrimeKG/kg.csv')
    parser.add_argument('--output', default='data/filtered_kg.parquet')
    args = parser.parse_args()
    filter_kg(args.kg_path, args.output)
