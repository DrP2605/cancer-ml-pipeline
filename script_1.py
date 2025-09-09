# Complete End-to-End ML Pipeline for Cancer Survival and Drug Response Prediction
# Optimized for standard scientific Python libraries
# Author: AI Research Assistant
# Date: September 2025

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, ElasticNet, Ridge
from sklearn.svm import SVC, SVR
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.metrics import (accuracy_score, roc_auc_score, mean_squared_error, 
                           mean_absolute_error, r2_score, classification_report,
                           confusion_matrix, precision_recall_curve)
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)

print("="*80)
print("COMPLETE ML PIPELINE FOR CANCER SURVIVAL & DRUG RESPONSE PREDICTION")
print("="*80)

# ============================================================================
# 1. DATA GENERATION AND LOADING
# ============================================================================

def generate_comprehensive_cancer_data(n_samples=2000, n_genes=1000, n_mutations=30):
    """Generate comprehensive synthetic cancer dataset"""
    
    print(f"\n1. GENERATING COMPREHENSIVE CANCER DATASET")
    print("-" * 60)
    
    # Sample IDs
    sample_ids = [f'TCGA-{i:05d}' for i in range(n_samples)]
    
    # Cancer types with realistic distribution
    cancer_types = np.random.choice(
        ['BRCA', 'LUAD', 'COAD', 'LGG', 'KIRC', 'HNSC', 'THCA', 'PRAD', 'SKCM', 'UCEC'], 
        n_samples, 
        p=[0.15, 0.12, 0.10, 0.10, 0.10, 0.08, 0.08, 0.08, 0.08, 0.11]
    )
    
    print(f"✓ Generated {n_samples} samples across 10 cancer types")
    
    # ========== RNA-seq Expression Data ==========
    gene_names = [f'GENE_{i+1:04d}' for i in range(n_genes)]
    rna_data = {'sample_id': sample_ids, 'cancer_type': cancer_types}
    
    for i, gene in enumerate(gene_names):
        # Base expression with log-normal distribution
        base_expression = np.random.lognormal(mean=4, sigma=1.8, size=n_samples)
        
        # Add cancer-type specific signatures
        if i < 100:  # First 100 genes are highly cancer-type specific
            cancer_multipliers = {
                'BRCA': np.random.uniform(1.5, 3.0) if i < 20 else 1.0,
                'LUAD': np.random.uniform(1.5, 2.5) if 20 <= i < 40 else 1.0,
                'COAD': np.random.uniform(1.3, 2.2) if 40 <= i < 60 else 1.0,
                'LGG': np.random.uniform(0.3, 0.7) if 60 <= i < 80 else 1.0,
                'KIRC': np.random.uniform(1.2, 2.0) if 80 <= i < 100 else 1.0
            }
            
            for j, cancer_type in enumerate(cancer_types):
                if cancer_type in cancer_multipliers:
                    base_expression[j] *= cancer_multipliers[cancer_type]
        
        # Add noise and outliers
        noise = np.random.normal(0, 0.1, n_samples)
        outlier_mask = np.random.random(n_samples) < 0.02  # 2% outliers
        outliers = np.random.lognormal(8, 1, n_samples) * outlier_mask
        
        final_expression = base_expression * (1 + noise) + outliers
        rna_data[gene] = np.log2(final_expression + 1)  # log2(TPM + 1)
    
    rna_df = pd.DataFrame(rna_data)
    print(f"✓ Generated RNA-seq data: {n_genes} genes")
    
    # ========== Mutation Data ==========
    # Key cancer genes with realistic mutation frequencies
    key_genes = [
        'TP53', 'KRAS', 'PIK3CA', 'BRAF', 'PTEN', 'APC', 'EGFR', 'IDH1', 
        'CDKN2A', 'RB1', 'VHL', 'ARID1A', 'CTNNB1', 'SMAD4', 'ATM',
        'BRCA1', 'BRCA2', 'MLH1', 'MSH2', 'STK11', 'LKB1', 'NF1', 
        'NOTCH1', 'FBXW7', 'FGFR3', 'PIK3R1', 'NRAS', 'HRAS', 'MET', 'ALK'
    ]
    
    mutation_data = {'sample_id': sample_ids}
    
    for gene in key_genes[:n_mutations]:
        # Base mutation frequencies
        base_freq = {
            'TP53': 0.50, 'KRAS': 0.30, 'PIK3CA': 0.25, 'BRAF': 0.20, 'PTEN': 0.15,
            'APC': 0.35, 'EGFR': 0.18, 'IDH1': 0.12, 'CDKN2A': 0.16, 'RB1': 0.08
        }
        
        freq = base_freq.get(gene, 0.10)
        
        # Cancer-type specific mutation rates
        cancer_specific_rates = {
            'COAD': {'APC': 0.8, 'KRAS': 0.5, 'TP53': 0.6},
            'BRCA': {'PIK3CA': 0.35, 'TP53': 0.3, 'BRCA1': 0.15, 'BRCA2': 0.12},
            'LUAD': {'KRAS': 0.35, 'TP53': 0.45, 'EGFR': 0.25, 'BRAF': 0.05},
            'LGG': {'IDH1': 0.75, 'TP53': 0.3, 'ARID1A': 0.2},
            'SKCM': {'BRAF': 0.6, 'NRAS': 0.25, 'TP53': 0.15}
        }
        
        mutations = np.zeros(n_samples)
        for j, cancer_type in enumerate(cancer_types):
            if cancer_type in cancer_specific_rates and gene in cancer_specific_rates[cancer_type]:
                prob = cancer_specific_rates[cancer_type][gene]
            else:
                prob = freq
            
            mutations[j] = np.random.binomial(1, prob)
        
        mutation_data[f'{gene}_mutation'] = mutations.astype(int)
    
    mutation_df = pd.DataFrame(mutation_data)
    print(f"✓ Generated mutation data: {n_mutations} key cancer genes")
    
    # ========== Clinical Survival Data ==========
    # Age distribution
    age = np.random.normal(65, 15, n_samples)
    age = np.clip(age, 20, 95)
    
    # Stage distribution
    stage = np.random.choice(['I', 'II', 'III', 'IV'], n_samples, p=[0.15, 0.25, 0.35, 0.25])
    
    # Performance status
    performance_status = np.random.choice([0, 1, 2], n_samples, p=[0.6, 0.3, 0.1])
    
    # Calculate survival based on multiple factors
    base_hazard = np.random.exponential(scale=800, size=n_samples)
    
    # Age effect (older = worse prognosis)
    age_effect = (age - 65) / 20 * 0.3
    
    # Cancer type effect
    cancer_hazard = {
        'LGG': -0.9, 'THCA': -0.7, 'PRAD': -0.5, 'BRCA': -0.3, 'KIRC': -0.1,
        'HNSC': 0.2, 'COAD': 0.1, 'UCEC': 0.0, 'LUAD': 0.4, 'SKCM': 0.6
    }
    cancer_effect = np.array([cancer_hazard.get(ct, 0) for ct in cancer_types])
    
    # Stage effect
    stage_hazard = {'I': -0.8, 'II': -0.4, 'III': 0.2, 'IV': 1.2}
    stage_effect = np.array([stage_hazard[s] for s in stage])
    
    # Performance status effect
    ps_effect = performance_status * 0.3
    
    # Molecular signatures effect
    high_risk_genes = [f'GENE_{i+1:04d}' for i in range(20)]  # First 20 genes
    high_risk_score = np.mean([rna_data[gene] for gene in high_risk_genes], axis=0)
    molecular_effect = (high_risk_score - np.mean(high_risk_score)) / np.std(high_risk_score) * 0.4
    
    # Key mutation effects
    tp53_effect = mutation_data.get('TP53_mutation', np.zeros(n_samples)) * 0.5
    oncogene_effect = (
        mutation_data.get('KRAS_mutation', np.zeros(n_samples)) * 0.3 +
        mutation_data.get('PIK3CA_mutation', np.zeros(n_samples)) * 0.2 +
        mutation_data.get('BRAF_mutation', np.zeros(n_samples)) * 0.25
    )
    
    # Combine all hazard effects
    log_hazard = (age_effect + cancer_effect + stage_effect + ps_effect + 
                 molecular_effect + tp53_effect + oncogene_effect)
    
    hazard_ratio = np.exp(log_hazard)
    survival_time = base_hazard / hazard_ratio
    survival_time = np.clip(survival_time, 30, 5000)  # 30 days to ~13.7 years
    
    # Censoring (administrative censoring + loss to follow-up)
    study_end_time = np.random.uniform(1000, 3000, n_samples)  # Study duration varies
    ltfu_prob = 0.15  # 15% lost to follow-up
    lost_to_followup = np.random.random(n_samples) < ltfu_prob
    
    observed_time = np.where(
        lost_to_followup,
        np.random.uniform(100, survival_time),  # Lost to follow-up at random time
        np.minimum(survival_time, study_end_time)  # Administrative censoring
    )
    
    vital_status = (survival_time <= observed_time).astype(int)  # 1=death observed, 0=censored
    
    clinical_data = {
        'sample_id': sample_ids,
        'age': age,
        'stage': stage,
        'performance_status': performance_status,
        'survival_time': observed_time,
        'vital_status': vital_status,
        'high_risk_score': high_risk_score
    }
    
    clinical_df = pd.DataFrame(clinical_data)
    print(f"✓ Generated clinical data with realistic survival patterns")
    print(f"  - Events observed: {vital_status.sum()} ({vital_status.mean()*100:.1f}%)")
    print(f"  - Median survival: {np.median(observed_time):.0f} days")
    
    # ========== Drug Response Data ==========
    drugs = [
        'Cisplatin', 'Paclitaxel', 'Temozolomide', 'Erlotinib', 'Sorafenib', 
        'Gemcitabine', 'Docetaxel', '5-Fluorouracil', 'Carboplatin', 'Oxaliplatin',
        'Gefitinib', 'Imatinib', 'Sunitinib', 'Bevacizumab', 'Cetuximab'
    ]
    
    drug_response_data = {'sample_id': sample_ids}
    
    for drug in drugs:
        # Base IC50 values (log scale)
        base_ic50 = np.random.normal(1.5, 1.2, n_samples)
        
        # Drug-specific molecular effects
        drug_effects = np.zeros(n_samples)
        
        # Targeted therapy effects
        if drug == 'Erlotinib' or drug == 'Gefitinib':  # EGFR inhibitors
            egfr_mut = mutation_data.get('EGFR_mutation', np.zeros(n_samples))
            drug_effects -= egfr_mut * 1.5  # Sensitive if EGFR mutated
            
        elif drug == 'Imatinib':  # BCR-ABL inhibitor (proxy effect)
            drug_effects -= np.random.binomial(1, 0.05, n_samples) * 2.0  # Rare but strong effect
            
        elif drug in ['Cisplatin', 'Carboplatin', 'Oxaliplatin']:  # DNA damaging platinum agents
            tp53_mut = mutation_data.get('TP53_mutation', np.zeros(n_samples))
            brca_mut = (mutation_data.get('BRCA1_mutation', np.zeros(n_samples)) + 
                       mutation_data.get('BRCA2_mutation', np.zeros(n_samples)))
            drug_effects += tp53_mut * 0.6  # Resistant if TP53 mutated
            drug_effects -= brca_mut * 0.8  # Sensitive if BRCA mutated (synthetic lethality)
            
        elif drug == 'Sorafenib' or drug == 'Sunitinib':  # Multi-kinase inhibitors
            # Effective in certain cancer types
            cancer_sensitivity = np.where(cancer_types == 'KIRC', -0.8,  # Sensitive in kidney cancer
                                np.where(cancer_types == 'SKCM', -0.4, 0.2))  # Moderately effective in melanoma
            drug_effects += cancer_sensitivity
            
        # Expression-based effects
        if drug in ['Paclitaxel', 'Docetaxel']:  # Taxanes - microtubule inhibitors
            # High expression of certain genes indicates resistance
            resistance_signature = np.mean([rna_data[f'GENE_{i+1:04d}'] for i in range(50, 70)], axis=0)
            drug_effects += (resistance_signature - np.mean(resistance_signature)) / np.std(resistance_signature) * 0.4
        
        # Cancer type effects
        cancer_drug_effects = {
            'BRCA': {'Cisplatin': -0.3, 'Paclitaxel': -0.2},
            'LUAD': {'Erlotinib': -0.5, 'Gefitinib': -0.4, 'Cisplatin': 0.1},
            'COAD': {'5-Fluorouracil': -0.4, 'Oxaliplatin': -0.3, 'Cetuximab': -0.2},
            'SKCM': {'Sorafenib': -0.3},
            'KIRC': {'Sorafenib': -0.8, 'Sunitinib': -0.7}
        }
        
        for j, cancer_type in enumerate(cancer_types):
            if cancer_type in cancer_drug_effects and drug in cancer_drug_effects[cancer_type]:
                drug_effects[j] += cancer_drug_effects[cancer_type][drug]
        
        # Final IC50 calculation with noise
        final_ic50 = base_ic50 + drug_effects + np.random.normal(0, 0.4, n_samples)
        drug_response_data[f'{drug}_IC50'] = final_ic50
    
    drug_df = pd.DataFrame(drug_response_data)
    print(f"✓ Generated drug response data: {len(drugs)} compounds")
    
    return rna_df, mutation_df, clinical_df, drug_df

# ============================================================================
# 2. ADVANCED DATA PREPROCESSING
# ============================================================================

def advanced_preprocessing(rna_df, mutation_df, clinical_df, drug_df):
    """Advanced preprocessing with feature selection and engineering"""
    
    print(f"\n2. ADVANCED DATA PREPROCESSING")
    print("-" * 60)
    
    # Merge all datasets
    merged_data = rna_df.merge(mutation_df, on='sample_id')
    merged_data = merged_data.merge(clinical_df, on='sample_id')
    merged_data = merged_data.merge(drug_df, on='sample_id')
    
    print(f"✓ Merged dataset shape: {merged_data.shape}")
    
    # Identify feature types
    gene_cols = [col for col in merged_data.columns if col.startswith('GENE_')]
    mutation_cols = [col for col in merged_data.columns if col.endswith('_mutation')]
    drug_cols = [col for col in merged_data.columns if col.endswith('_IC50')]
    
    print(f"✓ Gene expression features: {len(gene_cols)}")
    print(f"✓ Mutation features: {len(mutation_cols)}")
    print(f"✓ Drug response targets: {len(drug_cols)}")
    
    # Advanced gene expression feature selection
    print(f"\n2.1 Gene Expression Feature Selection")
    
    # Variance-based filtering (remove low-variance genes)
    gene_expression = merged_data[gene_cols]
    gene_variance = gene_expression.var()
    high_var_genes = gene_variance[gene_variance > gene_variance.quantile(0.75)].index.tolist()
    print(f"✓ High variance genes (top 25%): {len(high_var_genes)}")
    
    # Cancer-type differential expression
    differential_genes = []
    for cancer_type in merged_data['cancer_type'].unique():
        if len(merged_data[merged_data['cancer_type'] == cancer_type]) >= 50:  # Sufficient samples
            cancer_samples = merged_data['cancer_type'] == cancer_type
            other_samples = merged_data['cancer_type'] != cancer_type
            
            for gene in high_var_genes[:200]:  # Limit for computational efficiency
                cancer_expr = merged_data.loc[cancer_samples, gene]
                other_expr = merged_data.loc[other_samples, gene]
                
                # Simple t-test equivalent (mean difference scaled by pooled std)
                mean_diff = abs(cancer_expr.mean() - other_expr.mean())
                pooled_std = np.sqrt(((cancer_expr.var() * len(cancer_expr)) + 
                                    (other_expr.var() * len(other_expr))) / 
                                   (len(cancer_expr) + len(other_expr)))
                
                if pooled_std > 0:
                    effect_size = mean_diff / pooled_std
                    if effect_size > 0.5:  # Moderate effect size
                        differential_genes.append(gene)
    
    differential_genes = list(set(differential_genes))
    print(f"✓ Differentially expressed genes: {len(differential_genes)}")
    
    # Combine with most variable genes
    selected_genes = list(set(high_var_genes[:150] + differential_genes))
    print(f"✓ Total selected genes: {len(selected_genes)}")
    
    # Feature engineering
    print(f"\n2.2 Feature Engineering")
    
    # Mutation burden (total mutations per sample)
    merged_data['mutation_burden'] = merged_data[mutation_cols].sum(axis=1)
    
    # Pathway-level features (simplified - group genes into pathways)
    pathway_genes = {
        'DNA_repair': selected_genes[:25],
        'Cell_cycle': selected_genes[25:50],
        'Apoptosis': selected_genes[50:75],
        'Metabolism': selected_genes[75:100],
        'Immune': selected_genes[100:125]
    }
    
    for pathway, genes in pathway_genes.items():
        available_genes = [g for g in genes if g in merged_data.columns]
        if available_genes:
            merged_data[f'{pathway}_signature'] = merged_data[available_genes].mean(axis=1)
    
    # Age groups
    merged_data['age_group'] = pd.cut(merged_data['age'], 
                                    bins=[0, 50, 65, 80, 100], 
                                    labels=['Young', 'Middle', 'Senior', 'Elderly'])
    
    # Create final feature matrix
    clinical_features = ['age', 'performance_status', 'high_risk_score', 'mutation_burden']
    pathway_features = [f'{pw}_signature' for pw in pathway_genes.keys()]
    
    feature_cols = (selected_genes + mutation_cols + clinical_features + 
                   pathway_features + ['cancer_type', 'stage', 'age_group'])
    
    X_features = merged_data[feature_cols].copy()
    
    # Encode categorical variables
    categorical_cols = ['cancer_type', 'stage', 'age_group']
    for col in categorical_cols:
        if col in X_features.columns:
            le = LabelEncoder()
            X_features[f'{col}_encoded'] = le.fit_transform(X_features[col].astype(str))
    
    # Drop original categorical columns
    X_features = X_features.drop(categorical_cols, axis=1, errors='ignore')
    
    print(f"✓ Final feature matrix: {X_features.shape}")
    print(f"✓ Feature categories:")
    print(f"  - Gene expression: {len(selected_genes)}")
    print(f"  - Mutations: {len(mutation_cols)}")
    print(f"  - Clinical: {len(clinical_features)}")
    print(f"  - Pathway signatures: {len(pathway_features)}")
    print(f"  - Encoded categorical: 3")
    
    return merged_data, X_features, selected_genes, mutation_cols, drug_cols

# Save the first part and continue with the rest
print("PART 1/3 LOADED: Data generation and preprocessing functions ready")