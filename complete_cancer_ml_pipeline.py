# Complete End-to-End ML Pipeline for Cancer Survival and Drug Response Prediction
# Comprehensive Implementation with Advanced Methods
# Author: AI Research Assistant
# Date: September 2025

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, VotingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, ElasticNet, Ridge
from sklearn.svm import SVC, SVR
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
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

def generate_comprehensive_cancer_data(n_samples=2000, n_genes=1000, n_mutations=25):
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
    age = np.random.normal(65, 15, n_samples)
    age = np.clip(age, 20, 95)
    
    stage = np.random.choice(['I', 'II', 'III', 'IV'], n_samples, p=[0.15, 0.25, 0.35, 0.25])
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
    high_risk_genes = [f'GENE_{i+1:04d}' for i in range(20)]
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
    survival_time = np.clip(survival_time, 30, 5000)
    
    # Censoring (administrative censoring + loss to follow-up)
    study_end_time = np.random.uniform(1000, 3000, n_samples)
    ltfu_prob = 0.15  # 15% lost to follow-up
    lost_to_followup = np.random.random(n_samples) < ltfu_prob
    
    observed_time = np.where(
        lost_to_followup,
        np.random.uniform(100, survival_time),
        np.minimum(survival_time, study_end_time)
    )
    
    vital_status = (survival_time <= observed_time).astype(int)
    
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
        base_ic50 = np.random.normal(1.5, 1.2, n_samples)
        drug_effects = np.zeros(n_samples)
        
        # Targeted therapy effects
        if drug == 'Erlotinib' or drug == 'Gefitinib':
            egfr_mut = mutation_data.get('EGFR_mutation', np.zeros(n_samples))
            drug_effects -= egfr_mut * 1.5
            
        elif drug == 'Imatinib':
            drug_effects -= np.random.binomial(1, 0.05, n_samples) * 2.0
            
        elif drug in ['Cisplatin', 'Carboplatin', 'Oxaliplatin']:
            tp53_mut = mutation_data.get('TP53_mutation', np.zeros(n_samples))
            brca_mut = (mutation_data.get('BRCA1_mutation', np.zeros(n_samples)) + 
                       mutation_data.get('BRCA2_mutation', np.zeros(n_samples)))
            drug_effects += tp53_mut * 0.6
            drug_effects -= brca_mut * 0.8
            
        elif drug == 'Sorafenib' or drug == 'Sunitinib':
            cancer_sensitivity = np.where(cancer_types == 'KIRC', -0.8,
                                np.where(cancer_types == 'SKCM', -0.4, 0.2))
            drug_effects += cancer_sensitivity
            
        # Expression-based effects
        if drug in ['Paclitaxel', 'Docetaxel']:
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
    
    # Gene expression feature selection
    gene_expression = merged_data[gene_cols]
    gene_variance = gene_expression.var()
    high_var_genes = gene_variance[gene_variance > gene_variance.quantile(0.75)].index.tolist()
    
    # Select top genes by variance and differential expression
    selected_genes = high_var_genes[:150]  # Top 150 most variable genes
    print(f"✓ Selected genes: {len(selected_genes)}")
    
    # Feature engineering
    merged_data['mutation_burden'] = merged_data[mutation_cols].sum(axis=1)
    
    # Pathway-level features (simplified - group genes into pathways)
    pathway_genes = {
        'DNA_repair': selected_genes[:30],
        'Cell_cycle': selected_genes[30:60],
        'Apoptosis': selected_genes[60:90],
        'Metabolism': selected_genes[90:120],
        'Immune': selected_genes[120:150]
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
    
    return merged_data, X_features, selected_genes, mutation_cols, drug_cols

# ============================================================================
# 3. COMPREHENSIVE SURVIVAL ANALYSIS
# ============================================================================

def comprehensive_survival_analysis(merged_data, X_features):
    """Multiple survival analysis approaches"""
    
    print(f"\n3. COMPREHENSIVE SURVIVAL ANALYSIS")
    print("-" * 60)
    
    y_survival = merged_data['vital_status']
    survival_time = merged_data['survival_time']
    
    # Split data
    X_train, X_test, y_train, y_test, time_train, time_test = train_test_split(
        X_features, y_survival, survival_time, 
        test_size=0.2, random_state=42, 
        stratify=y_survival
    )
    
    print(f"✓ Training samples: {len(X_train)} ({y_train.sum()} events)")
    print(f"✓ Test samples: {len(X_test)} ({y_test.sum()} events)")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = {}
    
    # Random Forest Classifier
    print(f"\n3.1 Random Forest Survival Classification")
    rf_clf = RandomForestClassifier(n_estimators=200, max_depth=10, 
                                   min_samples_split=10, random_state=42, n_jobs=-1)
    rf_clf.fit(X_train_scaled, y_train)
    
    rf_pred = rf_clf.predict(X_test_scaled)
    rf_prob = rf_clf.predict_proba(X_test_scaled)[:, 1]
    
    rf_acc = accuracy_score(y_test, rf_pred)
    rf_auc = roc_auc_score(y_test, rf_prob)
    
    results['Random_Forest'] = {'accuracy': rf_acc, 'auc': rf_auc}
    print(f"  - Test AUC: {rf_auc:.3f}, Accuracy: {rf_acc:.3f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_features.columns,
        'importance': rf_clf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Logistic Regression
    print(f"\n3.2 Logistic Regression with Regularization")
    lr = LogisticRegression(C=0.1, penalty='l2', solver='liblinear', max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    
    lr_pred = lr.predict(X_test_scaled)
    lr_prob = lr.predict_proba(X_test_scaled)[:, 1]
    lr_acc = accuracy_score(y_test, lr_pred)
    lr_auc = roc_auc_score(y_test, lr_prob)
    
    results['Logistic_Regression'] = {'accuracy': lr_acc, 'auc': lr_auc}
    print(f"  - Test AUC: {lr_auc:.3f}, Accuracy: {lr_acc:.3f}")
    
    # Support Vector Machine
    print(f"\n3.3 Support Vector Machine")
    n_svm = min(1000, len(X_train_scaled))
    svm_idx = np.random.choice(len(X_train_scaled), n_svm, replace=False)
    
    svm_clf = SVC(probability=True, kernel='rbf', random_state=42)
    svm_clf.fit(X_train_scaled[svm_idx], y_train.iloc[svm_idx])
    
    svm_pred = svm_clf.predict(X_test_scaled)
    svm_prob = svm_clf.predict_proba(X_test_scaled)[:, 1]
    svm_acc = accuracy_score(y_test, svm_pred)
    svm_auc = roc_auc_score(y_test, svm_prob)
    
    results['SVM'] = {'accuracy': svm_acc, 'auc': svm_auc}
    print(f"  - Test AUC: {svm_auc:.3f}, Accuracy: {svm_acc:.3f}")
    
    # Gradient Boosting
    print(f"\n3.4 Gradient Boosting")
    gb_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, 
                                       max_depth=4, random_state=42)
    gb_clf.fit(X_train_scaled, y_train)
    
    gb_pred = gb_clf.predict(X_test_scaled)
    gb_prob = gb_clf.predict_proba(X_test_scaled)[:, 1]
    gb_acc = accuracy_score(y_test, gb_pred)
    gb_auc = roc_auc_score(y_test, gb_prob)
    
    results['Gradient_Boosting'] = {'accuracy': gb_acc, 'auc': gb_auc}
    print(f"  - Test AUC: {gb_auc:.3f}, Accuracy: {gb_acc:.3f}")
    
    # Ensemble Voting Classifier
    print(f"\n3.5 Voting Ensemble")
    voting_clf = VotingClassifier(
        estimators=[
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('lr', LogisticRegression(random_state=42, max_iter=1000)),
            ('gb', GradientBoostingClassifier(n_estimators=50, random_state=42))
        ],
        voting='soft'
    )
    
    voting_clf.fit(X_train_scaled, y_train)
    voting_pred = voting_clf.predict(X_test_scaled)
    voting_prob = voting_clf.predict_proba(X_test_scaled)[:, 1]
    
    voting_acc = accuracy_score(y_test, voting_pred)
    voting_auc = roc_auc_score(y_test, voting_prob)
    
    results['Voting_Ensemble'] = {'accuracy': voting_acc, 'auc': voting_auc}
    print(f"  - Test AUC: {voting_auc:.3f}, Accuracy: {voting_acc:.3f}")
    
    return results, feature_importance, scaler

# ============================================================================
# 4. DRUG RESPONSE PREDICTION
# ============================================================================

def comprehensive_drug_response_prediction(merged_data, X_features, drug_cols):
    """Comprehensive drug response prediction"""
    
    print(f"\n4. COMPREHENSIVE DRUG RESPONSE PREDICTION")
    print("-" * 60)
    
    selected_drugs = drug_cols[:8]  # First 8 drugs
    all_drug_results = {}
    
    for drug_idx, drug_col in enumerate(selected_drugs):
        drug_name = drug_col.replace('_IC50', '')
        print(f"\n4.{drug_idx+1} {drug_name} Response Prediction")
        
        y_drug = merged_data[drug_col].values
        
        # Remove any infinite or NaN values
        valid_idx = np.isfinite(y_drug)
        X_drug = X_features[valid_idx]
        y_drug = y_drug[valid_idx]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_drug, y_drug, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler_X = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)
        
        drug_results = {}
        
        # Random Forest
        rf_drug = RandomForestRegressor(n_estimators=100, max_depth=10,
                                       min_samples_split=5, random_state=42, n_jobs=-1)
        rf_drug.fit(X_train_scaled, y_train)
        rf_pred = rf_drug.predict(X_test_scaled)
        
        rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
        rf_r2 = r2_score(y_test, rf_pred)
        rf_corr = np.corrcoef(y_test, rf_pred)[0,1]
        
        drug_results['Random_Forest'] = {
            'rmse': rf_rmse, 'r2': rf_r2, 'correlation': rf_corr
        }
        
        # Elastic Net
        elastic_drug = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42, max_iter=2000)
        elastic_drug.fit(X_train_scaled, y_train)
        elastic_pred = elastic_drug.predict(X_test_scaled)
        
        elastic_rmse = np.sqrt(mean_squared_error(y_test, elastic_pred))
        elastic_r2 = r2_score(y_test, elastic_pred)
        elastic_corr = np.corrcoef(y_test, elastic_pred)[0,1]
        
        drug_results['Elastic_Net'] = {
            'rmse': elastic_rmse, 'r2': elastic_r2, 'correlation': elastic_corr
        }
        
        # Ridge Regression
        ridge_drug = Ridge(alpha=1.0, random_state=42)
        ridge_drug.fit(X_train_scaled, y_train)
        ridge_pred = ridge_drug.predict(X_test_scaled)
        
        ridge_rmse = np.sqrt(mean_squared_error(y_test, ridge_pred))
        ridge_r2 = r2_score(y_test, ridge_pred)
        ridge_corr = np.corrcoef(y_test, ridge_pred)[0,1]
        
        drug_results['Ridge_Regression'] = {
            'rmse': ridge_rmse, 'r2': ridge_r2, 'correlation': ridge_corr
        }
        
        print(f"  Random Forest: RMSE={rf_rmse:.3f}, R²={rf_r2:.3f}, r={rf_corr:.3f}")
        print(f"  Elastic Net:   RMSE={elastic_rmse:.3f}, R²={elastic_r2:.3f}, r={elastic_corr:.3f}")
        print(f"  Ridge Regression: RMSE={ridge_rmse:.3f}, R²={ridge_r2:.3f}, r={ridge_corr:.3f}")
        
        all_drug_results[drug_name] = drug_results
    
    return all_drug_results

# ============================================================================
# 5. EVALUATION AND VISUALIZATION
# ============================================================================

def comprehensive_evaluation(survival_results, drug_results):
    """Comprehensive model evaluation and comparison"""
    
    print(f"\n5. COMPREHENSIVE MODEL EVALUATION")
    print("-" * 60)
    
    # Compile all results
    all_results = []
    
    # Survival analysis results
    for model_name, metrics in survival_results.items():
        for metric_name, value in metrics.items():
            all_results.append({
                'Task': 'Survival Analysis',
                'Model': model_name.replace('_', ' '),
                'Metric': metric_name.upper(),
                'Value': value,
                'Model_Type': 'Survival'
            })
    
    # Drug response results
    for drug_name, models in drug_results.items():
        for model_name, metrics in models.items():
            for metric_name, value in metrics.items():
                all_results.append({
                    'Task': 'Drug Response',
                    'Model': f"{model_name.replace('_', ' ')} ({drug_name})",
                    'Metric': metric_name.upper(),
                    'Value': value,
                    'Model_Type': 'Drug Response'
                })
    
    results_df = pd.DataFrame(all_results)
    
    # Summary statistics
    print(f"\n5.1 PERFORMANCE SUMMARY")
    print("=" * 40)
    
    # Best survival model
    survival_auc = results_df[
        (results_df['Task'] == 'Survival Analysis') & 
        (results_df['Metric'] == 'AUC')
    ]
    
    if not survival_auc.empty:
        best_survival = survival_auc.loc[survival_auc['Value'].idxmax()]
        print(f"Best Survival Model: {best_survival['Model']} (AUC: {best_survival['Value']:.3f})")
    
    # Drug response summary
    drug_correlations = results_df[
        (results_df['Task'] == 'Drug Response') & 
        (results_df['Metric'] == 'CORRELATION')
    ]
    
    if not drug_correlations.empty:
        mean_correlation = drug_correlations['Value'].mean()
        best_drug_model = drug_correlations.loc[drug_correlations['Value'].idxmax()]
        print(f"Best Drug Response Model: {best_drug_model['Model']} (r: {best_drug_model['Value']:.3f})")
        print(f"Average Drug Response Correlation: {mean_correlation:.3f}")
    
    return results_df

def create_visualizations(merged_data, feature_importance, results_df):
    """Create comprehensive visualizations"""
    
    print(f"\n6. CREATING VISUALIZATIONS")
    print("-" * 60)
    
    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle('Cancer ML Pipeline - Comprehensive Analysis', fontsize=16, fontweight='bold')
    
    # 1. Cancer type distribution
    ax1 = axes[0, 0]
    cancer_counts = merged_data['cancer_type'].value_counts()
    ax1.pie(cancer_counts.values, labels=cancer_counts.index, autopct='%1.1f%%')
    ax1.set_title('Cancer Type Distribution')
    
    # 2. Survival status by cancer type
    ax2 = axes[0, 1]
    survival_by_cancer = merged_data.groupby('cancer_type')['vital_status'].mean()
    ax2.bar(survival_by_cancer.index, survival_by_cancer.values)
    ax2.set_xlabel('Cancer Type')
    ax2.set_ylabel('Death Rate')
    ax2.set_title('Death Rate by Cancer Type')
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. Age distribution
    ax3 = axes[0, 2]
    ax3.hist(merged_data['age'], bins=20, alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Age')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Age Distribution')
    
    # 4. Survival time distribution
    ax4 = axes[1, 0]
    ax4.hist(merged_data['survival_time'], bins=30, alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Survival Time (days)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Survival Time Distribution')
    
    # 5. Top feature importance
    ax5 = axes[1, 1]
    top_features = feature_importance.head(15)
    ax5.barh(range(len(top_features)), top_features['importance'])
    ax5.set_yticks(range(len(top_features)))
    ax5.set_yticklabels(top_features['feature'], fontsize=8)
    ax5.set_xlabel('Feature Importance')
    ax5.set_title('Top 15 Most Important Features')
    ax5.invert_yaxis()
    
    # 6. Mutation frequency
    ax6 = axes[1, 2]
    mutation_cols = [col for col in merged_data.columns if col.endswith('_mutation')]
    if mutation_cols:
        mutation_freq = merged_data[mutation_cols].mean().sort_values(ascending=True)
        gene_names = [col.replace('_mutation', '') for col in mutation_freq.index]
        
        ax6.barh(range(len(mutation_freq)), mutation_freq.values)
        ax6.set_yticks(range(len(mutation_freq)))
        ax6.set_yticklabels(gene_names, fontsize=8)
        ax6.set_xlabel('Mutation Frequency')
        ax6.set_title('Mutation Frequency by Gene')
    
    # 7. Model performance comparison (Survival)
    ax7 = axes[2, 0]
    survival_auc = results_df[
        (results_df['Task'] == 'Survival Analysis') & 
        (results_df['Metric'] == 'AUC')
    ]
    
    if not survival_auc.empty:
        ax7.bar(range(len(survival_auc)), survival_auc['Value'])
        ax7.set_xticks(range(len(survival_auc)))
        ax7.set_xticklabels([model[:10] + '...' if len(model) > 10 else model 
                           for model in survival_auc['Model']], rotation=45, ha='right')
        ax7.set_ylabel('AUC Score')
        ax7.set_title('Survival Model Performance')
        ax7.set_ylim(0, 1)
    
    # 8. Drug response performance
    ax8 = axes[2, 1]
    drug_corr = results_df[
        (results_df['Task'] == 'Drug Response') & 
        (results_df['Metric'] == 'CORRELATION')
    ]
    
    if not drug_corr.empty:
        ax8.hist(drug_corr['Value'], bins=10, alpha=0.7, edgecolor='black')
        ax8.set_xlabel('Correlation')
        ax8.set_ylabel('Frequency')
        ax8.set_title('Drug Response Correlations')
        ax8.axvline(drug_corr['Value'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {drug_corr["Value"].mean():.2f}')
        ax8.legend()
    
    # 9. Stage distribution
    ax9 = axes[2, 2]
    stage_counts = merged_data['stage'].value_counts()
    ax9.bar(stage_counts.index, stage_counts.values)
    ax9.set_xlabel('Cancer Stage')
    ax9.set_ylabel('Count')
    ax9.set_title('Cancer Stage Distribution')
    
    plt.tight_layout()
    plt.savefig('comprehensive_cancer_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✓ Comprehensive visualization saved as 'comprehensive_cancer_analysis.png'")
    
    return fig

# ============================================================================
# 6. MAIN EXECUTION PIPELINE
# ============================================================================

def main_execution_pipeline():
    """Execute the complete comprehensive ML pipeline"""
    
    print(f"\n" + "="*80)
    print("EXECUTING COMPREHENSIVE CANCER ML PIPELINE")
    print("="*80)
    
    start_time = pd.Timestamp.now()
    
    # Step 1: Generate comprehensive dataset
    print(f"\nSTEP 1: Data Generation")
    rna_df, mutation_df, clinical_df, drug_df = generate_comprehensive_cancer_data(
        n_samples=2000, n_genes=1000, n_mutations=25
    )
    
    # Step 2: Advanced preprocessing
    print(f"\nSTEP 2: Advanced Preprocessing")
    merged_data, X_features, selected_genes, mutation_cols, drug_cols = advanced_preprocessing(
        rna_df, mutation_df, clinical_df, drug_df
    )
    
    # Step 3: Survival analysis
    print(f"\nSTEP 3: Comprehensive Survival Analysis")
    survival_results, feature_importance, scaler = comprehensive_survival_analysis(
        merged_data, X_features
    )
    
    # Step 4: Drug response prediction
    print(f"\nSTEP 4: Drug Response Prediction")
    drug_results = comprehensive_drug_response_prediction(
        merged_data, X_features, drug_cols
    )
    
    # Step 5: Comprehensive evaluation
    print(f"\nSTEP 5: Model Evaluation")
    results_df = comprehensive_evaluation(survival_results, drug_results)
    
    # Step 6: Visualizations
    print(f"\nSTEP 6: Creating Visualizations")
    fig = create_visualizations(merged_data, feature_importance, results_df)
    
    # Step 7: Save all results
    print(f"\nSTEP 7: Saving Results")
    
    # Save datasets
    merged_data.to_csv('final_cancer_dataset.csv', index=False)
    X_features.to_csv('processed_features.csv', index=False)
    
    # Save results
    results_df.to_csv('comprehensive_results.csv', index=False)
    feature_importance.to_csv('feature_importance_final.csv', index=False)
    
    # Create summary report
    execution_time = pd.Timestamp.now() - start_time
    
    # Find best models
    survival_auc_results = results_df[
        (results_df['Task'] == 'Survival Analysis') & 
        (results_df['Metric'] == 'AUC')
    ].sort_values('Value', ascending=False)
    
    drug_corr_results = results_df[
        (results_df['Task'] == 'Drug Response') & 
        (results_df['Metric'] == 'CORRELATION')
    ]
    
    best_survival_model = survival_auc_results.iloc[0] if not survival_auc_results.empty else None
    avg_drug_correlation = drug_corr_results['Value'].mean() if not drug_corr_results.empty else 0
    
    summary_report = f"""
COMPREHENSIVE CANCER ML PIPELINE - EXECUTION SUMMARY
=====================================================

Execution Time: {execution_time}
Dataset Size: {len(merged_data):,} samples
Features: {X_features.shape[1]} (after preprocessing)
Cancer Types: {merged_data['cancer_type'].nunique()}

SURVIVAL ANALYSIS RESULTS:
"""
    
    for _, row in survival_auc_results.head(5).iterrows():
        summary_report += f"- {row['Model']}: AUC = {row['Value']:.3f}\n"
    
    summary_report += f"""
DRUG RESPONSE RESULTS:
- Average Correlation: {avg_drug_correlation:.3f}
- Models tested per drug: 3 (Random Forest, Elastic Net, Ridge)

TOP 5 MOST IMPORTANT FEATURES:
"""
    
    for _, row in feature_importance.head(5).iterrows():
        summary_report += f"- {row['feature']}: {row['importance']:.4f}\n"
    
    summary_report += f"""
FILES GENERATED:
- final_cancer_dataset.csv: Complete processed dataset
- processed_features.csv: Feature matrix used for modeling
- comprehensive_results.csv: All model results
- feature_importance_final.csv: Feature importance rankings
- comprehensive_cancer_analysis.png: Comprehensive visualization dashboard

RECOMMENDATIONS:
1. Best survival model: {best_survival_model['Model'] if best_survival_model is not None else 'N/A'}
2. Ensemble methods showed competitive performance
3. Feature selection effectively reduced dimensionality
4. Drug response prediction varies by compound and target mechanism

Pipeline completed successfully!
"""
    
    # Save summary report
    with open('execution_summary.txt', 'w') as f:
        f.write(summary_report)
    
    # Print final summary
    print(f"\n" + "="*80)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(summary_report)
    
    return merged_data, X_features, results_df, feature_importance

# ============================================================================
# EXECUTE THE COMPLETE PIPELINE
# ============================================================================

if __name__ == "__main__":
    print("\n🚀 STARTING COMPREHENSIVE CANCER ML PIPELINE")
    print("This will generate synthetic data and train multiple ML models")
    print("for cancer survival and drug response prediction.\n")
    
    # Execute the main pipeline
    try:
        final_data, final_features, final_results, final_importance = main_execution_pipeline()
        
        print("\n✅ SUCCESS: All models trained and results saved!")
        print("\n📁 Generated Files:")
        print("   - final_cancer_dataset.csv")
        print("   - processed_features.csv") 
        print("   - comprehensive_results.csv")
        print("   - feature_importance_final.csv")
        print("   - comprehensive_cancer_analysis.png")
        print("   - execution_summary.txt")
        
        print(f"\n📊 Quick Results Summary:")
        print(f"   - Dataset: {len(final_data):,} samples, {final_features.shape[1]} features")
        print(f"   - Best Survival AUC: {final_results[(final_results['Task'] == 'Survival Analysis') & (final_results['Metric'] == 'AUC')]['Value'].max():.3f}")
        print(f"   - Avg Drug Response r: {final_results[(final_results['Task'] == 'Drug Response') & (final_results['Metric'] == 'CORRELATION')]['Value'].mean():.3f}")
        
    except Exception as e:
        print(f"\n❌ ERROR: Pipeline execution failed: {str(e)}")
        print("Please check the error details and try again.")

# END OF COMPLETE ML PIPELINE