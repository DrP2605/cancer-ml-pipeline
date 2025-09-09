# Complete End-to-End ML Pipeline for Cancer Survival and Drug Response Prediction
# Date: September 2025

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.metrics import (accuracy_score, roc_auc_score, mean_squared_error, 
                           mean_absolute_error, r2_score, classification_report)
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
import xgboost as xgb
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

def generate_synthetic_cancer_data(n_samples=1000, n_genes=500, n_mutations=25):
    """Generate synthetic cancer dataset mimicking real RNA-seq and mutation data"""
    
    print("\n1. GENERATING SYNTHETIC CANCER DATASET")
    print("-" * 50)
    
    # Sample IDs
    sample_ids = [f'TCGA-{i:04d}' for i in range(n_samples)]
    
    # Cancer types (realistic distribution)
    cancer_types = np.random.choice(['BRCA', 'LUAD', 'COAD', 'LGG', 'KIRC', 'HNSC'], 
                                   n_samples, p=[0.25, 0.20, 0.15, 0.15, 0.15, 0.10])
    
    # Generate RNA-seq expression data (log2-transformed TPM values)
    # Simulate realistic gene expression with some genes being more variable
    gene_names = [f'GENE_{i+1:04d}' for i in range(n_genes)]
    
    rna_data = {}
    rna_data['sample_id'] = sample_ids
    rna_data['cancer_type'] = cancer_types
    
    for i, gene in enumerate(gene_names):
        # Some genes are cancer-type specific
        base_expression = np.random.lognormal(mean=3, sigma=1.5, size=n_samples)
        
        # Add cancer-type specific effects for some genes
        if i < 50:  # First 50 genes are cancer-type specific
            cancer_effect = np.where(cancer_types == 'BRCA', 2.0,
                           np.where(cancer_types == 'LUAD', 1.5,
                           np.where(cancer_types == 'COAD', 0.8, 1.0)))
            base_expression *= cancer_effect
        
        rna_data[gene] = base_expression
    
    rna_df = pd.DataFrame(rna_data)
    
    # Generate mutation data for key cancer genes
    key_genes = ['TP53', 'KRAS', 'PIK3CA', 'BRAF', 'PTEN', 'APC', 'EGFR', 'IDH1', 
                 'CDKN2A', 'RB1', 'VHL', 'ARID1A', 'CTNNB1', 'SMAD4', 'ATM',
                 'BRCA1', 'BRCA2', 'MLH1', 'MSH2', 'STK11', 'LKB1', 'NF1', 
                 'NOTCH1', 'FBXW7', 'FGFR3']
    
    mutation_data = {'sample_id': sample_ids}
    
    for gene in key_genes[:n_mutations]:
        # Mutation frequencies vary by cancer type and gene
        if gene in ['TP53']:
            prob = 0.4  # High frequency
        elif gene in ['KRAS', 'PIK3CA']:
            prob = 0.25  # Medium frequency
        else:
            prob = 0.15  # Lower frequency
            
        # Cancer-type specific mutation rates
        cancer_specific_prob = np.where(cancer_types == 'COAD', prob * 1.5,
                              np.where(cancer_types == 'BRCA', prob * 1.2,
                              np.where(cancer_types == 'LUAD', prob * 0.8, prob)))
        
        mutations = np.random.binomial(1, cancer_specific_prob, n_samples)
        mutation_data[f'{gene}_mutation'] = mutations
    
    mutation_df = pd.DataFrame(mutation_data)
    
    # Generate clinical survival data
    # Survival time influenced by cancer type, age, stage, and molecular features
    base_hazard = np.random.exponential(scale=600, size=n_samples)
    
    # Age effect
    age = np.random.normal(65, 12, n_samples)
    age = np.clip(age, 25, 90)
    age_effect = (age - 65) / 10 * 0.2  # 20% increase per decade
    
    # Cancer type effect
    cancer_effect = np.where(cancer_types == 'LGG', -0.8,  # Better prognosis
                    np.where(cancer_types == 'KIRC', -0.4,
                    np.where(cancer_types == 'BRCA', -0.2,
                    np.where(cancer_types == 'LUAD', 0.3, 0.0))))  # Worse prognosis
    
    # Stage effect (simplified)
    stage = np.random.choice(['I', 'II', 'III', 'IV'], n_samples, p=[0.2, 0.3, 0.3, 0.2])
    stage_effect = np.where(stage == 'I', -0.6,
                   np.where(stage == 'II', -0.3,
                   np.where(stage == 'III', 0.2, 0.8)))
    
    # Molecular effects (TP53, high-risk gene signature)
    tp53_effect = mutation_data.get('TP53_mutation', np.zeros(n_samples)) * 0.4
    
    # High expression of first few genes indicates poor prognosis
    high_risk_signature = np.mean([rna_data[f'GENE_{i+1:04d}'] for i in range(5)], axis=0)
    signature_effect = (high_risk_signature - np.mean(high_risk_signature)) / np.std(high_risk_signature) * 0.3
    
    # Combine all effects
    log_hazard = age_effect + cancer_effect + stage_effect + tp53_effect + signature_effect
    hazard_ratio = np.exp(log_hazard)
    
    survival_time = base_hazard / hazard_ratio
    survival_time = np.clip(survival_time, 10, 3000)  # Realistic range: 10-3000 days
    
    # Censoring (some patients are still alive at end of study)
    follow_up_time = np.random.uniform(500, 2000, n_samples)  # Study follow-up time
    observed_time = np.minimum(survival_time, follow_up_time)
    vital_status = (survival_time <= follow_up_time).astype(int)  # 1=death observed, 0=censored
    
    clinical_data = {
        'sample_id': sample_ids,
        'age': age,
        'stage': stage,
        'survival_time': observed_time,
        'vital_status': vital_status
    }
    clinical_df = pd.DataFrame(clinical_data)
    
    # Generate drug response data (IC50 values)
    drugs = ['Cisplatin', 'Paclitaxel', 'Temozolomide', 'Erlotinib', 'Sorafenib', 
             'Gemcitabine', 'Docetaxel', '5-Fluorouracil']
    
    drug_response_data = {'sample_id': sample_ids}
    
    for drug in drugs:
        # Drug response influenced by mutations and gene expression
        base_ic50 = np.random.normal(2.0, 0.8, n_samples)  # log IC50
        
        # Drug-specific effects
        if drug in ['Erlotinib']:  # EGFR inhibitor
            egfr_mut = mutation_data.get('EGFR_mutation', np.zeros(n_samples))
            drug_effect = egfr_mut * -1.0  # Sensitive if EGFR mutated
        elif drug in ['Cisplatin', 'Paclitaxel']:  # DNA damaging agents
            tp53_mut = mutation_data.get('TP53_mutation', np.zeros(n_samples))
            drug_effect = tp53_mut * 0.5  # Resistant if TP53 mutated
        else:
            drug_effect = 0
            
        # Expression-based effects (simplified)
        expr_effect = (signature_effect * 0.3)
        
        final_ic50 = base_ic50 + drug_effect + expr_effect + np.random.normal(0, 0.3, n_samples)
        drug_response_data[f'{drug}_IC50'] = final_ic50
    
    drug_df = pd.DataFrame(drug_response_data)
    
    print(f"✓ Generated {n_samples} samples with {n_genes} genes and {n_mutations} mutation features")
    print(f"✓ Cancer type distribution: {dict(pd.Series(cancer_types).value_counts())}")
    print(f"✓ Survival events: {vital_status.sum()} deaths, {(1-vital_status).sum()} censored")
    print(f"✓ Mean survival time: {observed_time.mean():.1f} days")
    
    return rna_df, mutation_df, clinical_df, drug_df

# ============================================================================
# 2. DATA PREPROCESSING AND FEATURE ENGINEERING
# ============================================================================

def preprocess_and_merge_data(rna_df, mutation_df, clinical_df, drug_df):
    """Comprehensive data preprocessing and feature engineering"""
    
    print("\n2. DATA PREPROCESSING AND FEATURE ENGINEERING")
    print("-" * 50)
    
    # Merge all datasets
    merged_data = rna_df.merge(mutation_df, on='sample_id')
    merged_data = merged_data.merge(clinical_df, on='sample_id')
    merged_data = merged_data.merge(drug_df, on='sample_id')
    
    print(f"✓ Merged dataset shape: {merged_data.shape}")
    
    # Identify feature columns
    gene_cols = [col for col in merged_data.columns if col.startswith('GENE_')]
    mutation_cols = [col for col in merged_data.columns if col.endswith('_mutation')]
    drug_cols = [col for col in merged_data.columns if col.endswith('_IC50')]
    
    print(f"✓ Gene expression features: {len(gene_cols)}")
    print(f"✓ Mutation features: {len(mutation_cols)}")
    print(f"✓ Drug response targets: {len(drug_cols)}")
    
    # Feature selection for gene expression (top 100 most variable genes)
    gene_expression = merged_data[gene_cols]
    gene_variance = gene_expression.var()
    top_genes = gene_variance.nlargest(100).index.tolist()
    
    print(f"✓ Selected top 100 most variable genes")
    
    # Create feature matrix
    feature_cols = top_genes + mutation_cols + ['age', 'cancer_type', 'stage']
    X_features = merged_data[feature_cols].copy()
    
    # Encode categorical variables
    le_cancer = LabelEncoder()
    le_stage = LabelEncoder()
    
    X_features['cancer_type_encoded'] = le_cancer.fit_transform(X_features['cancer_type'])
    X_features['stage_encoded'] = le_stage.fit_transform(X_features['stage'])
    
    # Drop original categorical columns
    X_features = X_features.drop(['cancer_type', 'stage'], axis=1)
    
    print(f"✓ Final feature matrix shape: {X_features.shape}")
    print(f"✓ Features: {X_features.columns.tolist()[:10]}... (showing first 10)")
    
    return merged_data, X_features, top_genes, mutation_cols, drug_cols

# ============================================================================
# 3. SURVIVAL ANALYSIS MODELS
# ============================================================================

def survival_analysis_models(merged_data, X_features):
    """Comprehensive survival analysis using multiple approaches"""
    
    print("\n3. SURVIVAL ANALYSIS MODELS")
    print("-" * 50)
    
    # Prepare survival data
    y_survival = merged_data['vital_status']
    survival_time = merged_data['survival_time']
    
    # Split data
    X_train, X_test, y_train, y_test, time_train, time_test = train_test_split(
        X_features, y_survival, survival_time, test_size=0.2, random_state=42, stratify=y_survival
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = {}
    
    # 3.1 Random Forest Classifier
    print("\n3.1 Random Forest Survival Classification")
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_classifier.fit(X_train_scaled, y_train)
    
    rf_pred = rf_classifier.predict(X_test_scaled)
    rf_prob = rf_classifier.predict_proba(X_test_scaled)[:, 1]
    
    rf_acc = accuracy_score(y_test, rf_pred)
    rf_auc = roc_auc_score(y_test, rf_prob)
    
    results['RF_Classification'] = {'accuracy': rf_acc, 'auc': rf_auc}
    print(f"Random Forest - Accuracy: {rf_acc:.3f}, AUC: {rf_auc:.3f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_features.columns,
        'importance': rf_classifier.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("Top 10 most important features:")
    print(feature_importance.head(10)[['feature', 'importance']].to_string(index=False))
    
    # 3.2 Cox Proportional Hazards Model
    print("\n3.2 Cox Proportional Hazards Model")
    
    # Prepare data for Cox model
    cox_data = pd.DataFrame(X_train_scaled, columns=[f'X{i}' for i in range(X_train_scaled.shape[1])])
    cox_data['duration'] = time_train.values
    cox_data['event'] = y_train.values
    
    # Fit Cox model with regularization
    cph = CoxPHFitter(penalizer=0.1)
    try:
        cph.fit(cox_data, duration_col='duration', event_col='event')
        
        # Predict on test set
        cox_test_data = pd.DataFrame(X_test_scaled, columns=[f'X{i}' for i in range(X_test_scaled.shape[1])])
        risk_scores = cph.predict_partial_hazard(cox_test_data)
        
        # Calculate C-index
        c_index = concordance_index(time_test, -risk_scores, y_test)
        results['Cox_Model'] = {'c_index': c_index}
        print(f"Cox Model - C-index: {c_index:.3f}")
        
    except Exception as e:
        print(f"Cox model failed: {str(e)}")
        results['Cox_Model'] = {'c_index': np.nan}
    
    # 3.3 Survival Time Regression
    print("\n3.3 Survival Time Regression")
    
    # Use uncensored samples for time regression
    uncensored_idx = y_train == 1
    if uncensored_idx.sum() > 50:  # Need sufficient uncensored samples
        
        X_uncensored = X_train_scaled[uncensored_idx]
        y_uncensored = time_train.values[uncensored_idx]
        
        # Random Forest Regressor
        rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf_regressor.fit(X_uncensored, y_uncensored)
        
        # Elastic Net
        elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
        elastic_net.fit(X_uncensored, y_uncensored)
        
        # Evaluate on test uncensored samples
        test_uncensored_idx = y_test == 1
        if test_uncensored_idx.sum() > 10:
            X_test_uncensored = X_test_scaled[test_uncensored_idx]
            y_test_uncensored = time_test.values[test_uncensored_idx]
            
            rf_pred_time = rf_regressor.predict(X_test_uncensored)
            en_pred_time = elastic_net.predict(X_test_uncensored)
            
            rf_rmse = np.sqrt(mean_squared_error(y_test_uncensored, rf_pred_time))
            en_rmse = np.sqrt(mean_squared_error(y_test_uncensored, en_pred_time))
            
            results['RF_Regression'] = {'rmse': rf_rmse}
            results['ElasticNet_Regression'] = {'rmse': en_rmse}
            
            print(f"RF Time Regression - RMSE: {rf_rmse:.1f} days")
            print(f"Elastic Net Time Regression - RMSE: {en_rmse:.1f} days")
    
    return results, feature_importance, scaler

# ============================================================================
# 4. DRUG RESPONSE PREDICTION
# ============================================================================

def drug_response_prediction(merged_data, X_features, drug_cols):
    """Predict drug response (IC50 values) using multiple ML approaches"""
    
    print("\n4. DRUG RESPONSE PREDICTION")
    print("-" * 50)
    
    drug_results = {}
    
    # Select representative drugs
    selected_drugs = drug_cols[:4]  # First 4 drugs
    
    for drug_col in selected_drugs:
        drug_name = drug_col.replace('_IC50', '')
        print(f"\n4.{selected_drugs.index(drug_col)+1} Predicting {drug_name} Response")
        
        y_drug = merged_data[drug_col]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_features, y_drug, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Random Forest
        rf_drug = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf_drug.fit(X_train_scaled, y_train)
        rf_pred = rf_drug.predict(X_test_scaled)
        
        # XGBoost
        xgb_drug = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, 
                                    max_depth=6, random_state=42, n_jobs=-1)
        xgb_drug.fit(X_train_scaled, y_train)
        xgb_pred = xgb_drug.predict(X_test_scaled)
        
        # Evaluate models
        rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
        rf_r2 = r2_score(y_test, rf_pred)
        rf_corr = np.corrcoef(y_test, rf_pred)[0,1]
        
        xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
        xgb_r2 = r2_score(y_test, xgb_pred)
        xgb_corr = np.corrcoef(y_test, xgb_pred)[0,1]
        
        drug_results[drug_name] = {
            'RF': {'rmse': rf_rmse, 'r2': rf_r2, 'correlation': rf_corr},
            'XGBoost': {'rmse': xgb_rmse, 'r2': xgb_r2, 'correlation': xgb_corr}
        }
        
        print(f"Random Forest - RMSE: {rf_rmse:.3f}, R²: {rf_r2:.3f}, Correlation: {rf_corr:.3f}")
        print(f"XGBoost - RMSE: {xgb_rmse:.3f}, R²: {xgb_r2:.3f}, Correlation: {xgb_corr:.3f}")
    
    return drug_results

# ============================================================================
# 5. ADVANCED DEEP LEARNING MODEL
# ============================================================================

def deep_learning_model(merged_data, X_features, top_genes, mutation_cols):
    """Multi-task deep learning model using TensorFlow/Keras"""
    
    print("\n5. ADVANCED DEEP LEARNING MODEL")
    print("-" * 50)
    
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
        
        # Prepare data
        gene_features = X_features[top_genes]
        mutation_features = X_features[mutation_cols]
        clinical_features = X_features[['age', 'cancer_type_encoded', 'stage_encoded']]
        
        # Targets
        y_survival = merged_data['vital_status']
        y_drug = merged_data['Cisplatin_IC50']  # Example drug
        
        # Split data
        indices = range(len(X_features))
        train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
        
        # Scale features
        gene_scaler = StandardScaler()
        mutation_scaler = StandardScaler()
        clinical_scaler = StandardScaler()
        
        gene_train = gene_scaler.fit_transform(gene_features.iloc[train_idx])
        gene_test = gene_scaler.transform(gene_features.iloc[test_idx])
        
        mutation_train = mutation_scaler.fit_transform(mutation_features.iloc[train_idx])
        mutation_test = mutation_scaler.transform(mutation_features.iloc[test_idx])
        
        clinical_train = clinical_scaler.fit_transform(clinical_features.iloc[train_idx])
        clinical_test = clinical_scaler.transform(clinical_features.iloc[test_idx])
        
        # Define multi-input model
        # Gene expression input
        gene_input = keras.Input(shape=(len(top_genes),), name='gene_input')
        gene_encoded = layers.Dense(256, activation='relu')(gene_input)
        gene_encoded = layers.Dropout(0.3)(gene_encoded)
        gene_encoded = layers.Dense(128, activation='relu')(gene_encoded)
        gene_encoded = layers.Dropout(0.2)(gene_encoded)
        
        # Mutation input
        mutation_input = keras.Input(shape=(len(mutation_cols),), name='mutation_input')
        mutation_encoded = layers.Dense(64, activation='relu')(mutation_input)
        mutation_encoded = layers.Dropout(0.2)(mutation_encoded)
        mutation_encoded = layers.Dense(32, activation='relu')(mutation_encoded)
        
        # Clinical input
        clinical_input = keras.Input(shape=(3,), name='clinical_input')
        clinical_encoded = layers.Dense(16, activation='relu')(clinical_input)
        clinical_encoded = layers.Dense(8, activation='relu')(clinical_encoded)
        
        # Fusion layer
        combined = layers.concatenate([gene_encoded, mutation_encoded, clinical_encoded])
        fusion = layers.Dense(128, activation='relu')(combined)
        fusion = layers.Dropout(0.3)(fusion)
        fusion = layers.Dense(64, activation='relu')(fusion)
        
        # Multi-task outputs
        survival_output = layers.Dense(1, activation='sigmoid', name='survival')(fusion)
        drug_output = layers.Dense(1, name='drug_response')(fusion)
        
        # Create and compile model
        model = keras.Model(
            inputs=[gene_input, mutation_input, clinical_input],
            outputs=[survival_output, drug_output]
        )
        
        model.compile(
            optimizer='adam',
            loss={
                'survival': 'binary_crossentropy',
                'drug_response': 'mse'
            },
            loss_weights={
                'survival': 1.0,
                'drug_response': 0.5
            },
            metrics={
                'survival': ['accuracy', 'auc'],
                'drug_response': ['mae']
            }
        )
        
        print("✓ Multi-task deep learning model created")
        print(f"✓ Model inputs: Gene ({len(top_genes)}), Mutation ({len(mutation_cols)}), Clinical (3)")
        
        # Train model
        history = model.fit(
            [gene_train, mutation_train, clinical_train],
            {
                'survival': y_survival.iloc[train_idx].values,
                'drug_response': y_drug.iloc[train_idx].values
            },
            validation_data=(
                [gene_test, mutation_test, clinical_test],
                {
                    'survival': y_survival.iloc[test_idx].values,
                    'drug_response': y_drug.iloc[test_idx].values
                }
            ),
            epochs=50,
            batch_size=32,
            verbose=0,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
            ]
        )
        
        # Evaluate model
        test_loss, survival_loss, drug_loss, survival_acc, survival_auc, drug_mae = model.evaluate(
            [gene_test, mutation_test, clinical_test],
            {
                'survival': y_survival.iloc[test_idx].values,
                'drug_response': y_drug.iloc[test_idx].values
            },
            verbose=0
        )
        
        print(f"✓ Deep Learning Model Performance:")
        print(f"  - Survival Classification: Accuracy = {survival_acc:.3f}, AUC = {survival_auc:.3f}")
        print(f"  - Drug Response: MAE = {drug_mae:.3f}")
        
        return {
            'survival_accuracy': survival_acc,
            'survival_auc': survival_auc,
            'drug_mae': drug_mae,
            'history': history.history
        }
        
    except ImportError:
        print("✗ TensorFlow not available - skipping deep learning model")
        return None
    except Exception as e:
        print(f"✗ Deep learning model failed: {str(e)}")
        return None

# ============================================================================
# 6. MODEL EVALUATION AND COMPARISON
# ============================================================================

def evaluate_and_compare_models(survival_results, drug_results, dl_results=None):
    """Comprehensive evaluation and comparison of all models"""
    
    print("\n6. MODEL EVALUATION AND COMPARISON")
    print("-" * 50)
    
    # Create summary results
    summary_data = []
    
    # Survival models
    for model_name, metrics in survival_results.items():
        for metric_name, value in metrics.items():
            summary_data.append({
                'Task': 'Survival Analysis',
                'Model': model_name,
                'Metric': metric_name.upper(),
                'Value': value
            })
    
    # Drug response models
    for drug_name, models in drug_results.items():
        for model_name, metrics in models.items():
            for metric_name, value in metrics.items():
                summary_data.append({
                    'Task': f'Drug Response ({drug_name})',
                    'Model': model_name,
                    'Metric': metric_name.upper(),
                    'Value': value
                })
    
    # Deep learning results
    if dl_results:
        summary_data.extend([
            {'Task': 'Survival Analysis', 'Model': 'Deep Learning', 'Metric': 'ACCURACY', 'Value': dl_results['survival_accuracy']},
            {'Task': 'Survival Analysis', 'Model': 'Deep Learning', 'Metric': 'AUC', 'Value': dl_results['survival_auc']},
            {'Task': 'Drug Response (Cisplatin)', 'Model': 'Deep Learning', 'Metric': 'MAE', 'Value': dl_results['drug_mae']}
        ])
    
    results_df = pd.DataFrame(summary_data)
    
    print("\n6.1 COMPREHENSIVE RESULTS SUMMARY")
    print("=" * 40)
    
    # Group by task and show best performing models
    for task in results_df['Task'].unique():
        task_results = results_df[results_df['Task'] == task]
        print(f"\n{task}:")
        print("-" * len(task))
        
        # Show results by metric
        for metric in task_results['Metric'].unique():
            metric_results = task_results[task_results['Metric'] == metric].copy()
            
            if metric in ['AUC', 'ACCURACY', 'R2', 'CORRELATION']:
                # Higher is better
                best_result = metric_results.loc[metric_results['Value'].idxmax()]
            else:
                # Lower is better (RMSE, MAE)
                best_result = metric_results.loc[metric_results['Value'].idxmin()]
            
            print(f"  {metric}: {best_result['Model']} = {best_result['Value']:.3f}")
    
    return results_df

# ============================================================================
# 7. VISUALIZATION AND REPORTING
# ============================================================================

def create_visualizations(merged_data, feature_importance, results_df):
    """Create comprehensive visualizations"""
    
    print("\n7. CREATING VISUALIZATIONS")
    print("-" * 50)
    
    plt.style.use('default')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Cancer ML Pipeline - Comprehensive Analysis', fontsize=16, fontweight='bold')
    
    # 1. Feature Importance
    ax1 = axes[0, 0]
    top_features = feature_importance.head(15)
    ax1.barh(range(len(top_features)), top_features['importance'])
    ax1.set_yticks(range(len(top_features)))
    ax1.set_yticklabels(top_features['feature'], fontsize=8)
    ax1.set_xlabel('Feature Importance')
    ax1.set_title('Top 15 Most Important Features')
    ax1.invert_yaxis()
    
    # 2. Cancer Type Distribution
    ax2 = axes[0, 1]
    cancer_counts = merged_data['cancer_type'].value_counts()
    ax2.pie(cancer_counts.values, labels=cancer_counts.index, autopct='%1.1f%%')
    ax2.set_title('Cancer Type Distribution')
    
    # 3. Survival Analysis
    ax3 = axes[0, 2]
    survival_by_cancer = merged_data.groupby('cancer_type')['vital_status'].agg(['mean', 'count'])
    ax3.bar(survival_by_cancer.index, survival_by_cancer['mean'])
    ax3.set_xlabel('Cancer Type')
    ax3.set_ylabel('Death Rate')
    ax3.set_title('Death Rate by Cancer Type')
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. Survival Time Distribution
    ax4 = axes[1, 0]
    ax4.hist(merged_data['survival_time'], bins=30, alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Survival Time (days)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Survival Time Distribution')
    
    # 5. Mutation Frequency
    ax5 = axes[1, 1]
    mutation_cols = [col for col in merged_data.columns if col.endswith('_mutation')]
    mutation_freq = merged_data[mutation_cols].mean().sort_values(ascending=True)
    ax5.barh(range(len(mutation_freq)), mutation_freq.values)
    ax5.set_yticks(range(len(mutation_freq)))
    ax5.set_yticklabels([col.replace('_mutation', '') for col in mutation_freq.index], fontsize=8)
    ax5.set_xlabel('Mutation Frequency')
    ax5.set_title('Mutation Frequency by Gene')
    
    # 6. Model Performance Comparison
    ax6 = axes[1, 2]
    # Show AUC scores for survival models
    survival_results = results_df[
        (results_df['Task'] == 'Survival Analysis') & 
        (results_df['Metric'] == 'AUC')
    ]
    
    if not survival_results.empty:
        ax6.bar(survival_results['Model'], survival_results['Value'])
        ax6.set_ylabel('AUC Score')
        ax6.set_title('Survival Model Performance (AUC)')
        ax6.tick_params(axis='x', rotation=45)
        ax6.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('cancer_ml_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Comprehensive visualization saved as 'cancer_ml_comprehensive_analysis.png'")
    
    return fig

# ============================================================================
# 8. MAIN EXECUTION PIPELINE
# ============================================================================

def main():
    """Execute the complete ML pipeline"""
    
    print("Starting comprehensive cancer ML pipeline execution...")
    
    # Generate synthetic data
    rna_df, mutation_df, clinical_df, drug_df = generate_synthetic_cancer_data(
        n_samples=1000, n_genes=500, n_mutations=25
    )
    
    # Preprocess and merge data
    merged_data, X_features, top_genes, mutation_cols, drug_cols = preprocess_and_merge_data(
        rna_df, mutation_df, clinical_df, drug_df
    )
    
    # Survival analysis
    survival_results, feature_importance, scaler = survival_analysis_models(merged_data, X_features)
    
    # Drug response prediction
    drug_results = drug_response_prediction(merged_data, X_features, drug_cols)
    
    # Deep learning model (optional)
    dl_results = deep_learning_model(merged_data, X_features, top_genes, mutation_cols)
    
    # Evaluate and compare models
    results_df = evaluate_and_compare_models(survival_results, drug_results, dl_results)
    
    # Create visualizations
    fig = create_visualizations(merged_data, feature_importance, results_df)
    
    # Save results
    results_df.to_csv('comprehensive_ml_results.csv', index=False)
    feature_importance.to_csv('feature_importance_rankings.csv', index=False)
    merged_data.to_csv('processed_cancer_dataset.csv', index=False)
    
    print("\n" + "="*80)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"✓ Results saved to 'comprehensive_ml_results.csv'")
    print(f"✓ Feature importance saved to 'feature_importance_rankings.csv'")
    print(f"✓ Processed dataset saved to 'processed_cancer_dataset.csv'")
    print(f"✓ Visualizations saved to 'cancer_ml_comprehensive_analysis.png'")
    
    # Final summary
    print(f"\n📊 FINAL SUMMARY:")
    print(f"   - Total samples analyzed: {len(merged_data)}")
    print(f"   - Total features used: {X_features.shape[1]}")
    print(f"   - Survival models trained: {len(survival_results)}")
    print(f"   - Drug response models: {sum(len(models) for models in drug_results.values())}")
    print(f"   - Best survival AUC: {results_df[(results_df['Task'] == 'Survival Analysis') & (results_df['Metric'] == 'AUC')]['Value'].max():.3f}")
    
    return merged_data, X_features, results_df, feature_importance

# Execute the complete pipeline
if __name__ == "__main__":
    merged_data, X_features, results_df, feature_importance = main()
