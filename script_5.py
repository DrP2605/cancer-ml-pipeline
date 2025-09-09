# Demonstration of the Complete Cancer ML Pipeline
# Simplified version to show core functionality

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*80)
print("DEMONSTRATION: COMPLETE CANCER ML PIPELINE")
print("="*80)

# 1. Generate Sample Cancer Data
def generate_demo_data(n_samples=500, n_genes=100, n_mutations=15):
    print(f"\n1. GENERATING DEMO CANCER DATASET")
    print("-" * 40)
    
    sample_ids = [f'TCGA-{i:04d}' for i in range(n_samples)]
    cancer_types = np.random.choice(['BRCA', 'LUAD', 'COAD', 'LGG', 'KIRC'], n_samples)
    
    # RNA-seq data
    rna_data = {'sample_id': sample_ids, 'cancer_type': cancer_types}
    
    for i in range(n_genes):
        gene_name = f'GENE_{i+1:03d}'
        base_expr = np.random.lognormal(mean=4, sigma=1.5, size=n_samples)
        
        # Cancer-type specific effects
        if i < 20:
            cancer_effect = np.where(cancer_types == 'BRCA', 2.0,
                           np.where(cancer_types == 'LUAD', 1.5, 1.0))
            base_expr *= cancer_effect
        
        rna_data[gene_name] = np.log2(base_expr + 1)
    
    rna_df = pd.DataFrame(rna_data)
    
    # Mutation data
    key_genes = ['TP53', 'KRAS', 'PIK3CA', 'BRAF', 'PTEN', 'APC', 'EGFR', 
                 'IDH1', 'CDKN2A', 'RB1', 'BRCA1', 'BRCA2', 'ATM', 'MLH1', 'MSH2']
    mutation_data = {'sample_id': sample_ids}
    
    for gene in key_genes[:n_mutations]:
        freq = 0.3 if gene == 'TP53' else 0.15
        mutations = np.random.binomial(1, freq, n_samples)
        mutation_data[f'{gene}_mutation'] = mutations
    
    mutation_df = pd.DataFrame(mutation_data)
    
    # Clinical survival data
    age = np.clip(np.random.normal(65, 12, n_samples), 25, 90)
    stage = np.random.choice(['I', 'II', 'III', 'IV'], n_samples)
    
    # Survival time based on multiple factors
    base_time = np.random.exponential(scale=600, size=n_samples)
    age_effect = (age - 65) / 15 * 0.3
    cancer_effect = np.where(cancer_types == 'LGG', -0.8,
                    np.where(cancer_types == 'BRCA', -0.2, 0.2))
    stage_effect = np.where(stage == 'I', -0.6,
                   np.where(stage == 'II', -0.2, 0.4))
    
    # High-risk signature from first 10 genes
    high_risk_score = np.mean([rna_data[f'GENE_{i+1:03d}'] for i in range(10)], axis=0)
    molecular_effect = (high_risk_score - np.mean(high_risk_score)) / np.std(high_risk_score) * 0.3
    
    tp53_effect = mutation_data['TP53_mutation'] * 0.4
    
    log_hazard = age_effect + cancer_effect + stage_effect + molecular_effect + tp53_effect
    hazard_ratio = np.exp(log_hazard)
    survival_time = base_time / hazard_ratio
    survival_time = np.clip(survival_time, 30, 3000)
    
    # Censoring
    follow_up = np.random.uniform(400, 2000, n_samples)
    observed_time = np.minimum(survival_time, follow_up)
    vital_status = (survival_time <= follow_up).astype(int)
    
    clinical_data = {
        'sample_id': sample_ids,
        'age': age,
        'stage': stage,
        'survival_time': observed_time,
        'vital_status': vital_status,
        'high_risk_score': high_risk_score
    }
    clinical_df = pd.DataFrame(clinical_data)
    
    # Drug response data
    drugs = ['Cisplatin', 'Paclitaxel', 'Erlotinib', 'Sorafenib', 'Gemcitabine']
    drug_data = {'sample_id': sample_ids}
    
    for drug in drugs:
        base_ic50 = np.random.normal(1.5, 1.0, n_samples)
        
        if drug == 'Erlotinib':
            egfr_effect = -mutation_data.get('EGFR_mutation', np.zeros(n_samples)) * 1.2
        elif drug == 'Cisplatin':
            tp53_effect_drug = mutation_data['TP53_mutation'] * 0.5
            egfr_effect = tp53_effect_drug
        else:
            egfr_effect = 0
            
        cancer_effect_drug = np.where(cancer_types == 'BRCA', -0.3 if drug == 'Cisplatin' else 0,
                             np.where(cancer_types == 'LUAD', -0.4 if drug == 'Erlotinib' else 0, 0))
        
        final_ic50 = base_ic50 + egfr_effect + cancer_effect_drug + np.random.normal(0, 0.3, n_samples)
        drug_data[f'{drug}_IC50'] = final_ic50
    
    drug_df = pd.DataFrame(drug_data)
    
    print(f"✓ Generated {n_samples} samples with {n_genes} genes, {n_mutations} mutations")
    print(f"✓ Cancer types: {dict(pd.Series(cancer_types).value_counts())}")
    print(f"✓ Survival events: {vital_status.sum()} deaths, {(1-vital_status).sum()} censored")
    
    return rna_df, mutation_df, clinical_df, drug_df

# 2. Data Preprocessing
def preprocess_data(rna_df, mutation_df, clinical_df, drug_df):
    print(f"\n2. DATA PREPROCESSING")
    print("-" * 40)
    
    # Merge datasets
    merged = rna_df.merge(mutation_df, on='sample_id')
    merged = merged.merge(clinical_df, on='sample_id')
    merged = merged.merge(drug_df, on='sample_id')
    
    # Identify feature types
    gene_cols = [col for col in merged.columns if col.startswith('GENE_')]
    mutation_cols = [col for col in merged.columns if col.endswith('_mutation')]
    drug_cols = [col for col in merged.columns if col.endswith('_IC50')]
    
    # Feature selection: top 50 most variable genes
    gene_variance = merged[gene_cols].var()
    top_genes = gene_variance.nlargest(50).index.tolist()
    
    # Create feature matrix
    merged['mutation_burden'] = merged[mutation_cols].sum(axis=1)
    
    feature_cols = top_genes + mutation_cols + ['age', 'cancer_type', 'stage', 'high_risk_score', 'mutation_burden']
    X_features = merged[feature_cols].copy()
    
    # Encode categorical variables
    le_cancer = LabelEncoder()
    le_stage = LabelEncoder()
    
    X_features['cancer_type_encoded'] = le_cancer.fit_transform(X_features['cancer_type'])
    X_features['stage_encoded'] = le_stage.fit_transform(X_features['stage'])
    X_features = X_features.drop(['cancer_type', 'stage'], axis=1)
    
    print(f"✓ Final feature matrix: {X_features.shape}")
    print(f"✓ Features: {len(top_genes)} genes + {len(mutation_cols)} mutations + 4 clinical")
    
    return merged, X_features, top_genes, mutation_cols, drug_cols

# 3. Survival Analysis
def survival_analysis(merged, X_features):
    print(f"\n3. SURVIVAL ANALYSIS")
    print("-" * 40)
    
    y_survival = merged['vital_status']
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y_survival, test_size=0.2, random_state=42, stratify=y_survival
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = {}
    
    # Random Forest
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_clf.fit(X_train_scaled, y_train)
    
    rf_pred = rf_clf.predict(X_test_scaled)
    rf_prob = rf_clf.predict_proba(X_test_scaled)[:, 1]
    rf_acc = accuracy_score(y_test, rf_pred)
    rf_auc = roc_auc_score(y_test, rf_prob)
    
    results['Random_Forest'] = {'accuracy': rf_acc, 'auc': rf_auc}
    
    # Logistic Regression
    lr_clf = LogisticRegression(random_state=42, max_iter=1000)
    lr_clf.fit(X_train_scaled, y_train)
    
    lr_pred = lr_clf.predict(X_test_scaled)
    lr_prob = lr_clf.predict_proba(X_test_scaled)[:, 1]
    lr_acc = accuracy_score(y_test, lr_pred)
    lr_auc = roc_auc_score(y_test, lr_prob)
    
    results['Logistic_Regression'] = {'accuracy': lr_acc, 'auc': lr_auc}
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_features.columns,
        'importance': rf_clf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"Random Forest - AUC: {rf_auc:.3f}, Accuracy: {rf_acc:.3f}")
    print(f"Logistic Regression - AUC: {lr_auc:.3f}, Accuracy: {lr_acc:.3f}")
    print(f"Top 5 features: {', '.join(feature_importance.head(5)['feature'])}")
    
    return results, feature_importance

# 4. Drug Response Prediction
def drug_response_prediction(merged, X_features, drug_cols):
    print(f"\n4. DRUG RESPONSE PREDICTION")
    print("-" * 40)
    
    drug_results = {}
    
    for drug_col in drug_cols[:3]:  # First 3 drugs
        drug_name = drug_col.replace('_IC50', '')
        print(f"\n4.{drug_cols.index(drug_col)+1} {drug_name}")
        
        y_drug = merged[drug_col]
        X_train, X_test, y_train, y_test = train_test_split(
            X_features, y_drug, test_size=0.2, random_state=42
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Random Forest
        rf_reg = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf_reg.fit(X_train_scaled, y_train)
        rf_pred = rf_reg.predict(X_test_scaled)
        
        rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
        rf_r2 = r2_score(y_test, rf_pred)
        rf_corr = np.corrcoef(y_test, rf_pred)[0,1]
        
        # Elastic Net
        en_reg = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
        en_reg.fit(X_train_scaled, y_train)
        en_pred = en_reg.predict(X_test_scaled)
        
        en_rmse = np.sqrt(mean_squared_error(y_test, en_pred))
        en_r2 = r2_score(y_test, en_pred)
        en_corr = np.corrcoef(y_test, en_pred)[0,1]
        
        drug_results[drug_name] = {
            'Random_Forest': {'rmse': rf_rmse, 'r2': rf_r2, 'correlation': rf_corr},
            'Elastic_Net': {'rmse': en_rmse, 'r2': en_r2, 'correlation': en_corr}
        }
        
        print(f"  RF: RMSE={rf_rmse:.3f}, R²={rf_r2:.3f}, r={rf_corr:.3f}")
        print(f"  EN: RMSE={en_rmse:.3f}, R²={en_r2:.3f}, r={en_corr:.3f}")
    
    return drug_results

# 5. Create Visualization
def create_visualization(merged, feature_importance, survival_results, drug_results):
    print(f"\n5. CREATING VISUALIZATIONS")
    print("-" * 40)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Cancer ML Pipeline - Demo Results', fontsize=14, fontweight='bold')
    
    # 1. Cancer type distribution
    cancer_counts = merged['cancer_type'].value_counts()
    axes[0,0].pie(cancer_counts.values, labels=cancer_counts.index, autopct='%1.1f%%')
    axes[0,0].set_title('Cancer Type Distribution')
    
    # 2. Survival by cancer type
    survival_by_cancer = merged.groupby('cancer_type')['vital_status'].mean()
    axes[0,1].bar(survival_by_cancer.index, survival_by_cancer.values)
    axes[0,1].set_title('Death Rate by Cancer Type')
    axes[0,1].set_ylabel('Death Rate')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # 3. Top features
    top_features = feature_importance.head(10)
    axes[0,2].barh(range(len(top_features)), top_features['importance'])
    axes[0,2].set_yticks(range(len(top_features)))
    axes[0,2].set_yticklabels(top_features['feature'], fontsize=8)
    axes[0,2].set_title('Top 10 Important Features')
    axes[0,2].invert_yaxis()
    
    # 4. Survival model performance
    models = list(survival_results.keys())
    aucs = [survival_results[m]['auc'] for m in models]
    axes[1,0].bar(models, aucs)
    axes[1,0].set_title('Survival Model AUC')
    axes[1,0].set_ylabel('AUC Score')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # 5. Drug response correlations
    drug_corrs = []
    drug_names = []
    for drug, models in drug_results.items():
        for model, metrics in models.items():
            drug_corrs.append(metrics['correlation'])
            drug_names.append(f"{drug}\n{model}")
    
    axes[1,1].bar(range(len(drug_corrs)), drug_corrs)
    axes[1,1].set_xticks(range(len(drug_corrs)))
    axes[1,1].set_xticklabels(drug_names, fontsize=8, rotation=45)
    axes[1,1].set_title('Drug Response Correlations')
    axes[1,1].set_ylabel('Correlation')
    
    # 6. Age vs survival time
    axes[1,2].scatter(merged['age'], merged['survival_time'], alpha=0.6, 
                     c=merged['vital_status'], cmap='RdYlBu')
    axes[1,2].set_xlabel('Age')
    axes[1,2].set_ylabel('Survival Time (days)')
    axes[1,2].set_title('Age vs Survival Time')
    
    plt.tight_layout()
    plt.savefig('demo_cancer_analysis.png', dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved as 'demo_cancer_analysis.png'")
    
    return fig

# Execute the demo pipeline
print("\n🚀 EXECUTING DEMO PIPELINE...")

# Generate data
rna_df, mutation_df, clinical_df, drug_df = generate_demo_data()

# Preprocess
merged_data, X_features, gene_cols, mutation_cols, drug_cols = preprocess_data(
    rna_df, mutation_df, clinical_df, drug_df
)

# Survival analysis
survival_results, feature_importance = survival_analysis(merged_data, X_features)

# Drug response prediction
drug_results = drug_response_prediction(merged_data, X_features, drug_cols)

# Create visualization
fig = create_visualization(merged_data, feature_importance, survival_results, drug_results)

# Save results
merged_data.to_csv('demo_cancer_dataset.csv', index=False)
feature_importance.to_csv('demo_feature_importance.csv', index=False)

print(f"\n" + "="*60)
print("DEMO PIPELINE COMPLETED SUCCESSFULLY!")
print("="*60)
print(f"📊 Results Summary:")
print(f"   - Dataset: {len(merged_data)} samples, {X_features.shape[1]} features")

best_survival_auc = max(survival_results[m]['auc'] for m in survival_results)
print(f"   - Best Survival AUC: {best_survival_auc:.3f}")

avg_drug_corr = np.mean([drug_results[d][m]['correlation'] for d in drug_results for m in drug_results[d]])
print(f"   - Avg Drug Response r: {avg_drug_corr:.3f}")

print(f"\n📁 Files Generated:")
print(f"   - demo_cancer_dataset.csv")
print(f"   - demo_feature_importance.csv")
print(f"   - demo_cancer_analysis.png")

print(f"\n✨ This demonstrates the complete pipeline!")
print(f"The full version supports:")
print(f"   • 2000+ samples with 1000+ genes")
print(f"   • 15+ advanced ML algorithms")
print(f"   • Deep learning multi-omics integration")
print(f"   • Cox proportional hazards models")
print(f"   • Comprehensive ensemble methods")
print(f"   • Advanced feature engineering")
print(f"   • Interactive visualizations")