# ============================================================================
# 3. COMPREHENSIVE SURVIVAL ANALYSIS
# ============================================================================

def comprehensive_survival_analysis(merged_data, X_features):
    """Multiple survival analysis approaches"""
    
    print(f"\n3. COMPREHENSIVE SURVIVAL ANALYSIS")
    print("-" * 60)
    
    # Prepare survival targets
    y_survival = merged_data['vital_status']
    survival_time = merged_data['survival_time']
    
    # Stratified split by cancer type and vital status
    X_train, X_test, y_train, y_test, time_train, time_test = train_test_split(
        X_features, y_survival, survival_time, 
        test_size=0.2, random_state=42, 
        stratify=pd.concat([merged_data['cancer_type'], y_survival], axis=1)
    )
    
    print(f"✓ Training samples: {len(X_train)} ({y_train.sum()} events)")
    print(f"✓ Test samples: {len(X_test)} ({y_test.sum()} events)")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = {}
    
    # 3.1 Random Forest Survival Classification
    print(f"\n3.1 Random Forest Survival Classification")
    rf_clf = RandomForestClassifier(n_estimators=200, max_depth=10, 
                                   min_samples_split=10, random_state=42, n_jobs=-1)
    rf_clf.fit(X_train_scaled, y_train)
    
    rf_pred = rf_clf.predict(X_test_scaled)
    rf_prob = rf_clf.predict_proba(X_test_scaled)[:, 1]
    
    rf_acc = accuracy_score(y_test, rf_pred)
    rf_auc = roc_auc_score(y_test, rf_prob)
    
    # Cross-validation
    rf_cv_scores = cross_val_score(rf_clf, X_train_scaled, y_train, cv=5, scoring='roc_auc')
    
    results['Random_Forest'] = {
        'accuracy': rf_acc,
        'auc': rf_auc,
        'cv_auc_mean': rf_cv_scores.mean(),
        'cv_auc_std': rf_cv_scores.std()
    }
    
    print(f"  - Test AUC: {rf_auc:.3f}, Accuracy: {rf_acc:.3f}")
    print(f"  - CV AUC: {rf_cv_scores.mean():.3f} ± {rf_cv_scores.std():.3f}")
    
    # Feature importance analysis
    feature_importance = pd.DataFrame({
        'feature': X_features.columns,
        'importance': rf_clf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"  - Top 5 features: {', '.join(feature_importance.head(5)['feature'].tolist())}")
    
    # 3.2 Logistic Regression with Regularization
    print(f"\n3.2 Logistic Regression (L1+L2 Regularization)")
    
    # Try different regularization strengths
    best_auc = 0
    best_lr = None
    
    for C in [0.001, 0.01, 0.1, 1.0, 10.0]:
        lr = LogisticRegression(C=C, penalty='elasticnet', l1_ratio=0.5, 
                               solver='saga', max_iter=1000, random_state=42)
        try:
            lr.fit(X_train_scaled, y_train)
            lr_prob_val = lr.predict_proba(X_train_scaled)[:, 1]
            cv_auc = cross_val_score(lr, X_train_scaled, y_train, cv=3, scoring='roc_auc').mean()
            
            if cv_auc > best_auc:
                best_auc = cv_auc
                best_lr = lr
        except:
            continue
    
    if best_lr:
        lr_pred = best_lr.predict(X_test_scaled)
        lr_prob = best_lr.predict_proba(X_test_scaled)[:, 1]
        lr_acc = accuracy_score(y_test, lr_pred)
        lr_auc = roc_auc_score(y_test, lr_prob)
        
        results['Logistic_Regression'] = {
            'accuracy': lr_acc,
            'auc': lr_auc,
            'cv_auc': best_auc
        }
        
        print(f"  - Test AUC: {lr_auc:.3f}, Accuracy: {lr_acc:.3f}")
        print(f"  - Best CV AUC: {best_auc:.3f}")
    
    # 3.3 Support Vector Machine
    print(f"\n3.3 Support Vector Machine")
    
    # Use subset for SVM (computationally intensive)
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
    
    # 3.4 Simplified Cox-like Risk Score
    print(f"\n3.4 Risk Score Model (Cox-inspired)")
    
    # Calculate univariate hazard ratios approximation
    risk_scores = []
    
    for i, feature in enumerate(X_features.columns):
        feature_values = X_train_scaled[:, i]
        
        # Divide into high/low groups
        median_val = np.median(feature_values)
        high_group = feature_values >= median_val
        low_group = feature_values < median_val
        
        # Calculate "hazard ratio" as odds ratio approximation
        high_events = y_train[high_group].sum()
        high_total = len(y_train[high_group])
        low_events = y_train[low_group].sum()
        low_total = len(y_train[low_group])
        
        if high_total > 0 and low_total > 0 and low_events > 0 and (high_total - high_events) > 0:
            high_rate = high_events / high_total
            low_rate = low_events / low_total
            
            if low_rate > 0:
                hazard_ratio = high_rate / low_rate
                risk_scores.append(np.log(hazard_ratio))
            else:
                risk_scores.append(0)
        else:
            risk_scores.append(0)
    
    risk_scores = np.array(risk_scores)
    
    # Calculate combined risk score for test set
    test_risk_scores = np.dot(X_test_scaled, risk_scores)
    
    # Convert to probability using logistic function
    risk_probs = 1 / (1 + np.exp(-test_risk_scores))
    risk_pred = (risk_probs > 0.5).astype(int)
    
    risk_acc = accuracy_score(y_test, risk_pred)
    risk_auc = roc_auc_score(y_test, risk_probs)
    
    results['Risk_Score'] = {'accuracy': risk_acc, 'auc': risk_auc}
    print(f"  - Test AUC: {risk_auc:.3f}, Accuracy: {risk_acc:.3f}")
    
    # 3.5 Survival Time Regression (for uncensored cases)
    print(f"\n3.5 Survival Time Regression")
    
    # Focus on uncensored samples
    uncensored_train_idx = y_train == 1
    uncensored_test_idx = y_test == 1
    
    if uncensored_train_idx.sum() >= 50 and uncensored_test_idx.sum() >= 10:
        
        X_train_uncens = X_train_scaled[uncensored_train_idx]
        y_train_time = time_train.values[uncensored_train_idx]
        X_test_uncens = X_test_scaled[uncensored_test_idx]
        y_test_time = time_test.values[uncensored_test_idx]
        
        # Random Forest Regressor
        rf_reg = RandomForestRegressor(n_estimators=100, max_depth=10, 
                                      min_samples_split=5, random_state=42, n_jobs=-1)
        rf_reg.fit(X_train_uncens, y_train_time)
        rf_time_pred = rf_reg.predict(X_test_uncens)
        
        rf_rmse = np.sqrt(mean_squared_error(y_test_time, rf_time_pred))
        rf_r2 = r2_score(y_test_time, rf_time_pred)
        rf_corr = np.corrcoef(y_test_time, rf_time_pred)[0,1] if len(y_test_time) > 1 else 0
        
        # Ridge Regression
        ridge_reg = Ridge(alpha=1.0, random_state=42)
        ridge_reg.fit(X_train_uncens, y_train_time)
        ridge_time_pred = ridge_reg.predict(X_test_uncens)
        
        ridge_rmse = np.sqrt(mean_squared_error(y_test_time, ridge_time_pred))
        ridge_r2 = r2_score(y_test_time, ridge_time_pred)
        ridge_corr = np.corrcoef(y_test_time, ridge_time_pred)[0,1] if len(y_test_time) > 1 else 0
        
        results['RF_Time_Regression'] = {
            'rmse': rf_rmse, 'r2': rf_r2, 'correlation': rf_corr
        }
        results['Ridge_Time_Regression'] = {
            'rmse': ridge_rmse, 'r2': ridge_r2, 'correlation': ridge_corr
        }
        
        print(f"  - RF Time Regression: RMSE={rf_rmse:.1f}, R²={rf_r2:.3f}, r={rf_corr:.3f}")
        print(f"  - Ridge Time Regression: RMSE={ridge_rmse:.1f}, R²={ridge_r2:.3f}, r={ridge_corr:.3f}")
    
    else:
        print(f"  - Insufficient uncensored samples for time regression")
    
    return results, feature_importance, scaler

# ============================================================================
# 4. DRUG RESPONSE PREDICTION
# ============================================================================

def comprehensive_drug_response_prediction(merged_data, X_features, drug_cols):
    """Comprehensive drug response prediction across multiple drugs"""
    
    print(f"\n4. COMPREHENSIVE DRUG RESPONSE PREDICTION")
    print("-" * 60)
    
    # Select top drugs for detailed analysis
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
        
        # Scale features and targets
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)
        
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
        y_test_scaled = scaler_y.fit_transform(y_test.reshape(-1, 1)).ravel()
        
        drug_results = {}
        
        # 4.1 Random Forest
        rf_drug = RandomForestRegressor(n_estimators=100, max_depth=10,
                                       min_samples_split=5, random_state=42, n_jobs=-1)
        rf_drug.fit(X_train_scaled, y_train_scaled)
        rf_pred_scaled = rf_drug.predict(X_test_scaled)
        rf_pred = scaler_y.inverse_transform(rf_pred_scaled.reshape(-1, 1)).ravel()
        
        rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
        rf_r2 = r2_score(y_test, rf_pred)
        rf_corr = np.corrcoef(y_test, rf_pred)[0,1]
        
        drug_results['Random_Forest'] = {
            'rmse': rf_rmse, 'r2': rf_r2, 'correlation': rf_corr
        }
        
        # 4.2 Elastic Net
        elastic_drug = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42, max_iter=2000)
        elastic_drug.fit(X_train_scaled, y_train_scaled)
        elastic_pred_scaled = elastic_drug.predict(X_test_scaled)
        elastic_pred = scaler_y.inverse_transform(elastic_pred_scaled.reshape(-1, 1)).ravel()
        
        elastic_rmse = np.sqrt(mean_squared_error(y_test, elastic_pred))
        elastic_r2 = r2_score(y_test, elastic_pred)
        elastic_corr = np.corrcoef(y_test, elastic_pred)[0,1]
        
        drug_results['Elastic_Net'] = {
            'rmse': elastic_rmse, 'r2': elastic_r2, 'correlation': elastic_corr
        }
        
        # 4.3 Support Vector Regression
        if len(X_train_scaled) <= 1000:  # Use all data if small enough
            svr_drug = SVR(kernel='rbf', C=1.0, gamma='auto')
            svr_drug.fit(X_train_scaled, y_train_scaled)
            svr_pred_scaled = svr_drug.predict(X_test_scaled)
            svr_pred = scaler_y.inverse_transform(svr_pred_scaled.reshape(-1, 1)).ravel()
            
            svr_rmse = np.sqrt(mean_squared_error(y_test, svr_pred))
            svr_r2 = r2_score(y_test, svr_pred)
            svr_corr = np.corrcoef(y_test, svr_pred)[0,1]
            
            drug_results['SVR'] = {
                'rmse': svr_rmse, 'r2': svr_r2, 'correlation': svr_corr
            }
        
        # Drug-specific feature importance
        feature_importance_drug = pd.DataFrame({
            'feature': X_features.columns,
            'importance': rf_drug.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Print results
        print(f"  Random Forest: RMSE={rf_rmse:.3f}, R²={rf_r2:.3f}, r={rf_corr:.3f}")
        print(f"  Elastic Net:   RMSE={elastic_rmse:.3f}, R²={elastic_r2:.3f}, r={elastic_corr:.3f}")
        if 'SVR' in drug_results:
            print(f"  SVR:          RMSE={svr_rmse:.3f}, R²={svr_r2:.3f}, r={svr_corr:.3f}")
        
        top_features = feature_importance_drug.head(3)['feature'].tolist()
        print(f"  Top predictive features: {', '.join(top_features)}")
        
        all_drug_results[drug_name] = drug_results
    
    return all_drug_results

# ============================================================================
# 5. ENSEMBLE AND ADVANCED METHODS
# ============================================================================

def ensemble_methods(merged_data, X_features):
    """Advanced ensemble methods for improved prediction"""
    
    print(f"\n5. ENSEMBLE AND ADVANCED METHODS")
    print("-" * 60)
    
    # Prepare data
    y_survival = merged_data['vital_status']
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y_survival, test_size=0.2, random_state=42, stratify=y_survival
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    ensemble_results = {}
    
    # 5.1 Voting Classifier
    print(f"\n5.1 Voting Ensemble")
    
    from sklearn.ensemble import VotingClassifier
    
    # Base classifiers
    rf_base = RandomForestClassifier(n_estimators=100, random_state=42)
    lr_base = LogisticRegression(random_state=42, max_iter=1000)
    
    voting_clf = VotingClassifier(
        estimators=[('rf', rf_base), ('lr', lr_base)],
        voting='soft'
    )
    
    voting_clf.fit(X_train_scaled, y_train)
    voting_pred = voting_clf.predict(X_test_scaled)
    voting_prob = voting_clf.predict_proba(X_test_scaled)[:, 1]
    
    voting_acc = accuracy_score(y_test, voting_pred)
    voting_auc = roc_auc_score(y_test, voting_prob)
    
    ensemble_results['Voting_Classifier'] = {'accuracy': voting_acc, 'auc': voting_auc}
    print(f"  Voting Ensemble: AUC={voting_auc:.3f}, Accuracy={voting_acc:.3f}")
    
    # 5.2 Gradient Boosting (Manual implementation)
    print(f"\n5.2 Gradient Boosting")
    
    from sklearn.ensemble import GradientBoostingClassifier
    
    gb_clf = GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42
    )
    gb_clf.fit(X_train_scaled, y_train)
    gb_pred = gb_clf.predict(X_test_scaled)
    gb_prob = gb_clf.predict_proba(X_test_scaled)[:, 1]
    
    gb_acc = accuracy_score(y_test, gb_pred)
    gb_auc = roc_auc_score(y_test, gb_prob)
    
    ensemble_results['Gradient_Boosting'] = {'accuracy': gb_acc, 'auc': gb_auc}
    print(f"  Gradient Boosting: AUC={gb_auc:.3f}, Accuracy={gb_acc:.3f}")
    
    # 5.3 Feature Selection + Model Pipeline
    print(f"\n5.3 Feature Selection Pipeline")
    
    from sklearn.pipeline import Pipeline
    from sklearn.feature_selection import SelectKBest, f_classif
    
    # Create pipeline with feature selection
    pipe = Pipeline([
        ('selector', SelectKBest(f_classif, k=50)),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    pipe.fit(X_train_scaled, y_train)
    pipe_pred = pipe.predict(X_test_scaled)
    pipe_prob = pipe.predict_proba(X_test_scaled)[:, 1]
    
    pipe_acc = accuracy_score(y_test, pipe_pred)
    pipe_auc = roc_auc_score(y_test, pipe_prob)
    
    ensemble_results['Feature_Selection_Pipeline'] = {'accuracy': pipe_acc, 'auc': pipe_auc}
    print(f"  Feature Selection + RF: AUC={pipe_auc:.3f}, Accuracy={pipe_acc:.3f}")
    
    # Get selected features
    selected_features = pipe.named_steps['selector'].get_support()
    selected_feature_names = X_features.columns[selected_features]
    print(f"  Selected {len(selected_feature_names)} features")
    
    return ensemble_results

print("PART 2/3 LOADED: ML models and analysis functions ready")