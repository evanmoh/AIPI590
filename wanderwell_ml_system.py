

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import re
import warnings
warnings.filterwarnings('ignore')


try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️  XGBoost not installed. Will use RandomForest instead.")
    print("   To install: pip install xgboost")


class WanderWellMLSystem:
    """Complete ML system for travel recommendations"""
    
    def __init__(self, places_df, reviews_df):
        """Initialize with places and reviews data"""
        self.places_df = places_df.copy()
        self.reviews_df = reviews_df.copy()
        
        # Encoders
        self.traveler_encoder = LabelEncoder()
        self.place_encoder = LabelEncoder()
        
        # Text vectorizer
        self.text_vectorizer = TfidfVectorizer(
            max_features=50,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Models
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        
        # Feature names
        self.feature_names = []
        
        print("✓ WanderWell ML System initialized")
    
    def extract_text_features(self, text):
        """Extract features from review text"""
        text = str(text).lower()
        
        features = {
            # Positive sentiment
            'mentions_beautiful': int(bool(re.search(r'beautiful|gorgeous|stunning|lovely|amazing', text))),
            'mentions_peaceful': int(bool(re.search(r'peaceful|quiet|calm|relaxing|serene', text))),
            'mentions_fun': int(bool(re.search(r'fun|enjoy|exciting|great|wonderful', text))),
            'mentions_perfect': int(bool(re.search(r'perfect|ideal|excellent|fantastic|incredible', text))),
            
            # Negative sentiment
            'mentions_crowded': int(bool(re.search(r'crowded|busy|packed|too many', text))),
            'mentions_parking': int(bool(re.search(r'parking|park|lot', text))),
            'mentions_accessibility': int(bool(re.search(r'wheelchair|accessible|stairs|difficult|walk', text))),
            'mentions_expensive': int(bool(re.search(r'expensive|pricey|costly|$$', text))),
            'mentions_disappointing': int(bool(re.search(r'disappointing|disappoint|unfortunately|poor', text))),
            
            # Descriptive
            'mentions_kids': int(bool(re.search(r'kids|children|family|child', text))),
            'mentions_romantic': int(bool(re.search(r'romantic|date|couple|partner', text))),
            'mentions_educational': int(bool(re.search(r'learn|educational|history|museum', text))),
            'mentions_nature': int(bool(re.search(r'nature|garden|park|outdoor|trees', text))),
            
            # Length and enthusiasm
            'text_length': len(text),
            'exclamation_marks': text.count('!'),
        }
        
        return features
    
    def prepare_training_data(self):
        """Prepare features and target for model training"""
        print("\n" + "=" * 80)
        print("PHASE 1: FEATURE ENGINEERING")
        print("=" * 80)
        
        # Extract text features for all reviews
        print("\n1. Extracting text features from reviews...")
        text_features_list = []
        for text in self.reviews_df['review_text']:
            text_features_list.append(self.extract_text_features(text))
        
        text_features_df = pd.DataFrame(text_features_list)
        print(f"   ✓ Extracted {len(text_features_df.columns)} text features")
        
        # Merge reviews with places
        print("\n2. Merging reviews with place attributes...")
        data = self.reviews_df.merge(
            self.places_df,
            left_on='place_id',
            right_on='places_id',
            how='left'
        )
        
        # Add text features
        data = pd.concat([data, text_features_df], axis=1)
        
        # Encode categorical variables
        print("\n3. Encoding categorical variables...")
        data['traveler_type_encoded'] = self.traveler_encoder.fit_transform(data['traveler_type'])
        data['place_id_encoded'] = self.place_encoder.fit_transform(data['place_id'])
        
        # TF-IDF on review text 
        print("\n4. Creating TF-IDF features from review text...")
        tfidf_matrix = self.text_vectorizer.fit_transform(data['review_text'])
        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=[f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])]
        )
        data = pd.concat([data, tfidf_df], axis=1)
        
        # Select feature columns
        feature_cols = [
            'traveler_type_encoded',
            'place_id_encoded',
            'duration_hours',
            'avg_cost_usd',
            'morning_fit',
            'lunch_fit',
            'evening_fit',
            'overall_rating',
        ]
        
        # Add place vibe flags if they exist
        vibe_cols = ['is_mixed', 'is_cultural', 'is_chill', 'is_adventure']
        for col in vibe_cols:
            if col in data.columns:
                feature_cols.append(col)
        
        # Add text features
        feature_cols.extend(text_features_df.columns.tolist())
        
        # Add TF-IDF features
        feature_cols.extend(tfidf_df.columns.tolist())
        
        # Prepare X and y
        X = data[feature_cols].fillna(0)
        y = data['review_star']
        groups = data['place_id']  # For group-based cross-validation
        
        self.feature_names = feature_cols
        
        print(f"\n✓ Training data prepared:")
        print(f"  - {len(X)} samples")
        print(f"  - {len(feature_cols)} features")
        print(f"  - {data['place_id'].nunique()} unique places")
        print(f"  - Target: review_star (1-5)")
        
        return X, y, groups, data
    
    def train_models(self, X, y, groups):
        """Train multiple models with cross-validation"""
        print("\n" + "=" * 80)
        print("PHASE 2: MODEL TRAINING")
        print("=" * 80)
        
        # Group K-Fold: ensures same place stays in same fold
        gkf = GroupKFold(n_splits=5)
        
        # Define models to train
        models_to_train = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        }
        
        if XGBOOST_AVAILABLE:
            models_to_train['XGBoost'] = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        
        results = {}
        
        for model_name, model in models_to_train.items():
            print(f"\n{'=' * 40}")
            print(f"Training: {model_name}")
            print(f"{'=' * 40}")
            
            fold_maes = []
            fold_rmses = []
            fold_r2s = []
            
            for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), 1):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                # Train
                model.fit(X_train, y_train)
                
                # Predict
                y_pred = model.predict(X_test)
                
                # Clip predictions to 1-5 range
                y_pred = np.clip(y_pred, 1, 5)
                
                # Metrics
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
                
                fold_maes.append(mae)
                fold_rmses.append(rmse)
                fold_r2s.append(r2)
                
                print(f"  Fold {fold}: MAE={mae:.3f}, RMSE={rmse:.3f}, R²={r2:.3f}")
            
            # Average metrics
            avg_mae = np.mean(fold_maes)
            avg_rmse = np.mean(fold_rmses)
            avg_r2 = np.mean(fold_r2s)
            
            results[model_name] = {
                'model': model,
                'mae': avg_mae,
                'rmse': avg_rmse,
                'r2': avg_r2,
                'fold_maes': fold_maes
            }
            
            print(f"\n  Average: MAE={avg_mae:.3f}, RMSE={avg_rmse:.3f}, R²={avg_r2:.3f}")
        
        # Select best model (lowest MAE)
        best_name = min(results, key=lambda k: results[k]['mae'])
        self.best_model_name = best_name
        self.best_model = results[best_name]['model']
        
        # Retrain best model on full data
        print(f"\n{'=' * 80}")
        print(f"BEST MODEL: {best_name} (MAE: {results[best_name]['mae']:.3f})")
        print(f"{'=' * 80}")
        print(f"\nRetraining {best_name} on full dataset...")
        self.best_model.fit(X, y)
        
        self.models = results
        
        print(f"✓ Model training complete!")
        
        return results
    
    def get_feature_importance(self, top_n=20):
        """Get feature importance from best model"""
        if self.best_model is None:
            return None
        
        # Get feature importance
        if hasattr(self.best_model, 'feature_importances_'):
            # Tree-based models
            importances = self.best_model.feature_importances_
        elif hasattr(self.best_model, 'coef_'):
            # Linear models
            importances = np.abs(self.best_model.coef_).flatten()
        else:
            return None
        
        # Create dataframe
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(top_n)
        
        return feature_importance_df
    
    def get_top_contributing_features(self, feature_values, top_n=2):
        """Get top N features contributing to this prediction (for explainability)"""
        if self.best_model is None:
            return []
        
        # Get feature importance from model
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
        elif hasattr(self.best_model, 'coef_'):
            importances = np.abs(self.best_model.coef_).flatten()
        else:
            return []
        
        # Calculate contribution = importance × feature_value
        contributions = []
        for i, (feature_name, importance) in enumerate(zip(self.feature_names, importances)):
            feature_value = feature_values.iloc[0, i]
            
            # Only include features that are "active" (non-zero) and important
            if feature_value > 0 and importance > 0.01:
                contribution = importance * feature_value
                contributions.append({
                    'feature': feature_name,
                    'importance': importance,
                    'value': feature_value,
                    'contribution': contribution
                })
        
        # Sort by contribution
        contributions.sort(key=lambda x: x['contribution'], reverse=True)
        
        return contributions[:top_n]
    
    def explain_features_human_readable(self, features, vibe_match_score=0):
        """Convert feature names to human-readable explanations"""
        explanations = {
            'overall_rating': 'Highly rated by previous visitors',
            'is_chill': 'Perfect for a relaxed, peaceful experience',
            'is_cultural': 'Rich cultural and educational experience',
            'is_adventure': 'Exciting and adventurous activity',
            'is_mixed': 'Offers diverse experiences',
            'morning_fit': 'Ideal for morning visits',
            'lunch_fit': 'Great lunch spot',
            'evening_fit': 'Perfect for evening activities',
            'mentions_beautiful': 'Visitors describe it as beautiful',
            'mentions_peaceful': 'Known for peaceful atmosphere',
            'mentions_fun': 'Highly enjoyable according to reviews',
            'mentions_perfect': 'Visitors say it\'s perfect',
            'mentions_romantic': 'Popular with couples',
            'mentions_kids': 'Great for families with children',
            'mentions_educational': 'Educational and informative',
            'mentions_nature': 'Beautiful natural setting',
            'duration_hours': 'Right amount of time to spend',
            'avg_cost_usd': 'Good value for money',
        }
        
        readable_reasons = []
        
        # PRIORITY: If vibe matches, this should be reason #1
        if vibe_match_score >= 1.0:
            readable_reasons.append('Matches your preferred vibe perfectly')
        elif vibe_match_score >= 0.7:
            readable_reasons.append('Versatile for your travel style')
        
        # Add feature-based reasons
        for feature_info in features:
            feature_name = feature_info['feature']
            
            # Try exact match first
            if feature_name in explanations:
                readable_reasons.append(explanations[feature_name])
            # Handle traveler type matches
            elif 'traveler_type' in feature_name:
                readable_reasons.append('Popular with similar travelers')
            # Handle TF-IDF features (skip these for user display)
            elif feature_name.startswith('tfidf_'):
                continue
            else:
                # Generic fallback
                clean_name = feature_name.replace('_', ' ').replace('mentions ', '').title()
                readable_reasons.append(f'Good {clean_name.lower()}')
        
        return readable_reasons
    
    def calculate_vibe_match_score(self, user_profile, place_row):
        """Calculate how well place matches user's requested vibe"""
        user_vibe = user_profile.get('vibe', 'Mixed').lower()
        
        # Check if place has the requested vibe
        vibe_column = f'is_{user_vibe}'
        
        # Get place vibes
        place_vibes = {
            'is_mixed': place_row.get('is_mixed', 0),
            'is_cultural': place_row.get('is_cultural', 0),
            'is_chill': place_row.get('is_chill', 0),
            'is_adventure': place_row.get('is_adventure', 0)
        }
        
        # Perfect match: place has exactly the vibe user wants
        if vibe_column in place_vibes and place_vibes[vibe_column] == 1:
            return 1.0
        
        # Partial match: place is "mixed" and user wants any vibe
        if place_vibes['is_mixed'] == 1 and user_vibe != 'mixed':
            return 0.7  # Mixed places are versatile, good for any vibe
        
        # No match
        return 0.0
    
    def predict_rating(self, user_profile, place_row):
        """Predict rating for a user-place pair using HYBRID approach"""
        if self.best_model is None:
            raise ValueError("Model not trained yet!")
        
        # ========================================
        # COMPONENT 1: ML PREDICTION (50%)
        # ========================================
        
        # Create feature vector
        features = {}
        
        # Encode traveler type
        traveler_type = user_profile.get('companions', 'Solo')
        if traveler_type in self.traveler_encoder.classes_:
            features['traveler_type_encoded'] = self.traveler_encoder.transform([traveler_type])[0]
        else:
            features['traveler_type_encoded'] = 0
        
        # Encode place
        place_id = place_row['places_id']
        if place_id in self.place_encoder.classes_:
            features['place_id_encoded'] = self.place_encoder.transform([place_id])[0]
        else:
            features['place_id_encoded'] = 0
        
        # Place attributes
        features['duration_hours'] = place_row.get('duration_hours', 0)
        features['avg_cost_usd'] = place_row.get('avg_cost_usd', 0)
        features['morning_fit'] = place_row.get('morning_fit', 0)
        features['lunch_fit'] = place_row.get('lunch_fit', 0)
        features['evening_fit'] = place_row.get('evening_fit', 0)
        features['overall_rating'] = place_row.get('overall_rating', 3.0)
        
        # Vibe flags
        for col in ['is_mixed', 'is_cultural', 'is_chill', 'is_adventure']:
            features[col] = place_row.get(col, 0)
        
        # Text features (set to average values since we don't have specific review)
        text_feature_cols = [col for col in self.feature_names if col.startswith('mentions_') or col in ['text_length', 'exclamation_marks']]
        for col in text_feature_cols:
            features[col] = 0  # Neutral
        
        # TF-IDF features (set to zero)
        tfidf_cols = [col for col in self.feature_names if col.startswith('tfidf_')]
        for col in tfidf_cols:
            features[col] = 0
        
        # Create feature array in correct order
        X_pred = pd.DataFrame([features])[self.feature_names].fillna(0)
        
        # ML prediction (1-5 scale)
        ml_predicted_rating = self.best_model.predict(X_pred)[0]
        ml_predicted_rating = np.clip(ml_predicted_rating, 1, 5)
        
        # ========================================
        # COMPONENT 2: VIBE MATCH SCORE (50%)
        # ========================================
        
        vibe_match = self.calculate_vibe_match_score(user_profile, place_row)
        vibe_score = vibe_match * 5.0  # Convert 0-1 to 0-5 scale
        
        # ========================================
        # HYBRID FINAL SCORE: 50% ML + 50% VIBE
        # ========================================
        
        final_rating = 0.5 * ml_predicted_rating + 0.5 * vibe_score
        final_rating = np.clip(final_rating, 1, 5)
        
        return final_rating, X_pred
    
    def generate_recommendations(self, user_profile, top_k=10):
        """Generate ranked recommendations for a user"""
        city = user_profile.get('destination', 'NYC')
        time_slot = user_profile.get('time_slot', 'morning')  # morning, lunch, evening
        
        # Filter places by city
        available_places = self.places_df[self.places_df['city'] == city].copy()
        
        # Filter by time fit
        time_col = f'{time_slot}_fit'
        available_places = available_places[available_places[time_col] > 0]
        
        if len(available_places) == 0:
            return pd.DataFrame()
        
        # Predict rating for each place
        predictions = []
        for idx, place_row in available_places.iterrows():
            # HYBRID prediction (50% ML + 50% vibe match)
            predicted_rating, feature_values = self.predict_rating(user_profile, place_row)
            
            # Calculate vibe match for reasoning
            vibe_match_score = self.calculate_vibe_match_score(user_profile, place_row)
            
            # Get top contributing features
            top_features = self.get_top_contributing_features(feature_values, top_n=2)
            
            # Convert to human-readable reasons (pass vibe match score)
            reasons = self.explain_features_human_readable(top_features, vibe_match_score)
            
            # Ensure we have exactly 2 reasons (pad if necessary)
            while len(reasons) < 2:
                reasons.append('Recommended based on your preferences')
            reasons = reasons[:2]  # Take only top 2
            
            predictions.append({
                'place_id': place_row['places_id'],
                'name': place_row['name'],
                'category': place_row['category'],
                'predicted_rating': predicted_rating,
                'cost': place_row['avg_cost_usd'],
                'duration': place_row['duration_hours'],
                'overall_rating': place_row.get('overall_rating', 0),
                'reason_1': reasons[0],
                'reason_2': reasons[1],
                'vibe_match': vibe_match_score,  # For debugging
            })
        
        # Create dataframe and sort
        results_df = pd.DataFrame(predictions).sort_values('predicted_rating', ascending=False).head(top_k)
        
        return results_df


def main():
    """Main execution"""
    print("=" * 80)
    print("WANDERWELL ML SYSTEM - 100% ML APPROACH")
    print("=" * 80)
    
    # Load data
    print("\nLoading data...")
    
    import os
    
    # Try multiple locations
    if os.path.exists('Places.xlsx'):
        places_df = pd.read_excel('Places.xlsx')
        reviews_df = pd.read_excel('Reviews.xlsx')
    elif os.path.exists('/mnt/project/Places.xlsx'):
        places_df = pd.read_excel('/mnt/project/Places.xlsx')
        reviews_df = pd.read_excel('/mnt/project/Reviews.xlsx')
    else:
        raise FileNotFoundError(
            "Cannot find Places.xlsx and Reviews.xlsx!\n"
            "Please make sure these files are in the same folder as this script."
        )
    
    # Check if we have the enhanced places file with vibe tags
    try:
        if os.path.exists('Places_Enhanced.xlsx'):
            places_enhanced = pd.read_excel('Places_Enhanced.xlsx')
        elif os.path.exists('/mnt/user-data/outputs/Places_Enhanced.xlsx'):
            places_enhanced = pd.read_excel('/mnt/user-data/outputs/Places_Enhanced.xlsx')
        else:
            places_enhanced = None
            
        if places_enhanced is not None and all(col in places_enhanced.columns for col in ['is_mixed', 'is_cultural', 'is_chill', 'is_adventure']):
            places_df = places_enhanced
            print("✓ Using enhanced places data with vibe tags")
        else:
            print("⚠️  Vibe tags not found in enhanced file, using original")
    except:
        print("⚠️  Enhanced places file not found, using original")
    
    print(f"✓ Loaded {len(places_df)} places")
    print(f"✓ Loaded {len(reviews_df)} reviews")
    
    # Initialize system
    ml_system = WanderWellMLSystem(places_df, reviews_df)
    
    # Prepare data
    X, y, groups, full_data = ml_system.prepare_training_data()
    
    # Train models
    results = ml_system.train_models(X, y, groups)
    
    # Feature importance
    print("\n" + "=" * 80)
    print("PHASE 3: FEATURE IMPORTANCE ANALYSIS")
    print("=" * 80)
    
    feature_importance = ml_system.get_feature_importance(top_n=20)
    if feature_importance is not None:
        print("\nTop 20 Most Important Features:")
        print(feature_importance.to_string(index=False))
    
    # Save model performance
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    
    performance_data = []
    for model_name, result in results.items():
        performance_data.append({
            'Model': model_name,
            'MAE': result['mae'],
            'RMSE': result['rmse'],
            'R²': result['r2'],
            'Is_Best': '✓' if model_name == ml_system.best_model_name else ''
        })
    
    performance_df = pd.DataFrame(performance_data)
    
    # Save to Excel
    output_path = '/mnt/user-data/outputs/WanderWell_ML_Model_Performance.xlsx'
    
    # Use local path if not in Claude environment
    import os
    if not os.path.exists('/mnt/user-data/outputs'):
        output_path = 'WanderWell_ML_Model_Performance.xlsx'
    
    with pd.ExcelWriter(output_path) as writer:
        performance_df.to_excel(writer, sheet_name='Model_Performance', index=False)
        if feature_importance is not None:
            feature_importance.to_excel(writer, sheet_name='Feature_Importance', index=False)
    
    print(f"\n✓ Model performance saved to: {output_path}")
    
    # Test recommendation
    print("\n" + "=" * 80)
    print("PHASE 4: TESTING RECOMMENDATIONS")
    print("=" * 80)
    
    test_profile = {
        'destination': 'Durham',
        'companions': 'Couples',
        'vibe': 'Chill',  # Now actually used!
        'time_slot': 'morning'
    }
    
    print(f"\nTest Query: {test_profile}")
    recommendations = ml_system.generate_recommendations(test_profile, top_k=5)
    
    if len(recommendations) > 0:
        print("\nTop 5 Recommendations (HYBRID: 50% ML + 50% Vibe Match):")
        print(recommendations[['name', 'predicted_rating', 'vibe_match', 'cost', 'reason_1']].to_string(index=False))
        print("\nNote: predicted_rating = 0.5*ML_score + 0.5*vibe_match*5")
    else:
        print("No recommendations found (check time_fit values)")
    
    # Test different vibes to show variation
    print("\n" + "=" * 80)
    print("TESTING DIFFERENT VIBES (Same user: Durham, Couples, Morning)")
    print("=" * 80)
    
    for vibe in ['Chill', 'Cultural', 'Adventure', 'Mixed']:
        test_profile['vibe'] = vibe
        recs = ml_system.generate_recommendations(test_profile, top_k=3)
        if len(recs) > 0:
            print(f"\n{vibe} vibe → Top 3:")
            for idx, row in recs.iterrows():
                print(f"  {row['name']} (score: {row['predicted_rating']:.2f}, vibe_match: {row['vibe_match']:.1f})")
        else:
            print(f"\n{vibe} vibe → No recommendations")
    
    print("\n" + "=" * 80)
    print("✓ ML SYSTEM READY!")
    print("=" * 80)
    print("\nNext step: Run scenario_generator_ml.py to create all 384 scenarios")
    
    return ml_system


if __name__ == '__main__':
    ml_system = main()
