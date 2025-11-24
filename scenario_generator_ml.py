

import pandas as pd
import numpy as np
from wanderwell_ml_system import WanderWellMLSystem
import itertools


class MLScenarioGenerator:
    """Generate all scenarios using ML recommendations"""
    
    def __init__(self, ml_system, places_df):
        """Initialize with trained ML system"""
        self.ml_system = ml_system
        self.places_df = places_df
        
        # Define all scenario parameters
        self.cities = ['Durham', 'New York']
        self.companions = ['Solo', 'Couples', 'Friends', 'Family']
        self.vibes = ['Cultural', 'Mixed', 'Adventure', 'Chill']
        self.durations = [1, 2, 3, 4]  # days
        self.times = ['morning', 'lunch', 'evening']
        
        print("✓ ML Scenario Generator initialized")
    
    def generate_single_scenario(self, city, companions, vibe, duration, day, time_slot):
        """Generate recommendations for a single scenario"""
        user_profile = {
            'destination': city,
            'companions': companions,
            'vibe': vibe,
            'time_slot': time_slot
        }
        
        # Get recommendations from ML model
        recommendations = self.ml_system.generate_recommendations(user_profile, top_k=10)
        
        return recommendations
    
    def generate_daily_itinerary(self, city, companions, vibe, duration, excluded_places=None):
        """Generate complete itinerary for given duration"""
        if excluded_places is None:
            excluded_places = set()
        
        itinerary = []
        total_cost = 0
        
        for day in range(1, duration + 1):
            day_cost = 0
            day_places = []
            
            for time_slot in self.times:
                # Get recommendations for this time slot
                user_profile = {
                    'destination': city,
                    'companions': companions,
                    'vibe': vibe,
                    'time_slot': time_slot
                }
                
                recommendations = self.ml_system.generate_recommendations(user_profile, top_k=20)
                
                # Filter out already used places
                recommendations = recommendations[~recommendations['place_id'].isin(excluded_places)]
                
                if len(recommendations) > 0:
                    # Pick top recommendation
                    top_rec = recommendations.iloc[0]
                    
                    place_info = {
                        'day': day,
                        'time_slot': time_slot,
                        'place_id': top_rec['place_id'],
                        'name': top_rec['name'],
                        'category': top_rec['category'],
                        'predicted_rating': top_rec['predicted_rating'],
                        'cost': top_rec['cost'],
                        'duration': top_rec['duration'],
                        'reason_1': top_rec['reason_1'],
                        'reason_2': top_rec['reason_2'],
                    }
                    
                    itinerary.append(place_info)
                    excluded_places.add(top_rec['place_id'])
                    day_cost += top_rec['cost']
                    day_places.append(top_rec['name'])
                else:
                    # No recommendation available
                    itinerary.append({
                        'day': day,
                        'time_slot': time_slot,
                        'place_id': None,
                        'name': 'No recommendation',
                        'category': None,
                        'predicted_rating': 0,
                        'cost': 0,
                        'duration': 0,
                        'reason_1': '',
                        'reason_2': '',
                    })
            
            total_cost += day_cost
        
        return itinerary, total_cost
    
    def generate_all_scenarios(self):
        """Generate ALL possible scenarios"""
        print("\n" + "=" * 80)
        print("GENERATING ALL SCENARIOS")
        print("=" * 80)
        
        all_scenarios = []
        scenario_id = 1
        
        total_combinations = len(self.cities) * len(self.companions) * len(self.vibes) * len(self.durations)
        print(f"\nTotal scenario combinations: {total_combinations}")
        print("Progress:")
        
        progress_counter = 0
        
        for city in self.cities:
            for companions in self.companions:
                for vibe in self.vibes:
                    for duration in self.durations:
                        progress_counter += 1
                        
                        if progress_counter % 10 == 0:
                            print(f"  {progress_counter}/{total_combinations} scenarios generated...", end='\r')
                        
                        # Generate itinerary
                        itinerary, total_cost = self.generate_daily_itinerary(
                            city, companions, vibe, duration
                        )
                        
                        # Create scenario record
                        scenario = {
                            'Scenario_ID': f'SC{scenario_id:03d}',
                            'City': city,
                            'Companions': companions,
                            'Vibe': vibe,
                            'Duration_Days': duration,
                            'Total_Spots': len([x for x in itinerary if x['place_id'] is not None]),
                            'Total_Cost_USD': total_cost,
                            'Cost_Per_Day': total_cost / duration if duration > 0 else 0
                        }
                        
                        # Add places for each time slot
                        for item in itinerary:
                            day = item['day']
                            time_slot = item['time_slot']
                            col_prefix = f'Day{day}_{time_slot.capitalize()}'
                            
                            scenario[f'{col_prefix}_Place'] = item['name']
                            scenario[f'{col_prefix}_Category'] = item['category']
                            scenario[f'{col_prefix}_PredRating'] = item['predicted_rating']
                            scenario[f'{col_prefix}_Cost'] = item['cost']
                            scenario[f'{col_prefix}_Reason1'] = item['reason_1']
                            scenario[f'{col_prefix}_Reason2'] = item['reason_2']
                        
                        all_scenarios.append(scenario)
                        scenario_id += 1
        
        print(f"\n✓ Generated {len(all_scenarios)} scenarios")
        
        return pd.DataFrame(all_scenarios)
    
    def export_to_excel(self, scenarios_df, output_path):
        """Export scenarios to Excel with multiple sheets"""
        print("\n" + "=" * 80)
        print("EXPORTING TO EXCEL")
        print("=" * 80)
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Sheet 1: Summary
            summary_cols = ['Scenario_ID', 'City', 'Companions', 'Vibe', 'Duration_Days', 
                          'Total_Spots', 'Total_Cost_USD', 'Cost_Per_Day']
            scenarios_df[summary_cols].to_excel(writer, sheet_name='Scenarios_Summary', index=False)
            
            # Sheet 2: Full scenarios (all columns)
            scenarios_df.to_excel(writer, sheet_name='All_Scenarios_Detail', index=False)
            
            # Sheet 3: By City
            for city in self.cities:
                city_scenarios = scenarios_df[scenarios_df['City'] == city]
                sheet_name = f'{city.replace(" ", "_")}_Scenarios'[:31]  # Excel sheet name limit
                city_scenarios[summary_cols].to_excel(writer, sheet_name=sheet_name, index=False)
            
            # Sheet 4: Usage Guide
            guide_data = {
                'Step': [
                    '1. Find Scenario',
                    '2. Check Details',
                    '3. Get Places',
                    '4. Show to User'
                ],
                'Description': [
                    'User selects: NYC, Couples, Chill, 2 days',
                    'Look up in "All_Scenarios_Detail" sheet',
                    'Read Day1_Morning_Place, Day1_Lunch_Place, etc.',
                    'Display itinerary with total cost'
                ],
                'Example': [
                    'City=NYC, Companions=Couples, Vibe=Chill, Duration=2',
                    'Scenario_ID: SC042 (example)',
                    'Day1_Morning: Central Park, Day1_Lunch: Chelsea Market...',
                    'Total Cost: $180 ($90/day)'
                ]
            }
            pd.DataFrame(guide_data).to_excel(writer, sheet_name='Usage_Guide', index=False)
        
        print(f"\n✓ Excel file created: {output_path}")
        print(f"  - {len(scenarios_df)} scenarios")
        print(f"  - {len(writer.sheets)} sheets")


def main():
    """Main execution"""
    print("=" * 80)
    print("WANDERWELL ML SCENARIO GENERATOR")
    print("=" * 80)
    
    # Load data
    print("\n1. Loading data...")
    
    # Try multiple locations for data files
    import os
    
    # Look for files in current directory first
    if os.path.exists('Places.xlsx'):
        places_df = pd.read_excel('Places.xlsx')
        reviews_df = pd.read_excel('Reviews.xlsx')
    elif os.path.exists('/mnt/project/Places.xlsx'):
        places_df = pd.read_excel('/mnt/project/Places.xlsx')
        reviews_df = pd.read_excel('/mnt/project/Reviews.xlsx')
    else:
        raise FileNotFoundError(
            "Cannot find Places.xlsx and Reviews.xlsx!\n"
            "Please make sure these files are in the same folder as this script.\n"
            "Your folder should look like:\n"
            "  wanderwell/\n"
            "  ├── wanderwell_ml_system.py\n"
            "  ├── scenario_generator_ml.py\n"
            "  ├── run_ml_system.py\n"
            "  ├── Places.xlsx  ← Need this!\n"
            "  └── Reviews.xlsx ← Need this!"
        )
    
    # Try to load enhanced places
    try:
        if os.path.exists('Places_Enhanced.xlsx'):
            places_enhanced = pd.read_excel('Places_Enhanced.xlsx')
        elif os.path.exists('/mnt/user-data/outputs/Places_Enhanced.xlsx'):
            places_enhanced = pd.read_excel('/mnt/user-data/outputs/Places_Enhanced.xlsx')
        else:
            places_enhanced = None
            
        if places_enhanced is not None and all(col in places_enhanced.columns for col in ['is_mixed', 'is_cultural', 'is_chill', 'is_adventure']):
            places_df = places_enhanced
            print("   ✓ Using enhanced places with vibe tags")
        else:
            print("   ⚠️  Using original places data")
    except:
        print("   ⚠️  Using original places data")
    
    # Initialize and train ML system
    print("\n2. Training ML model...")
    ml_system = WanderWellMLSystem(places_df, reviews_df)
    X, y, groups, full_data = ml_system.prepare_training_data()
    results = ml_system.train_models(X, y, groups)
    
    print(f"\n   ✓ Best model: {ml_system.best_model_name}")
    print(f"   ✓ MAE: {results[ml_system.best_model_name]['mae']:.3f}")
    
    # Initialize scenario generator
    print("\n3. Initializing scenario generator...")
    generator = MLScenarioGenerator(ml_system, places_df)
    
    # Generate all scenarios
    print("\n4. Generating all scenarios...")
    scenarios_df = generator.generate_all_scenarios()
    
    # Export to Excel
    output_path = '/mnt/user-data/outputs/WanderWell_ML_All_Scenarios.xlsx'
    
    # Use local path if not in Claude environment
    import os
    if not os.path.exists('/mnt/user-data/outputs'):
        output_path = 'WanderWell_ML_All_Scenarios.xlsx'
    
    generator.export_to_excel(scenarios_df, output_path)
    
    # Show sample
    print("\n" + "=" * 80)
    print("SAMPLE SCENARIOS")
    print("=" * 80)
    
    sample = scenarios_df.head(3)
    display_cols = ['Scenario_ID', 'City', 'Companions', 'Vibe', 'Duration_Days', 'Total_Cost_USD']
    print("\n" + sample[display_cols].to_string(index=False))
    
    print("\n" + "=" * 80)
    print("✓ COMPLETE!")
    print("=" * 80)
    print(f"\nGenerated: {len(scenarios_df)} scenarios")
    print(f"Output: {output_path}")
    print("\nNext: Give Excel file to software engineer for API integration")


if __name__ == '__main__':
    main()
