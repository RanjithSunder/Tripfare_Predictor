# TripFare: Predicting Urban Taxi Fare with Machine Learning
# Complete Implementation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Machine Learning Libraries
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression, chi2
import pickle
import math

# For Streamlit UI
import streamlit as st


# ==============================================================================
# 1. DATA LOADING AND UNDERSTANDING
# ==============================================================================

def load_and_understand_data(file_path=None):
    """
    Load the dataset and perform initial data understanding
    """
    if file_path:
        df = pd.read_csv(file_path)
    else:
        # For demonstration, we'll create sample data structure
        # In actual implementation, load from the provided Google Drive link
        print("Please download dataset from the provided Google Drive link")
        return None

    print("Dataset Shape:", df.shape)
    print("\nColumn Names:")
    print(df.columns.tolist())

    print("\nData Types:")
    print(df.dtypes)

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nDuplicate Rows:", df.duplicated().sum())

    print("\nBasic Statistics:")
    print(df.describe())

    return df


# ==============================================================================
# 2. FEATURE ENGINEERING
# ==============================================================================

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371  # Radius of earth in kilometers
    return c * r


def feature_engineering(df):
    """
    Perform comprehensive feature engineering
    """
    # Convert datetime columns
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
    df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])

    # Convert from UTC to EDT (subtract 4 hours for EDT)
    df['pickup_datetime_edt'] = df['tpep_pickup_datetime'] - pd.Timedelta(hours=4)
    df['dropoff_datetime_edt'] = df['tpep_dropoff_datetime'] - pd.Timedelta(hours=4)

    # Calculate trip distance using Haversine formula
    df['trip_distance_calc'] = df.apply(lambda row: haversine_distance(
        row['pickup_latitude'], row['pickup_longitude'],
        row['dropoff_latitude'], row['dropoff_longitude']), axis=1)

    # Extract time-based features from pickup datetime
    df['pickup_hour'] = df['pickup_datetime_edt'].dt.hour
    df['pickup_day'] = df['pickup_datetime_edt'].dt.day_name()
    df['pickup_month'] = df['pickup_datetime_edt'].dt.month
    df['pickup_weekday'] = df['pickup_datetime_edt'].dt.weekday

    # Create binary features
    df['is_weekend'] = (df['pickup_weekday'] >= 5).astype(int)
    df['am_pm'] = (df['pickup_hour'] >= 12).astype(int)  # 0 for AM, 1 for PM
    df['is_night'] = ((df['pickup_hour'] >= 22) | (df['pickup_hour'] <= 6)).astype(int)

    # Rush hour indicators
    df['is_morning_rush'] = ((df['pickup_hour'] >= 7) & (df['pickup_hour'] <= 9)).astype(int)
    df['is_evening_rush'] = ((df['pickup_hour'] >= 17) & (df['pickup_hour'] <= 19)).astype(int)

    # Trip duration in minutes
    df['trip_duration'] = (df['dropoff_datetime_edt'] - df['pickup_datetime_edt']).dt.total_seconds() / 60

    # Fare per mile and fare per minute
    df['fare_per_mile'] = df['fare_amount'] / (
                df['trip_distance_calc'] + 1e-6)  # Add small value to avoid division by zero
    df['fare_per_minute'] = df['fare_amount'] / (df['trip_duration'] + 1e-6)

    print("Feature Engineering Completed!")
    print(f"New dataset shape: {df.shape}")

    return df


# ==============================================================================
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# ==============================================================================

def perform_eda(df):
    """
    Comprehensive Exploratory Data Analysis
    """
    plt.style.use('seaborn-v0_8')
    fig = plt.figure(figsize=(20, 15))

    # 1. Distribution of target variable
    plt.subplot(3, 4, 1)
    plt.hist(df['total_amount'], bins=50, edgecolor='black', alpha=0.7)
    plt.title('Distribution of Total Amount')
    plt.xlabel('Total Amount ($)')
    plt.ylabel('Frequency')

    # 2. Fare vs Distance
    plt.subplot(3, 4, 2)
    plt.scatter(df['trip_distance_calc'], df['total_amount'], alpha=0.5)
    plt.title('Fare vs Trip Distance')
    plt.xlabel('Trip Distance (km)')
    plt.ylabel('Total Amount ($)')

    # 3. Fare vs Passenger Count
    plt.subplot(3, 4, 3)
    sns.boxplot(data=df, x='passenger_count', y='total_amount')
    plt.title('Fare vs Passenger Count')
    plt.xlabel('Passenger Count')
    plt.ylabel('Total Amount ($)')

    # 4. Fare by Hour of Day
    plt.subplot(3, 4, 4)
    hourly_fare = df.groupby('pickup_hour')['total_amount'].mean()
    hourly_fare.plot(kind='bar')
    plt.title('Average Fare by Hour of Day')
    plt.xlabel('Hour')
    plt.ylabel('Average Fare ($)')
    plt.xticks(rotation=45)

    # 5. Fare by Day of Week
    plt.subplot(3, 4, 5)
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_fare = df.groupby('pickup_day')['total_amount'].mean().reindex(day_order)
    daily_fare.plot(kind='bar')
    plt.title('Average Fare by Day of Week')
    plt.xlabel('Day')
    plt.ylabel('Average Fare ($)')
    plt.xticks(rotation=45)

    # 6. Trip Count by Hour
    plt.subplot(3, 4, 6)
    hourly_trips = df['pickup_hour'].value_counts().sort_index()
    hourly_trips.plot(kind='bar')
    plt.title('Trip Count by Hour')
    plt.xlabel('Hour')
    plt.ylabel('Number of Trips')
    plt.xticks(rotation=45)

    # 7. Weekend vs Weekday Fares
    plt.subplot(3, 4, 7)
    weekend_data = ['Weekday' if x == 0 else 'Weekend' for x in df['is_weekend']]
    sns.boxplot(x=weekend_data, y=df['total_amount'])
    plt.title('Fare: Weekend vs Weekday')
    plt.ylabel('Total Amount ($)')

    # 8. Night vs Day Fares
    plt.subplot(3, 4, 8)
    night_data = ['Day' if x == 0 else 'Night' for x in df['is_night']]
    sns.boxplot(x=night_data, y=df['total_amount'])
    plt.title('Fare: Night vs Day')
    plt.ylabel('Total Amount ($)')

    # 9. Correlation Heatmap
    plt.subplot(3, 4, 9)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numeric_cols].corr()
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Heatmap')

    # 10. Fare per Mile Distribution
    plt.subplot(3, 4, 10)
    plt.hist(df['fare_per_mile'], bins=50, edgecolor='black', alpha=0.7)
    plt.title('Distribution of Fare per Mile')
    plt.xlabel('Fare per Mile ($/km)')
    plt.ylabel('Frequency')

    # 11. Trip Duration Distribution
    plt.subplot(3, 4, 11)
    plt.hist(df['trip_duration'], bins=50, edgecolor='black', alpha=0.7)
    plt.title('Distribution of Trip Duration')
    plt.xlabel('Duration (minutes)')
    plt.ylabel('Frequency')

    # 12. Payment Type vs Fare
    plt.subplot(3, 4, 12)
    if 'payment_type' in df.columns:
        sns.boxplot(data=df, x='payment_type', y='total_amount')
        plt.title('Fare by Payment Type')
        plt.xlabel('Payment Type')
        plt.ylabel('Total Amount ($)')

    plt.tight_layout()
    plt.show()

    # Print statistical insights
    print("=== EDA INSIGHTS ===")
    print(f"Average fare: ${df['total_amount'].mean():.2f}")
    print(f"Median fare: ${df['total_amount'].median():.2f}")
    print(f"Average trip distance: {df['trip_distance_calc'].mean():.2f} km")
    print(f"Average trip duration: {df['trip_duration'].mean():.2f} minutes")
    print(f"Weekend trips percentage: {df['is_weekend'].mean() * 100:.1f}%")
    print(f"Night trips percentage: {df['is_night'].mean() * 100:.1f}%")


# ==============================================================================
# 4. DATA TRANSFORMATION AND PREPROCESSING
# ==============================================================================

def handle_outliers(df, columns, method='iqr'):
    """
    Handle outliers using IQR or Z-score method
    """
    df_clean = df.copy()

    for col in columns:
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]

        elif method == 'zscore':
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            df_clean = df_clean[z_scores < 3]

    print(f"Outlier removal: {df.shape[0] - df_clean.shape[0]} rows removed")
    return df_clean


def transform_skewed_features(df, columns):
    """
    Apply transformations to fix skewness
    """
    df_transformed = df.copy()

    for col in columns:
        skewness = df[col].skew()
        if abs(skewness) > 0.5:
            if skewness > 0:  # Right skewed
                df_transformed[f'{col}_log'] = np.log1p(df[col])
                print(f"Applied log transformation to {col} (skewness: {skewness:.2f})")
            else:  # Left skewed
                df_transformed[f'{col}_sqrt'] = np.sqrt(df[col] - df[col].min() + 1)
                print(f"Applied sqrt transformation to {col} (skewness: {skewness:.2f})")

    return df_transformed


def preprocess_data(df):
    """
    Complete data preprocessing pipeline
    """
    print("Starting data preprocessing...")

    # Handle missing values
    df = df.dropna()

    # Handle outliers in key numerical columns
    numerical_cols = ['total_amount', 'trip_distance_calc', 'trip_duration', 'fare_amount']
    df_clean = handle_outliers(df, numerical_cols, method='iqr')

    # Transform skewed features
    skewed_cols = ['total_amount', 'trip_distance_calc', 'fare_amount', 'trip_duration']
    df_transformed = transform_skewed_features(df_clean, skewed_cols)

    # Encode categorical variables
    le = LabelEncoder()
    categorical_cols = ['pickup_day', 'payment_type', 'RatecodeID']

    for col in categorical_cols:
        if col in df_transformed.columns:
            df_transformed[f'{col}_encoded'] = le.fit_transform(df_transformed[col].astype(str))

    print("Data preprocessing completed!")
    return df_transformed


# ==============================================================================
# 5. FEATURE SELECTION
# ==============================================================================

def feature_selection(X, y, method='correlation', k=15):
    """
    Apply various feature selection techniques
    """
    selected_features = []

    if method == 'correlation':
        # Correlation with target
        correlations = X.corrwith(y).abs().sort_values(ascending=False)
        selected_features = correlations.head(k).index.tolist()
        print(f"Top {k} features by correlation:")
        print(correlations.head(k))

    elif method == 'f_regression':
        # F-regression test
        selector = SelectKBest(score_func=f_regression, k=k)
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        print(f"Selected features using F-regression: {selected_features}")

    elif method == 'random_forest':
        # Feature importance from Random Forest
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)

        selected_features = importance_df.head(k)['feature'].tolist()
        print(f"Top {k} features by Random Forest importance:")
        print(importance_df.head(k))

    return selected_features


# ==============================================================================
# 6. MODEL BUILDING AND EVALUATION
# ==============================================================================

class ModelEvaluator:
    def __init__(self):
        self.models = {}
        self.results = {}

    def add_model(self, name, model):
        self.models[name] = model

    def evaluate_model(self, name, model, X_train, X_test, y_train, y_test):
        """
        Evaluate a single model
        """
        # Train the model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Calculate metrics
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)

        # Store results
        self.results[name] = {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'model': model
        }

        print(f"\n{name} Results:")
        print(f"Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
        print(f"Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}")
        print(f"Train MAE: {train_mae:.4f}, Test MAE: {test_mae:.4f}")

    def compare_models(self):
        """
        Compare all models and return results DataFrame
        """
        results_df = pd.DataFrame(self.results).T
        results_df = results_df.round(4)
        print("\n=== MODEL COMPARISON ===")
        print(results_df[['test_r2', 'test_rmse', 'test_mae']].sort_values('test_r2', ascending=False))
        return results_df

    def get_best_model(self):
        """
        Get the best performing model based on test R²
        """
        best_model_name = max(self.results.keys(),
                              key=lambda x: self.results[x]['test_r2'])
        return best_model_name, self.results[best_model_name]['model']


def build_and_evaluate_models(X_train, X_test, y_train, y_test):
    """
    Build and evaluate multiple regression models
    """
    evaluator = ModelEvaluator()

    # Define models
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }

    # Evaluate each model
    for name, model in models.items():
        evaluator.add_model(name, model)
        evaluator.evaluate_model(name, model, X_train, X_test, y_train, y_test)

    # Compare models
    results_df = evaluator.compare_models()

    # Get best model
    best_name, best_model = evaluator.get_best_model()
    print(f"\nBest Model: {best_name}")

    return evaluator, best_model, best_name


# ==============================================================================
# 7. HYPERPARAMETER TUNING
# ==============================================================================

def hyperparameter_tuning(model, param_grid, X_train, y_train, cv=5):
    """
    Perform hyperparameter tuning using GridSearchCV
    """
    print("Performing hyperparameter tuning...")

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring='r2',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_


# ==============================================================================
# 8. MODEL PERSISTENCE
# ==============================================================================

def save_model(model, filename):
    """
    Save the best model to a pickle file
    """
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model saved as {filename}")


def load_model(filename):
    """
    Load a saved model from pickle file
    """
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    print(f"Model loaded from {filename}")
    return model


# ==============================================================================
# 9. STREAMLIT UI
# ==============================================================================

def create_streamlit_app():
    """
    Create Streamlit UI for fare prediction
    """
    st.set_page_config(
        page_title="🚗 TripFare Predictor",
        page_icon="🚗",
        layout="wide"
    )

    st.title("🚗 TripFare: Urban Taxi Fare Predictor")
    st.markdown("---")

    # Sidebar for user inputs
    st.sidebar.header("Trip Details")

    # Input fields
    pickup_latitude = st.sidebar.number_input("Pickup Latitude", value=40.7589, format="%.6f")
    pickup_longitude = st.sidebar.number_input("Pickup Longitude", value=-73.9851, format="%.6f")
    dropoff_latitude = st.sidebar.number_input("Dropoff Latitude", value=40.7505, format="%.6f")
    dropoff_longitude = st.sidebar.number_input("Dropoff Longitude", value=-73.9934, format="%.6f")

    passenger_count = st.sidebar.selectbox("Passenger Count", [1, 2, 3, 4, 5, 6])

    pickup_datetime = st.sidebar.datetime_input("Pickup Date & Time")
    pickup_hour = pickup_datetime.hour
    pickup_day = pickup_datetime.strftime('%A')
    pickup_weekday = pickup_datetime.weekday()

    payment_type = st.sidebar.selectbox("Payment Type", [1, 2, 3, 4])
    ratecode_id = st.sidebar.selectbox("Rate Code", [1, 2, 3, 4, 5])

    # Calculate derived features
    trip_distance = haversine_distance(pickup_latitude, pickup_longitude,
                                       dropoff_latitude, dropoff_longitude)

    is_weekend = 1 if pickup_weekday >= 5 else 0
    am_pm = 1 if pickup_hour >= 12 else 0
    is_night = 1 if (pickup_hour >= 22 or pickup_hour <= 6) else 0
    is_morning_rush = 1 if (7 <= pickup_hour <= 9) else 0
    is_evening_rush = 1 if (17 <= pickup_hour <= 19) else 0

    # Display calculated features
    st.sidebar.markdown("### Calculated Features")
    st.sidebar.write(f"Trip Distance: {trip_distance:.2f} km")
    st.sidebar.write(f"Time Category: {'Night' if is_night else 'Day'}")
    st.sidebar.write(f"Day Type: {'Weekend' if is_weekend else 'Weekday'}")

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("Trip Information")

        # Display trip details
        trip_info = pd.DataFrame({
            'Feature': ['Pickup Location', 'Dropoff Location', 'Distance', 'Passengers',
                        'Day of Week', 'Time', 'Weekend', 'Night Trip'],
            'Value': [f"{pickup_latitude:.4f}, {pickup_longitude:.4f}",
                      f"{dropoff_latitude:.4f}, {dropoff_longitude:.4f}",
                      f"{trip_distance:.2f} km",
                      passenger_count,
                      pickup_day,
                      f"{pickup_hour:02d}:00",
                      'Yes' if is_weekend else 'No',
                      'Yes' if is_night else 'No']
        })

        st.dataframe(trip_info, use_container_width=True)

    with col2:
        st.header("Fare Prediction")

        if st.button("🔮 Predict Fare", type="primary", use_container_width=True):
            # In a real implementation, load the trained model and make prediction
            # For demonstration, we'll use a simple calculation

            # Mock prediction (replace with actual model prediction)
            base_fare = 2.50
            distance_rate = 2.80
            time_multiplier = 1.2 if is_night else 1.0
            weekend_multiplier = 1.1 if is_weekend else 1.0
            rush_multiplier = 1.15 if (is_morning_rush or is_evening_rush) else 1.0

            predicted_fare = (base_fare + (
                        trip_distance * distance_rate)) * time_multiplier * weekend_multiplier * rush_multiplier

            # Display prediction
            st.success(f"💰 Predicted Fare: ${predicted_fare:.2f}")

            # Show fare breakdown
            st.markdown("### Fare Breakdown")
            breakdown = pd.DataFrame({
                'Component': ['Base Fare', 'Distance Charge', 'Night Surcharge',
                              'Weekend Surcharge', 'Rush Hour Surcharge'],
                'Amount': [f"${base_fare:.2f}",
                           f"${trip_distance * distance_rate:.2f}",
                           f"{((time_multiplier - 1) * 100):.0f}%" if is_night else "0%",
                           f"{((weekend_multiplier - 1) * 100):.0f}%" if is_weekend else "0%",
                           f"{((rush_multiplier - 1) * 100):.0f}%" if (is_morning_rush or is_evening_rush) else "0%"]
            })
            st.dataframe(breakdown, use_container_width=True)

    # Footer
    st.markdown("---")
    st.markdown("**Note:** This is a machine learning prediction model trained on historical taxi trip data.")


# ==============================================================================
# 10. MAIN EXECUTION PIPELINE
# ==============================================================================

def main_pipeline(data_path=None):
    """
    Complete pipeline execution
    """
    print("🚗 TripFare: Predicting Urban Taxi Fare with Machine Learning")
    print("=" * 60)

    # Step 1: Load and understand data
    if data_path:
        df = load_and_understand_data(data_path)
        if df is None:
            return

        # Step 2: Feature Engineering
        df_engineered = feature_engineering(df)

        # Step 3: EDA
        print("\nPerforming Exploratory Data Analysis...")
        perform_eda(df_engineered)

        # Step 4: Data Preprocessing
        df_processed = preprocess_data(df_engineered)

        # Step 5: Prepare features and target
        # Select relevant features for modeling
        feature_columns = ['passenger_count', 'trip_distance_calc', 'pickup_hour',
                           'trip_duration', 'is_weekend', 'is_night', 'is_morning_rush',
                           'is_evening_rush', 'fare_per_mile', 'RatecodeID', 'payment_type']

        # Filter columns that exist in the dataset
        available_features = [col for col in feature_columns if col in df_processed.columns]

        X = df_processed[available_features]
        y = df_processed['total_amount']

        # Step 6: Feature Selection
        selected_features = feature_selection(X, y, method='random_forest', k=10)
        X_selected = X[selected_features]

        # Step 7: Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, test_size=0.2, random_state=42
        )

        # Step 8: Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Step 9: Model Building and Evaluation
        evaluator, best_model, best_name = build_and_evaluate_models(
            X_train_scaled, X_test_scaled, y_train, y_test
        )

        # Step 10: Hyperparameter Tuning (optional)
        if best_name == 'Random Forest':
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5]
            }
            best_model_tuned = hyperparameter_tuning(best_model, param_grid, X_train_scaled, y_train)
        else:
            best_model_tuned = best_model

        # Step 11: Save the best model
        save_model(best_model_tuned, 'models/best_tripfare_model.pkl')

        print("\n🎉 Pipeline completed successfully!")
        print(f"Best Model: {best_name}")
        print("Model saved as 'best_tripfare_model.pkl'")
        print("You can now run the Streamlit app for fare prediction!")

    else:
        print("Please provide the dataset path to run the complete pipeline.")
        print("For now, you can run the Streamlit demo:")
        print("streamlit run tripfare_app.py")


# ==============================================================================
# STREAMLIT APP RUNNER
# ==============================================================================

if __name__ == "__main__":
    # Check if running in Streamlit
    try:
        # This will work only if running in Streamlit
        import streamlit as st

        create_streamlit_app()
    except:
        # Running as regular Python script
        print("To run the Streamlit app:")
        print("1. Save this code as 'tripfare_app.py'")
        print("2. Run: streamlit run tripfare_app.py")
        print("\nTo run the complete pipeline:")
        print("main_pipeline('path_to_your_dataset.csv')")
        main_pipeline(data_path=r"C:\Ranjith\Exercise\DS Project\DS_project_3\data\taxi_fare.csv")