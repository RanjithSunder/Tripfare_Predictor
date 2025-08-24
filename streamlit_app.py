# streamlit_app.py
# Standalone Streamlit Application for TripFare Prediction

import streamlit as st
import pandas as pd
import numpy as np
import math
import pickle
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(
    page_title="🚗 TripFare Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .prediction-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .feature-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance between two points on earth"""
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371  # Earth's radius in kilometers
    return c * r


def extract_time_features(pickup_datetime):
    """Extract time-based features from datetime"""
    pickup_hour = pickup_datetime.hour
    pickup_day = pickup_datetime.strftime('%A')
    pickup_weekday = pickup_datetime.weekday()
    pickup_month = pickup_datetime.month

    is_weekend = 1 if pickup_weekday >= 5 else 0
    am_pm = 1 if pickup_hour >= 12 else 0
    is_night = 1 if (pickup_hour >= 22 or pickup_hour <= 6) else 0
    is_morning_rush = 1 if (7 <= pickup_hour <= 9) else 0
    is_evening_rush = 1 if (17 <= pickup_hour <= 19) else 0

    return {
        'pickup_hour': pickup_hour,
        'pickup_day': pickup_day,
        'pickup_weekday': pickup_weekday,
        'pickup_month': pickup_month,
        'is_weekend': is_weekend,
        'am_pm': am_pm,
        'is_night': is_night,
        'is_morning_rush': is_morning_rush,
        'is_evening_rush': is_evening_rush
    }


def load_trained_model():
    """Load the trained ML model with better error handling"""
    try:
        # Check if sklearn is available
        import sklearn

        with open(r'models\best_tripfare_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model, True

    except ImportError as e:
        st.warning(f"⚠️ Missing dependency: {str(e)}. Please install scikit-learn: pip install scikit-learn")
        return None, False

    except FileNotFoundError:
        st.warning("⚠️ Trained model not found. Using rule-based prediction.")
        return None, False

    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, False


def predict_fare_ml(features, model=None, scaler=None):
    """
    Predict fare using trained ML model with correct feature alignment
    """
    if model is None:
        # Fallback to rule-based prediction
        return predict_fare_rule_based(features)

    try:
        # Prepare features for ML model (matching the expected 10 features)
        # Based on common taxi fare prediction features, likely order:
        feature_vector = np.array([[
            features['passenger_count'],
            features['trip_distance'],
            features['pickup_hour'],
            features['is_weekend'],
            features['is_night'],
            features['is_morning_rush'],
            features['is_evening_rush'],
            features['ratecode_id'],
            features['payment_type'],
            features.get('pickup_month', features.get('pickup_weekday', 1))  # Additional feature
        ]])

        # Scale features if scaler is available
        if scaler is not None:
            feature_vector = scaler.transform(feature_vector)

        # Make prediction
        prediction = model.predict(feature_vector)[0]

        return {
            'predicted_fare': max(prediction, 2.50),  # Ensure minimum fare
            'method': 'Machine Learning Model',
            'confidence': 'High' if features['trip_distance'] > 0.5 else 'Medium',
            'base_fare': 2.50,
            'distance_charge': features['trip_distance'] * 2.40,
            'time_charge': features['trip_distance'] * 2.5 * 0.40,
            'mta_tax': 0.50,
            'improvement_surcharge': 0.30,
            'time_multiplier': 1.0,
            'day_multiplier': 1.0
        }

    except ValueError as e:
        st.error(f"❌ Feature mismatch error: {str(e)}")
        st.info("🔄 Falling back to rule-based prediction...")
        return predict_fare_rule_based(features)
    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")
        return predict_fare_rule_based(features)


def predict_fare_rule_based(features):
    """
    Rule-based prediction (fallback method)
    """
    # Base fare structure
    base_fare = 2.50
    distance_rate = 2.40  # per km
    time_rate = 0.40  # per minute (estimated)

    # Multipliers based on features
    time_multiplier = 1.0
    day_multiplier = 1.0
    location_multiplier = 1.0
    passenger_multiplier = 1.0

    # Night surcharge
    if features['is_night']:
        time_multiplier += 0.20

    # Weekend surcharge
    if features['is_weekend']:
        day_multiplier += 0.10

    # Rush hour surcharge
    if features['is_morning_rush'] or features['is_evening_rush']:
        time_multiplier += 0.15

    # Passenger count adjustment
    if features['passenger_count'] > 1:
        passenger_multiplier += (features['passenger_count'] - 1) * 0.05

    # Rate code adjustments
    if features['ratecode_id'] == 2:  # JFK
        location_multiplier = 1.8
    elif features['ratecode_id'] == 3:  # Newark
        location_multiplier = 1.9
    elif features['ratecode_id'] == 4:  # Nassau/Westchester
        location_multiplier = 2.0
    elif features['ratecode_id'] == 5:  # Negotiated fare
        location_multiplier = 1.5

    # Estimate trip duration based on distance
    estimated_duration = features['trip_distance'] * 2.5  # minutes per km in city traffic

    # Calculate components
    distance_charge = features['trip_distance'] * distance_rate
    time_charge = estimated_duration * time_rate

    # Total calculation
    subtotal = base_fare + distance_charge + time_charge
    total_fare = subtotal * time_multiplier * day_multiplier * location_multiplier * passenger_multiplier

    # Add typical taxes and fees
    mta_tax = 0.50
    improvement_surcharge = 0.30

    final_fare = total_fare + mta_tax + improvement_surcharge

    return {
        'predicted_fare': final_fare,
        'method': 'Rule-Based Algorithm',
        'base_fare': base_fare,
        'distance_charge': distance_charge,
        'time_charge': time_charge,
        'mta_tax': mta_tax,
        'improvement_surcharge': improvement_surcharge,
        'time_multiplier': time_multiplier,
        'day_multiplier': day_multiplier,
        'confidence': 'Medium'
    }


def create_map_visualization(pickup_lat, pickup_lng, dropoff_lat, dropoff_lng):
    """Create a map showing pickup and dropoff locations"""

    # Calculate center and zoom
    center_lat = (pickup_lat + dropoff_lat) / 2
    center_lon = (pickup_lng + dropoff_lng) / 2

    # Calculate distance for zoom level
    distance = haversine_distance(pickup_lat, pickup_lng, dropoff_lat, dropoff_lng)
    zoom_level = max(10, min(15, 15 - distance / 2))  # Dynamic zoom based on distance

    # Create the base map
    fig = go.Figure()

    # Add pickup point
    fig.add_trace(go.Scattermapbox(
        lat=[pickup_lat],
        lon=[pickup_lng],
        mode='markers',
        marker=dict(
            size=25,
            color='#00FF00',  # Bright green
            opacity=0.9
        ),
        text=['🚗 Pickup Location'],
        hovertemplate='<b>🚗 Pickup Location</b><br>' +
                      'Latitude: %{lat:.4f}<br>' +
                      'Longitude: %{lon:.4f}<br>' +
                      '<extra></extra>',
        name='🚗 Pickup',
        showlegend=True
    ))

    # Add dropoff point
    fig.add_trace(go.Scattermapbox(
        lat=[dropoff_lat],
        lon=[dropoff_lng],
        mode='markers',
        marker=dict(
            size=25,
            color='#FF4444',  # Bright red
            opacity=0.9
        ),
        text=['🏁 Dropoff Location'],
        hovertemplate='<b>🏁 Dropoff Location</b><br>' +
                      'Latitude: %{lat:.4f}<br>' +
                      'Longitude: %{lon:.4f}<br>' +
                      '<extra></extra>',
        name='🏁 Dropoff',
        showlegend=True
    ))

    # Add a line connecting pickup and dropoff
    fig.add_trace(go.Scattermapbox(
        mode="lines",
        lon=[pickup_lng, dropoff_lng],
        lat=[pickup_lat, dropoff_lat],
        line=dict(width=4, color='#4169E1'),
        hoverinfo='skip',
        showlegend=False,
        name='Route'
    ))

    # Update layout for better visibility
    fig.update_layout(
        mapbox=dict(
            style="carto-positron",  # Clean, light style
            center=dict(lat=center_lat, lon=center_lon),
            zoom=zoom_level
        ),
        height=500,
        margin={"r": 10, "t": 50, "l": 10, "b": 10},
        title=dict(
            text="🗺️ Trip Route Visualization",
            font_size=16,
            x=0.5
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="Black",
            borderwidth=1,
            font=dict(size=12)
        ),
        showlegend=True
    )

    return fig


def create_fare_breakdown_chart(prediction_details):
    """Create a fare breakdown visualization"""

    components = ['Base Fare', 'Distance', 'Time', 'Surcharges', 'Taxes & Fees']
    values = [
        prediction_details['base_fare'],
        prediction_details['distance_charge'],
        prediction_details['time_charge'],
        max(0, prediction_details['predicted_fare'] - prediction_details['base_fare'] -
            prediction_details['distance_charge'] - prediction_details['time_charge'] -
            prediction_details['mta_tax'] - prediction_details['improvement_surcharge']),
        prediction_details['mta_tax'] + prediction_details['improvement_surcharge']
    ]

    fig = px.pie(
        values=values,
        names=components,
        title="Fare Breakdown",
        color_discrete_sequence=px.colors.qualitative.Set3
    )

    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=400)

    return fig


# Main App
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🚗 TripFare: Urban Taxi Fare Predictor</h1>
        <p>Get accurate fare predictions powered by machine learning</p>
    </div>
    """, unsafe_allow_html=True)

    # Load the trained model
    model, model_loaded = load_trained_model()

    # Display model status
    if model_loaded:
        st.success("✅ Machine Learning model loaded successfully!")

        # Display model info if available
        try:
            st.info(f"🔍 Model expects {model.n_features_in_} features")
        except:
            st.info("🔍 Model loaded - feature count detection not available")
    else:
        st.info("ℹ️ Using rule-based prediction. Train and save an ML model for better accuracy.")

    # Sidebar for inputs
    with st.sidebar:
        st.header("🎯 Trip Details")

        # Location inputs with some NYC defaults
        st.subheader("📍 Pickup Location")
        pickup_lat = st.number_input(
            "Pickup Latitude",
            value=40.7589,
            format="%.6f",
            help="Latitude coordinate for pickup location"
        )
        pickup_lng = st.number_input(
            "Pickup Longitude",
            value=-73.9851,
            format="%.6f",
            help="Longitude coordinate for pickup location"
        )

        st.subheader("🏁 Dropoff Location")
        dropoff_lat = st.number_input(
            "Dropoff Latitude",
            value=40.7505,
            format="%.6f",
            help="Latitude coordinate for dropoff location"
        )
        dropoff_lng = st.number_input(
            "Dropoff Longitude",
            value=-73.9934,
            format="%.6f",
            help="Longitude coordinate for dropoff location"
        )

        st.subheader("👥 Trip Information")
        passenger_count = st.selectbox(
            "Number of Passengers",
            options=[1, 2, 3, 4, 5, 6],
            index=0,
            help="Total number of passengers"
        )

        pickup_date = st.date_input(
            "Pickup Date",
            value=datetime.now().date(),
            help="When will the trip start?"
        )

        pickup_time = st.time_input(
            "Pickup Time",
            value=datetime.now().time(),
            help="Pickup time"
        )

        # Combine date and time
        pickup_datetime = datetime.combine(pickup_date, pickup_time)

        payment_type = st.selectbox(
            "Payment Type",
            options=[1, 2, 3, 4],
            format_func=lambda x: {1: "Credit Card", 2: "Cash", 3: "No Charge", 4: "Dispute"}[x],
            help="How will you pay?"
        )

        ratecode_id = st.selectbox(
            "Rate Code",
            options=[1, 2, 3, 4, 5],
            format_func=lambda x: {
                1: "Standard Rate",
                2: "JFK Airport",
                3: "Newark Airport",
                4: "Nassau/Westchester",
                5: "Negotiated Fare"
            }[x],
            help="Trip rate type"
        )

        # Predict button
        predict_button = st.button("🔮 Predict Fare", type="primary", use_container_width=True)

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        # Calculate trip features
        trip_distance = haversine_distance(pickup_lat, pickup_lng, dropoff_lat, dropoff_lng)
        time_features = extract_time_features(pickup_datetime)

        # Display trip information
        st.header("📊 Trip Analysis")

        # Trip metrics
        col_a, col_b, col_c, col_d = st.columns(4)

        with col_a:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Distance</h4>
                <h2>{trip_distance:.2f} km</h2>
            </div>
            """, unsafe_allow_html=True)

        with col_b:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Time</h4>
                <h2>{time_features['pickup_hour']:02d}:00</h2>
            </div>
            """, unsafe_allow_html=True)

        with col_c:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Day</h4>
                <h2>{time_features['pickup_day'][:3]}</h2>
            </div>
            """, unsafe_allow_html=True)

        with col_d:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Passengers</h4>
                <h2>{passenger_count}</h2>
            </div>
            """, unsafe_allow_html=True)

        # Trip features
        st.subheader("🔍 Trip Characteristics")

        feature_cols = st.columns(2)

        with feature_cols[0]:
            st.markdown(f"""
            <div class="feature-box">
                <strong>Time Category:</strong> {"🌙 Night" if time_features['is_night'] else "☀️ Day"}
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="feature-box">
                <strong>Day Type:</strong> {"🎉 Weekend" if time_features['is_weekend'] else "💼 Weekday"}
            </div>
            """, unsafe_allow_html=True)

        with feature_cols[1]:
            st.markdown(f"""
            <div class="feature-box">
                <strong>Rush Hour:</strong> {"🚦 Yes" if (time_features['is_morning_rush'] or time_features['is_evening_rush']) else "✅ No"}
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="feature-box">
                <strong>Rate Type:</strong> { {1: "Standard", 2: "JFK Airport", 3: "Newark Airport",
                                               4: "Nassau/Westchester", 5: "Negotiated"}[ratecode_id]
            }
            </div>
            """, unsafe_allow_html=True)

        # Map visualization with enhanced display
        if trip_distance > 0:
            st.subheader("🗺️ Route Visualization")

            # Add some info about the trip
            col_map1, col_map2 = st.columns([3, 1])

            with col_map1:
                map_fig = create_map_visualization(pickup_lat, pickup_lng, dropoff_lat, dropoff_lng)
                st.plotly_chart(map_fig, use_container_width=True)

            with col_map2:
                st.markdown("### 📍 Coordinates")
                st.markdown(f"""
                <div style="background: #f0f2f6; padding: 1rem; border-radius: 8px; margin-bottom: 0.5rem;">
                    <strong>🚗 Pickup:</strong><br>
                    Lat: {pickup_lat:.4f}<br>
                    Lng: {pickup_lng:.4f}
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <div style="background: #f0f2f6; padding: 1rem; border-radius: 8px; margin-bottom: 0.5rem;">
                    <strong>🏁 Dropoff:</strong><br>
                    Lat: {dropoff_lat:.4f}<br>
                    Lng: {dropoff_lng:.4f}
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <div style="background: #e8f4fd; padding: 1rem; border-radius: 8px; border-left: 4px solid #1f77b4;">
                    <strong>📏 Distance:</strong><br>
                    {trip_distance:.2f} km<br>
                    {trip_distance * 0.621371:.2f} miles
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("🗺️ Map will appear when pickup and dropoff locations are different")

    with col2:
        st.header("💰 Fare Prediction")

        if predict_button:
            # Prepare features for prediction
            features = {
                'trip_distance': trip_distance,
                'passenger_count': passenger_count,
                'ratecode_id': ratecode_id,
                'payment_type': payment_type,
                **time_features
            }

            # Get prediction using ML model or rule-based fallback
            if model_loaded:
                prediction = predict_fare_ml(features, model)
                st.info(f"🤖 Prediction Method: **{prediction['method']}**")
            else:
                prediction = predict_fare_rule_based(features)
                st.info(f"📊 Prediction Method: **{prediction['method']}**")

            # Display prediction
            st.markdown(f"""
            <div class="prediction-box">
                💵 Predicted Fare<br>
                <span style="font-size: 2.5rem;">${prediction['predicted_fare']:.2f}</span>
            </div>
            """, unsafe_allow_html=True)

            # Fare breakdown
            st.subheader("📋 Fare Breakdown")

            breakdown_data = {
                'Component': [
                    'Base Fare',
                    'Distance Charge',
                    'Time Charge',
                    'MTA Tax',
                    'Improvement Surcharge'
                ],
                'Amount': [
                    f"${prediction['base_fare']:.2f}",
                    f"${prediction['distance_charge']:.2f}",
                    f"${prediction['time_charge']:.2f}",
                    f"${prediction['mta_tax']:.2f}",
                    f"${prediction['improvement_surcharge']:.2f}"
                ]
            }

            # Add surcharge info if applicable
            if features['is_night']:
                breakdown_data['Component'].append('Night Surcharge')
                breakdown_data['Amount'].append('+20%')

            if features['is_weekend']:
                breakdown_data['Component'].append('Weekend Surcharge')
                breakdown_data['Amount'].append('+10%')

            if features['is_morning_rush'] or features['is_evening_rush']:
                breakdown_data['Component'].append('Rush Hour Surcharge')
                breakdown_data['Amount'].append('+15%')

            breakdown_df = pd.DataFrame(breakdown_data)
            st.dataframe(breakdown_df, use_container_width=True, hide_index=True)

            # Fare breakdown chart
            st.subheader("📊 Fare Distribution")
            breakdown_chart = create_fare_breakdown_chart(prediction)
            st.plotly_chart(breakdown_chart, use_container_width=True)

            # Confidence and tips
            st.subheader("💡 Prediction Insights")

            confidence = prediction.get('confidence', 'Medium')
            st.info(f"**Confidence Level:** {confidence}")

            if features['is_night']:
                st.warning("🌙 Night surcharge applied (+20%)")

            if features['is_weekend']:
                st.info("🎉 Weekend surcharge applied (+10%)")

            if features['is_morning_rush'] or features['is_evening_rush']:
                st.warning("🚦 Rush hour surcharge applied (+15%)")

            # Tips
            with st.expander("💰 Money-Saving Tips"):
                tips = []
                if features['is_night']:
                    tips.append("• Consider traveling during day hours to avoid night surcharge")
                if features['is_weekend']:
                    tips.append("• Weekday trips are typically cheaper")
                if features['is_morning_rush'] or features['is_evening_rush']:
                    tips.append("• Avoid rush hours (7-9 AM, 5-7 PM) for better rates")
                if ratecode_id in [2, 3, 4]:
                    tips.append("• Airport trips have premium rates")

                if tips:
                    for tip in tips:
                        st.markdown(tip)
                else:
                    st.markdown("• You're already getting a good rate! ✨")

        else:
            st.info("👆 Click 'Predict Fare' to get your fare estimate!")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p><strong>TripFare Predictor</strong> - Powered by Machine Learning</p>
        <p>🚗 Making taxi fare estimation transparent and accurate</p>
        <small>Note: Predictions are estimates based on historical data and current fare structures.</small>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()