import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import random

# Location coordinates (accurate Dubai neighborhood centroids)
location_coords = {
    'Downtown Dubai': [25.1972, 55.2744],
    'Dubai Marina': [25.0775, 55.1319],
    'Jumeirah Village Circle': [25.0665, 55.2170],
    'Palm Jumeirah': [25.1124, 55.1390],
    'Business Bay': [25.1860, 55.2683],
    'Deira': [25.2697, 55.3090],
    'Al Barsha': [25.0978, 55.2044]
}

# Coordinate generator for each location to avoid water markers
np.random.seed(42)

def generate_coordinates(location):
    base_lat, base_lon = location_coords[location]
    if location == 'Palm Jumeirah':
        return base_lat + np.random.uniform(-0.0018, 0.0018), base_lon + np.random.uniform(-0.0018, 0.0018)
    return base_lat + np.random.uniform(-0.005, 0.005), base_lon + np.random.uniform(-0.005, 0.005)

locations = np.random.choice(list(location_coords.keys()), 100)
coords = [generate_coordinates(loc) for loc in locations]

property_images = {
    'Apartment': [
        'https://images.unsplash.com/photo-1494526585095-c41746248156?auto=format&fit=crop&w=800&q=80',
        'https://images.unsplash.com/photo-1560448204-e02f11c3d0e2?auto=format&fit=crop&w=800&q=80',
        'https://images.unsplash.com/photo-1505693416388-ac5ce068fe85?auto=format&fit=crop&w=800&q=80'
    ],
    'Villa': [
        'https://images.unsplash.com/photo-1505693416388-ac5ce068fe85?auto=format&fit=crop&w=800&q=80',
        'https://images.unsplash.com/photo-1523217582562-09d0def993a6?auto=format&fit=crop&w=800&q=80',
        'https://images.unsplash.com/photo-1449941966412-47a169e7a28e?auto=format&fit=crop&w=800&q=80'
    ],
    'Townhouse': [
        'https://images.unsplash.com/photo-1453384912939-7f3abb0c632f?auto=format&fit=crop&w=800&q=80',
        'https://images.unsplash.com/photo-1599423300746-b62533397364?auto=format&fit=crop&w=800&q=80',
        'https://images.unsplash.com/photo-1568605114967-8130f3a36994?auto=format&fit=crop&w=800&q=80'
    ],
    'Office': [
        'https://images.unsplash.com/photo-1494526585095-c41746248156?auto=format&fit=crop&w=800&q=80',
        'https://images.unsplash.com/photo-1532375810709-f994e8758dc5?auto=format&fit=crop&w=800&q=80',
        'https://images.unsplash.com/photo-1454165804606-c3d57bc86b40?auto=format&fit=crop&w=800&q=80'
    ],
    'Land': [
        'https://images.unsplash.com/photo-1500530855697-b586d89ba3ee?auto=format&fit=crop&w=800&q=80',
        'https://images.unsplash.com/photo-1501785888041-af3ef285b470?auto=format&fit=crop&w=800&q=80',
        'https://images.unsplash.com/photo-1472220625704-91e1462799b2?auto=format&fit=crop&w=800&q=80'
    ]
}

def get_image_url(property_type, index):
    urls = property_images.get(property_type, property_images['Apartment'])
    return urls[index % len(urls)]

property_types = np.random.choice(['Apartment', 'Villa', 'Townhouse', 'Office', 'Land'], 100)

# Sample data for Dubai real estate properties
np.random.seed(42)
data = {
    'id': list(range(1, 101)),
    'price': np.random.randint(300000, 10000000, 100),
    'land_size': np.random.randint(500, 10000, 100),  # sq ft
    'location': locations,
    'property_type': property_types,
    'bedrooms': np.random.randint(0, 6, 100),
    'bathrooms': np.random.randint(1, 5, 100),
    'furnished': np.random.choice([True, False], 100),
    'parking': np.random.choice([True, False], 100),
    'pool': np.random.choice([True, False], 100),
    'gym': np.random.choice([True, False], 100),
    'metro_distance': np.random.randint(0, 5000, 100),  # meters
    'building_age': np.random.randint(0, 50, 100),
    'floor': np.random.randint(1, 50, 100),
    'view_type': np.random.choice(['City', 'Sea', 'Garden', 'Mountain'], 100),
    'image_url': [get_image_url(property_types[i], i) for i in range(100)],
    'lat': [coord[0] for coord in coords],
    'lon': [coord[1] for coord in coords]
}

df = pd.DataFrame(data)

# Train ML model for price prediction
features = ['land_size', 'bedrooms', 'bathrooms', 'furnished', 'parking', 'pool', 'gym', 'metro_distance', 'building_age', 'floor']
X = df[features]
y = df['price']
X = pd.get_dummies(X, drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Streamlit app
st.set_page_config(page_title="Dubai Smart Real Estate Intelligence Platform", layout="wide")

st.markdown(
    """
    <style>
        .stApp, .css-18e3th9, .main, .block-container, .css-1d391kg {
            background: linear-gradient(180deg, #031628 0%, #06335f 100%) !important;
            color: #f8fafc !important;
        }
        body {
            background-color: #031628 !important;
            color: #f8fafc !important;
        }
        .css-1d391kg {
            background-color: transparent !important;
        }
        .css-18e3th9 {
            background-color: transparent !important;
        }
        .css-1m76ban, .css-1lcbmhc, .css-1d391kg, .css-10trblm {
            background: rgba(8, 30, 64, 0.92) !important;
            color: #f8fafc !important;
        }
        .stSidebar {
            background: linear-gradient(180deg, #071d3a 0%, #0f3f6d 100%) !important;
        }
        .css-1d391kg .stSelectbox, .css-1d391kg .stTextInput, .css-1d391kg .stSlider, .css-1d391kg .stRadio {
            color: #f8fafc !important;
        }
        label, .streamlit-expanderHeader, .css-1x8cf1s, .css-1f0qwjs {
            color: #f8fafc !important;
        }
        .stButton>button {
            background-color: #1f5aac !important;
            color: #ffffff !important;
            border-radius: 12px !important;
            padding: 9px 16px !important;
            border: none !important;
        }
        .st-bz {
            background-color: rgba(255, 255, 255, 0.05) !important;
        }
        .hero-banner {
            background: linear-gradient(135deg, #072449, #163c82);
            color: white;
            padding: 28px 32px;
            border-radius: 24px;
            box-shadow: 0 25px 70px rgba(9, 30, 66, 0.25);
            margin-bottom: 24px;
        }
        .property-card {
            background: rgba(255, 255, 255, 0.08);
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.14);
            padding: 18px;
            box-shadow: 0 18px 40px rgba(0, 0, 0, 0.18);
            margin-bottom: 18px;
        }
        .property-card h4 {
            margin: 0 0 8px 0;
            color: #ffffff;
        }
        .section-heading {
            color: #e2efff;
            margin-bottom: 12px;
        }
        .metric-card {
            background: rgba(255, 255, 255, 0.08);
            border-radius: 20px;
            padding: 18px;
            box-shadow: 0 18px 36px rgba(0, 0, 0, 0.12);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        .streamlit-expanderHeader {
            font-weight: 700;
            color: #f8fafc !important;
        }
        .css-10trblm, .css-1jzxamu {
            color: #f8fafc !important;
        }
        .stTextInput>div>div>input, .stSelectbox>div>div>div, .stTextArea>div>div>textarea {
            background: rgba(255,255,255,0.08) !important;
            color: #f8fafc !important;
            border: 1px solid rgba(255,255,255,0.16) !important;
        }
        .css-1ybwtax, .css-1tk77yz {
            color: #f8fafc !important;
        }
        .st-bf {
            background-color: rgba(255, 255, 255, 0.04) !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar
st.sidebar.title("🏠 Dubai Real Estate")
page = st.sidebar.radio("Navigate", ["Property Finder", "Price Prediction", "ROI Calculator", "Smart Recommendations", "Map View", "Interactive Dashboard", "Property Management", "AI Chatbot", "Analytics Dashboard"])

if page == "Property Finder":
    st.markdown(
        "<div class='hero-banner'><h1>Dubai Real Estate Property Finder</h1><p>Find your dream property in Dubai based on your budget, size, and location preferences.</p></div>",
        unsafe_allow_html=True,
    )

    # User inputs
    col1, col2 = st.columns(2)

    with col1:
        budget = st.number_input("Enter your maximum budget (AED)", min_value=0, value=2000000, step=50000)

    with col2:
        min_land_size = st.slider("Minimum land size (sq ft)", 0, 10000, 0)

    # Filters
    st.subheader("Filters")
    col3, col4, col5 = st.columns(3)

    with col3:
        sort_by = st.selectbox("Sort by", ["price", "land_size", "bedrooms"], index=0)

    with col4:
        sort_order = st.radio("Sort order", ["Ascending", "Descending"], index=0)

    with col5:
        property_type_filter = st.multiselect("Property Type", df['property_type'].unique(), default=df['property_type'].unique())

    # Additional filters
    col6, col7, col8 = st.columns(3)
    with col6:
        min_bedrooms = st.slider("Min Bedrooms", 0, 5, 0)
    with col7:
        furnished_filter = st.multiselect("Furnished", [True, False], default=[True, False])
    with col8:
        location_filter = st.multiselect("Location", df['location'].unique(), default=df['location'].unique())

    # Filter data
    filtered_df = df[
        (df['price'] <= budget) &
        (df['land_size'] >= min_land_size) &
        (df['bedrooms'] >= min_bedrooms) &
        (df['property_type'].isin(property_type_filter)) &
        (df['furnished'].isin(furnished_filter)) &
        (df['location'].isin(location_filter))
    ]

    # Sort data
    ascending = sort_order == "Ascending"
    filtered_df = filtered_df.sort_values(by=sort_by, ascending=ascending)

    st.subheader(f"Found {len(filtered_df)} properties matching your criteria")

    if len(filtered_df) > 0:
        # Display properties
        for _, row in filtered_df.iterrows():
            with st.container():
                col_img, col_details = st.columns([1, 2])
                with col_img:
                    st.image(row['image_url'], width=200)
                with col_details:
                    st.markdown(f"**Price:** {row['price']:,} AED")
                    st.markdown(f"**Land Size:** {row['land_size']} sq ft")
                    st.markdown(f"**Location:** {row['location']}")
                    st.markdown(f"**Type:** {row['property_type']}")
                    st.markdown(f"**Bedrooms:** {row['bedrooms']}, **Bathrooms:** {row['bathrooms']}")
                    st.markdown(f"**Furnished:** {'Yes' if row['furnished'] else 'No'}, **Parking:** {'Yes' if row['parking'] else 'No'}")
                st.divider()
    else:
        st.write("No properties found matching your criteria. Try adjusting your budget or filters.")

elif page == "Price Prediction":
    st.title("🤖 Property Price Prediction")
    st.markdown("Predict property prices using machine learning!")

    col1, col2 = st.columns(2)
    with col1:
        pred_location = st.selectbox("Location", list(location_coords.keys()))
        pred_size = st.number_input("Land Size (sq ft)", min_value=500, value=1000)
        pred_bedrooms = st.slider("Bedrooms", 0, 5, 2)
        pred_bathrooms = st.slider("Bathrooms", 1, 4, 2)
        pred_furnished = st.checkbox("Furnished")
    with col2:
        pred_parking = st.checkbox("Parking")
        pred_pool = st.checkbox("Pool")
        pred_gym = st.checkbox("Gym")
        pred_metro_dist = st.slider("Metro Distance (m)", 0, 5000, 1000)
        pred_age = st.slider("Building Age (years)", 0, 50, 5)
        pred_floor = st.slider("Floor Number", 1, 50, 10)

    if st.button("Predict Price"):
        # Prepare input for model
        input_data = pd.DataFrame({
            'land_size': [pred_size],
            'bedrooms': [pred_bedrooms],
            'bathrooms': [pred_bathrooms],
            'furnished': [pred_furnished],
            'parking': [pred_parking],
            'pool': [pred_pool],
            'gym': [pred_gym],
            'metro_distance': [pred_metro_dist],
            'building_age': [pred_age],
            'floor': [pred_floor]
        })
        input_data = pd.get_dummies(input_data, drop_first=True)
        # Ensure same columns as training
        for col in X_train.columns:
            if col not in input_data.columns:
                input_data[col] = 0
        input_data = input_data[X_train.columns]

        prediction = model.predict(input_data)[0]
        st.success(f"Estimated Price: {prediction:,.0f} AED")
        st.info(f"Model MAE: {mean_absolute_error(y_test, model.predict(X_test)):,.0f} AED")

elif page == "ROI Calculator":
    st.title("📈 ROI Calculator / Investment Analyzer")
    st.markdown("Calculate potential returns on your investment!")

    property_price = st.number_input("Property Price (AED)", min_value=0, value=1000000)
    annual_rent = st.number_input("Annual Rental Income (AED)", min_value=0, value=80000)
    appreciation_rate = st.slider("Annual Appreciation Rate (%)", 0.0, 10.0, 3.0)
    years = st.slider("Investment Period (years)", 1, 10, 5)

    if st.button("Calculate ROI"):
        rental_yield = (annual_rent / property_price) * 100
        future_value = property_price * (1 + appreciation_rate/100) ** years
        total_return = (annual_rent * years) + (future_value - property_price)
        roi = (total_return / property_price) * 100

        st.metric("Rental Yield", f"{rental_yield:.1f}%")
        st.metric("Future Value", f"{future_value:,.0f} AED")
        st.metric(f"ROI after {years} years", f"{roi:.1f}%")

elif page == "Smart Recommendations":
    st.title("🎯 Smart Property Recommendations")
    st.markdown("Get personalized property recommendations!")

    user_budget = st.number_input("Your Budget (AED)", min_value=0, value=1500000)
    preferred_location = st.selectbox("Preferred Location", list(location_coords.keys()))
    family_size = st.slider("Family Size", 1, 10, 4)

    if st.button("Get Recommendations"):
        # Simple recommendation logic
        recommended = df[
            (df['price'] <= user_budget * 1.2) &  # Within 20% of budget
            (df['location'] == preferred_location) &
            (df['bedrooms'] >= max(1, family_size // 2))  # At least half the bedrooms
        ].head(5)

        if len(recommended) > 0:
            st.subheader("Recommended Properties:")
            for _, row in recommended.iterrows():
                st.markdown(f"- {row['property_type']} in {row['location']}: {row['price']:,} AED, {row['bedrooms']} bedrooms")
        else:
            st.write("No recommendations found. Try adjusting your preferences.")

elif page == "Map View":
    st.title("🗺️ Property Map View")
    st.markdown("Explore properties on an interactive Dubai map with improved location accuracy!")

    query = st.text_input("Search properties by location or type")
    if query:
        search_results = df[
            df['location'].str.contains(query, case=False, na=False) |
            df['property_type'].str.contains(query, case=False, na=False)
        ].reset_index(drop=True)
    else:
        search_results = df.reset_index(drop=True)

    st.markdown(f"**Properties shown on map:** {len(search_results)}")

    if len(search_results) > 0:
        select_options = search_results.apply(
            lambda row: f"{row['id']} | {row['property_type']} in {row['location']} | AED {row['price']:,}",
            axis=1
        ).tolist()
        selected = st.selectbox("Highlight a property", ["None"] + select_options)
        selected_id = None
        if selected != "None":
            selected_id = int(selected.split(" | ")[0])
    else:
        selected_id = None

    center = [25.2048, 55.2708]
    zoom_level = 11
    if selected_id is not None:
        selected_row = search_results[search_results['id'] == selected_id].iloc[0]
        center = [selected_row['lat'], selected_row['lon']]
        zoom_level = 14

    m = folium.Map(location=center, zoom_start=zoom_level, tiles='OpenStreetMap', width='100%', height='650')
    marker_cluster = MarkerCluster().add_to(m)

    for _, row in search_results.iterrows():
        popup_html = f"""
            <div style='font-size:14px; line-height:1.4'>
                <strong>{row['property_type']}</strong><br>
                {row['location']}<br>
                Price: {row['price']:,} AED<br>
                Beds: {row['bedrooms']}, Baths: {row['bathrooms']}<br>
                View: {row['view_type']}<br>
                <img src='{row['image_url']}' width='240' />
            </div>
        """
        is_selected = selected_id is not None and row['id'] == selected_id
        folium.Marker(
            location=[row['lat'], row['lon']],
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"{row['location']} - {row['bedrooms']} bed",
            icon=folium.Icon(color='red' if is_selected else 'blue', icon='home', prefix='fa')
        ).add_to(marker_cluster)

    if selected_id is None and len(search_results) > 0:
        lat_min, lat_max = search_results['lat'].min(), search_results['lat'].max()
        lon_min, lon_max = search_results['lon'].min(), search_results['lon'].max()
        m.fit_bounds([[lat_min, lon_min], [lat_max, lon_max]])

    st_folium(m, width=1200, height=650)

    if selected_id is not None:
        selected_row = search_results[search_results['id'] == selected_id].iloc[0]
        st.markdown("### Selected Property Details")
        st.markdown(f"**Location:** {selected_row['location']}")
        st.markdown(f"**Type:** {selected_row['property_type']}")
        st.markdown(f"**Price:** {selected_row['price']:,} AED")
        st.markdown(f"**Land Size:** {selected_row['land_size']} sq ft")
        st.markdown(f"**Bedrooms:** {selected_row['bedrooms']}, Bathrooms: {selected_row['bathrooms']}")
        st.image(selected_row['image_url'], width=400)
    else:
        st.write("Search or select a property to highlight it on the map.")

elif page == "Interactive Dashboard":
    st.markdown(
        "<div class='hero-banner'><h1>📊 Interactive Dashboard</h1><p>Explore Dubai real estate market trends and insights through interactive visualizations.</p></div>",
        unsafe_allow_html=True,
    )
    
    st.markdown("### Market Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"<div class='metric-card'><p style='text-align: center; color: #f8fafc;'><strong>Total Listings</strong><br/><span style='font-size: 2em; color: #1f5aac;'>{len(df)}</span></p></div>", unsafe_allow_html=True)
    with col2:
        avg_price = df['price'].mean()
        st.markdown(f"<div class='metric-card'><p style='text-align: center; color: #f8fafc;'><strong>Avg Price</strong><br/><span style='font-size: 1.8em; color: #1f5aac;'>AED {avg_price/1e6:.1f}M</span></p></div>", unsafe_allow_html=True)
    with col3:
        avg_size = df['land_size'].mean()
        st.markdown(f"<div class='metric-card'><p style='text-align: center; color: #f8fafc;'><strong>Avg Size</strong><br/><span style='font-size: 1.8em; color: #1f5aac;'>{avg_size:.0f} sqft</span></p></div>", unsafe_allow_html=True)
    with col4:
        most_common = df['property_type'].mode()[0]
        st.markdown(f"<div class='metric-card'><p style='text-align: center; color: #f8fafc;'><strong>Popular Type</strong><br/><span style='font-size: 1.5em; color: #1f5aac;'>{most_common}</span></p></div>", unsafe_allow_html=True)
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Price Distribution by Location")
        price_by_location = df.groupby('location')['price'].mean().sort_values(ascending=False)
        st.bar_chart(price_by_location)
    
    with col2:
        st.markdown("### Property Count by Type")
        type_counts = df['property_type'].value_counts()
        st.bar_chart(type_counts)
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Average Price by Bedroom Count")
        price_by_bed = df.groupby('bedrooms')['price'].mean().sort_index()
        st.line_chart(price_by_bed)
    
    with col2:
        st.markdown("### Properties by Location")
        location_counts = df['location'].value_counts()
        st.bar_chart(location_counts)
    
    st.divider()
    
    st.markdown("### Market Insights")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        furnished_count = (df['furnished'] == True).sum()
        unfurnished_count = (df['furnished'] == False).sum()
        st.markdown(f"""
        <div class='metric-card'>
            <h4 style='color: #e2efff; margin-top: 0;'>Furnished vs Unfurnished</h4>
            <p style='color: #f8fafc;'>
            <strong>Furnished:</strong> {furnished_count} ({furnished_count/len(df)*100:.0f}%)<br/>
            <strong>Unfurnished:</strong> {unfurnished_count} ({unfurnished_count/len(df)*100:.0f}%)
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        parking_count = (df['parking'] == True).sum()
        pool_count = (df['pool'] == True).sum()
        st.markdown(f"""
        <div class='metric-card'>
            <h4 style='color: #e2efff; margin-top: 0;'>Amenities</h4>
            <p style='color: #f8fafc;'>
            <strong>With Parking:</strong> {parking_count} ({parking_count/len(df)*100:.0f}%)<br/>
            <strong>With Pool:</strong> {pool_count} ({pool_count/len(df)*100:.0f}%)
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_metro_dist = df['metro_distance'].mean()
        avg_age = df['building_age'].mean()
        st.markdown(f"""
        <div class='metric-card'>
            <h4 style='color: #e2efff; margin-top: 0;'>Average Metrics</h4>
            <p style='color: #f8fafc;'>
            <strong>Metro Distance:</strong> {avg_metro_dist:.0f}m<br/>
            <strong>Building Age:</strong> {avg_age:.1f} years
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    st.markdown("### Tableau Dashboard (Original)")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.link_button("📊 Open Original Tableau Dashboard", "http://localhost:8000/interactive.html", use_container_width=True)
    
    st.markdown("""
    <div style='background: rgba(255,255,255,0.05); padding: 20px; border-radius: 12px; text-align: center; color: #f8fafc;'>
        <p><strong>Advanced visualizations</strong> with detailed market analysis, price trends, and location intelligence.</p>
        <p>Access the full interactive dashboard using the button above or through the original HTML interface.</p>
    </div>
    """, unsafe_allow_html=True)

elif page == "Property Management":
    st.title("🏢 Property Management")
    st.markdown("Add, edit, or manage properties!")

    if 'properties' not in st.session_state:
        st.session_state.properties = df.copy()

    action = st.radio("Action", ["View Properties", "Add Property"])

    if action == "View Properties":
        st.dataframe(st.session_state.properties)
    elif action == "Add Property":
        with st.form("add_property"):
            new_price = st.number_input("Price")
            new_size = st.number_input("Land Size")
            new_location = st.selectbox("Location", list(location_coords.keys()))
            new_type = st.selectbox("Type", ['Apartment', 'Villa', 'Townhouse', 'Office', 'Land'])
            new_bedrooms = st.slider("Bedrooms", 0, 5)
            submitted = st.form_submit_button("Add Property")
            if submitted:
                new_id = len(st.session_state.properties) + 1
                new_row = pd.DataFrame({
                    'id': [new_id],
                    'price': [new_price],
                    'land_size': [new_size],
                    'location': [new_location],
                    'property_type': [new_type],
                    'bedrooms': [new_bedrooms],
                    'bathrooms': [2],
                    'furnished': [False],
                    'parking': [True],
                    'pool': [False],
                    'gym': [False],
                    'metro_distance': [1000],
                    'building_age': [0],
                    'floor': [1],
                    'view_type': ['City'],
                    'image_url': [get_image_url(new_type, new_id)],
                    'lat': [generate_coordinates(new_location)[0]],
                    'lon': [generate_coordinates(new_location)[1]]
                })
                st.session_state.properties = pd.concat([st.session_state.properties, new_row], ignore_index=True)
                st.success("Property added!")

elif page == "AI Chatbot":
    st.title("🧠 AI Chatbot")
    st.markdown("Ask me about Dubai real estate!")

    user_question = st.text_input("Ask a question:")
    if st.button("Ask"):
        # Simple chatbot responses
        responses = {
            "best villa under 2m": "Check out villas in Dubai Marina or Palm Jumeirah under 2M AED.",
            "apartments near metro": "Look for properties in Business Bay or Al Barsha close to metro stations.",
            "roi in marina": "Dubai Marina typically offers 5-7% rental yield."
        }
        response = responses.get(user_question.lower(), "I'm still learning! Please check the other sections for detailed information.")
        st.write(response)

elif page == "Analytics Dashboard":
    st.title("📊 Analytics Dashboard")
    st.markdown("Key insights and trends!")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Properties", len(df))
    with col2:
        st.metric("Average Price", f"{df['price'].mean():,.0f} AED")
    with col3:
        st.metric("Most Common Type", df['property_type'].mode()[0])

    st.subheader("Price Distribution")
    st.bar_chart(df.groupby('location')['price'].mean())

    st.subheader("Property Types")
    st.bar_chart(df['property_type'].value_counts())

# Additional features in sidebar
st.sidebar.header("Additional Features")
if st.sidebar.button("Show Price Statistics"):
    st.sidebar.write(df['price'].describe())

if st.sidebar.button("Show Location Distribution"):
    location_counts = df['location'].value_counts()
    st.sidebar.bar_chart(location_counts)