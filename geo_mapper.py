import streamlit as st
import pydeck as pdk

def plot_map(df):
    # Try to detect latitude and longitude columns
    lat_candidates = [col for col in df.columns if 'lat' in col.lower() or 'latitude' in col.lower()]
    lon_candidates = [col for col in df.columns if 'lon' in col.lower() or 'longitude' in col.lower() or 'long' in col.lower()]

    lat_col = lat_candidates[0] if lat_candidates else None
    lon_col = lon_candidates[0] if lon_candidates else None

    if lat_col and lon_col:
        try:
            # Drop rows with null lat/lon
            map_df = df[[lat_col, lon_col]].dropna()
            map_df = map_df.astype({lat_col: float, lon_col: float})  # Ensure numeric

            if map_df.empty:
                st.warning("Latitude/Longitude data is missing or invalid.")
                return

            st.pydeck_chart(pdk.Deck(
                map_style='mapbox://styles/mapbox/light-v9',
                initial_view_state=pdk.ViewState(
                    latitude=map_df[lat_col].mean(),
                    longitude=map_df[lon_col].mean(),
                    zoom=4,
                    pitch=0,
                ),
                layers=[
                    pdk.Layer(
                        "ScatterplotLayer",
                        data=map_df,
                        get_position=f"[{lon_col}, {lat_col}]",
                        get_radius=8000,
                        get_color=[200, 30, 0, 140],
                        pickable=True,
                    ),
                ],
                tooltip={"text": f"{lat_col}: {{{lat_col}}}\n{lon_col}: {{{lon_col}}}"}
            ))
        except Exception as e:
            st.error(f"Error plotting map: {e}")
    else:
        st.warning("Could not find latitude and longitude columns. Please ensure your data includes columns named like 'lat', 'latitude', 'lon', or 'longitude'.")
