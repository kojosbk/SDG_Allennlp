import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px


from geopy.geocoders import Nominatim
# import the library and its Marker clusterization service
import folium
from folium.plugins import MarkerCluster

st.title('Shell: Associated Locations & Organizations')

# load data
#@st.cache
def load_data():
    data = pd.read_csv('locations.csv')
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    return data

#function to get longitude and latitude data from country name
geolocator = Nominatim(user_agent="http")
#load data
data = load_data()

data["loc"] = data["location"].apply(geolocator.geocode)
data["point"]= data["loc"].apply(lambda loc: tuple(loc.point) if loc else None)
data[['lat', 'lon', 'altitude']] = pd.DataFrame(data['point'].to_list(), index=data.index)

#df = data[['lat', 'lon', 'altitude']] 
locations = st.multiselect('Select one or more locations', data['location'])
pick_all_locations = st.checkbox(' or all locations')

location_selected = False
if not pick_all_locations:
    if locations:
        data = data[data['location'].isin(locations)]
        location_selected = True
else:
    location_selected = True

if location_selected:
    st.dataframe(data.style)

st.map(data[["lat","lon"]])

fig = px.bar(data, x=data["counts"], y=data["location"], orientation='h', color="location", width=800, height=1000).update_yaxes(categoryorder="total ascending")

st.plotly_chart(fig)