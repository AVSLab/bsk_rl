import folium

# Create a map centered on the world
m = folium.Map(location=[0, 0], zoom_start=2)

# Ground station coordinates and names
stations = [
    ("Boulder", 40.01, -105.25),
    ("Merritt", 50.11, -97.26),
    ("Singapore", 1.34, 103.81),
    ("Weilheim", 47.82, 11.09),
    ("Santiago", -33.45, -70.67),
    ("Dongara", -29.25, 114.87),
    ("Hawaii", 20.71, -156.25)
]

# Add circles for each ground station
for name, lat, lon in stations:
    folium.Marker([lat, lon], popup=name).add_to(m)
    folium.Circle(
        location=[lat, lon],
        radius=1550000,  # Radius in meters (1550 km)
        color="blue",
        fill=True,
        fill_opacity=0.2
    ).add_to(m)

# Save the map
m.save("ground_station_coverage.html")
