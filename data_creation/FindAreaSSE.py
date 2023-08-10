import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import math

# returns if a point is in a area circle
# IN: center_x,center_y -> coordinates of centre,
# IN: radius -> how large of circle
# IN: x, y -> the coordinates of the point
def in_circle(center_x, center_y, radius, x, y):
    dist = math.sqrt((center_x - x) ** 2 + (center_y - y) ** 2)
    return dist <= radius

# finds the number of valid SSE's for a station
# IN: pandas df - stations_c [name, lat, lon]
# IN: pandas df - sortSSE [date, lat, lon]
# OUT: pandas df - selectedSSE [date, lat, lon]
def find_use_stations(station_c, sortSSE, rad = 1):
    i = 0
    selectedSSE = []
    for i_sse in range(sortSSE.shape[0]):
        valid = False
        sse = sortSSE.iloc[i_sse]
        i_station = 0
        while (valid==False and i_station < station_c.shape[0]):
            
            station = station_c.iloc[i_station]
            result = in_circle(station['lon'], station['lat'], rad, sse['lon'], sse['lat'])
            if (result):
                valid = True
                selectedSSE.append(sse)
            i_station = i_station + 1
    
    # selectedSSE = pd.DataFrame(selectedSSE)
    return selectedSSE
            

# returns visualisation of the Area where the stations are
# IN: station_c -> pandas dataframe, [date, lat, lon]
# OUT: null
def display_visualisation(station_c, sortSSE, display_circles = False, rad=1):
    fig, ax = plt.subplots()
    circles = []

    if (display_circles):
        for i in range(len(station_c)):
            circle1 = plt.Circle((station_c.iloc[i]['lon'], station_c.iloc[i]['lat']), rad, color='grey', alpha=0.2)
            circles.append(circle1)
        for x in circles:
            ax.add_patch(x)

    ax.scatter(station_c['lon'], station_c['lat'], marker="s", s = 10)
    ax.scatter(sortSSE['lon'], sortSSE['lat'], marker="^", color="red", alpha=0.2)    
    plt.title("SSE's and Stations")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    plt.show();


# # # create fake stations array
# stationsarray = np.empty((0,3))
# stationsarray = np.append(stationsarray, np.array([["020978", 35.4866, 135.7556]]), axis=0)
# stationsarray = np.append(stationsarray, np.array([["021082", 32.8522, 132.6292]]), axis=0)
# stationsarray = np.append(stationsarray, np.array([["021012", 33.6688, 135.8821]]), axis=0)
# stationsarray = np.append(stationsarray, np.array([["020999", 34.4203, 136.4745]]), axis=0)
# stationsarray = np.append(stationsarray, np.array([["021000", 33.8771, 133.9283]]), axis=0)

# stations_cord_df = pd.DataFrame(stationsarray, columns=["station", "lat", "lon"])
# stations_cord_df["lat"] = stations_cord_df["lat"].astype(float)
# stations_cord_df["lon"] = stations_cord_df["lon"].astype(float)

# knownSSE = pd.read_csv("Nishimura/Nishimura2013.csv")
# knownSSE = knownSSE[['date', 'lat', 'lon']]
# sortedSSE = knownSSE.sort_values(by="date").drop_duplicates().reset_index(drop=True)

# one_station = stations_cord_df.iloc[3:5]
# one_sse = knownSSE.iloc[0:2]


# print(sortedSSE.shape[0])
# selectedSSE = find_use_stations(stations_cord_df, sortedSSE)

# display_visualisation(stations_cord_df, sortedSSE, display_circles=True)
# display_visualisation(stations_cord_df, selectedSSE, display_circles=True)




# results = in_circle(one_station['lon'].iloc[0], one_station['lat'].iloc[0], 1, one_sse['lon'].iloc[1], one_sse['lat'].iloc[1])
# print(results)

# print(one_station['lon'])
# results = in_circle(one_station['lon'], one_station['lat'], 1, one_sse['lon'], one_sse['lat'])
# print(results)