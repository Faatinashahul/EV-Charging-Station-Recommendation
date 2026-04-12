import pandas as pd
import math

# -------------------------------
# Load data
# -------------------------------

stations = pd.read_csv("datasets/stations.csv")
predictions = pd.read_csv("results/predictions.csv")

# FIX: match sizes
data = stations.head(len(predictions)).copy()
data["predicted_demand"] = predictions["predicted_demand"].values


# -------------------------------
# Distance function
# -------------------------------
def distance(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


# -------------------------------
# Pincode → lat/lon using geopy
# -------------------------------
def pincode_to_latlon(pincode: str):
    """
    Converts a Chinese postcode (e.g. '518000') to (latitude, longitude)
    using the free Nominatim geocoding API.
    Requires internet connection and geopy installed:
        pip install geopy
    """
    try:
        from geopy.geocoders import Nominatim
        from geopy.exc import GeocoderTimedOut, GeocoderServiceError

        geolocator = Nominatim(user_agent="stgat_recommend")
        location = geolocator.geocode({"postalcode": pincode, "country": "China"}, timeout=10)

        if location is None:
            raise ValueError(
                f"Postcode '{pincode}' could not be found. "
                "Please check the postcode or try a nearby one."
            )

        return location.latitude, location.longitude

    except ImportError:
        raise ImportError(
            "geopy is not installed. Run: pip install geopy"
        )
    except GeocoderTimedOut:
        raise ConnectionError("Geocoding timed out. Check your internet connection.")
    except GeocoderServiceError as e:
        raise ConnectionError(f"Geocoding service error: {e}")


# -------------------------------
# Recommendation function
# -------------------------------
def recommend(user_lat, user_lon, top_k=3):
    # Calculate distance from user to each station
    data["dist"] = data.apply(
        lambda row: distance(user_lat, user_lon, row["latitude"], row["longitude"]),
        axis=1
    )
    data["dist_km"] = data["dist"] 
    # Take 10 nearest stations
    nearby = data.sort_values(by="dist").head(10).copy()

    # Score = demand / capacity (lower is better → less crowded)
    # nearby["score"] = nearby["predicted_demand"] / nearby["count"]
    nearby["score"] = nearby["predicted_demand"]
    # Sort by best (lowest score = most available)
    recommended = nearby.sort_values(by=["score", "dist_km"])

    return recommended.head(top_k)[
        ["station_id", "predicted_demand", "count", "score", "dist","dist_km"]
    ]


# -------------------------------
# Main: accept pincode as input
# -------------------------------
if __name__ == "__main__":
    print("=== EV Charging Station Recommender ===\n")

    pincode = input("Enter your pincode (Chinese postcode, e.g. 518000): ").strip()

    print(f"\nLooking up coordinates for postcode '{pincode}'...")
    try:
        user_lat, user_lon = pincode_to_latlon(pincode)
        print(f"Location found: latitude={user_lat:.5f}, longitude={user_lon:.5f}")
    except (ValueError, ImportError, ConnectionError) as e:
        print(f"\nError: {e}")
        exit(1)

    result = recommend(user_lat, user_lon)

    print("\nRecommended Charging Stations:\n")
    print(result.to_string(index=False))

    print("\nWhy these stations?\n")

    for _, row in result.iterrows():
        print(
            f"Station {int(row['station_id'])}: "
            f"Low demand ({row['predicted_demand']:.2f}), "
            f"capacity {int(row['count'])}, "
            f"{row['dist_km']:.2f} km away → less waiting time"
        )