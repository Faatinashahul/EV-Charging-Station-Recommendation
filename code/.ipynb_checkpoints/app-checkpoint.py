import streamlit as st
import pandas as pd
import math

st.set_page_config(
    page_title="ChargeSZ · EV Station Finder",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
    background-color: #0a0e1a;
    color: #e8eaf0;
}
.stApp { background-color: #0a0e1a; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem 4rem 3rem; max-width: 900px; }

/* Hero */
.hero {
    text-align: center;
    padding: 3.5rem 0 2.5rem 0;
}
.hero-badge {
    display: inline-block;
    background: linear-gradient(135deg, #00e5a0 0%, #00b4d8 100%);
    color: #0a0e1a;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    padding: 0.35rem 1rem;
    border-radius: 100px;
    margin-bottom: 1.2rem;
}
.hero h1 {
    font-size: clamp(2.2rem, 5vw, 3.5rem);
    font-weight: 800;
    line-height: 1.1;
    margin: 0 0 1rem 0;
    background: linear-gradient(135deg, #ffffff 30%, #00e5a0 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero p {
    color: #7b8299;
    font-size: 1rem;
    font-weight: 300;
    max-width: 480px;
    margin: 0 auto;
    line-height: 1.7;
}

/* Divider */
.divider {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, #1e2535, transparent);
    margin: 0.5rem 0 2rem 0;
}

/* Search */
.search-wrap {
    background: #111827;
    border: 1px solid #1e2a3a;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
}
.search-label {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #00e5a0;
    margin-bottom: 0.6rem;
}

/* Input */
.stTextInput > div > div > input {
    background: #0d1420 !important;
    border: 1.5px solid #1e2a3a !important;
    border-radius: 10px !important;
    color: #e8eaf0 !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 1.05rem !important;
    padding: 0.75rem 1rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: #00e5a0 !important;
    box-shadow: 0 0 0 3px rgba(0,229,160,0.12) !important;
}

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #00e5a0 0%, #00b4d8 100%) !important;
    color: #0a0e1a !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.75rem 2.2rem !important;
    width: 100% !important;
}
.stButton > button:hover { opacity: 0.88 !important; }

/* Button column alignment */
div[data-testid="column"]:last-child .stButton {
    margin-top: 1.85rem;
}

/* Cards */
.cards-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1.2rem;
    margin-top: 1.5rem;
}
.station-card {
    background: #111827;
    border: 1px solid #1e2a3a;
    border-radius: 16px;
    padding: 1.6rem 1.8rem;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s, border-color 0.2s;
}
.station-card:hover { transform: translateY(-3px); border-color: #2a3a50; }
.station-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
}
.card-rank-1::before { background: linear-gradient(90deg, #00e5a0, #00b4d8); }
.card-rank-2::before { background: linear-gradient(90deg, #f59e0b, #f97316); }
.card-rank-3::before { background: linear-gradient(90deg, #6366f1, #8b5cf6); }

.card-top { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 1.2rem; }
.card-id { font-size: 1.4rem; font-weight: 800; color: #ffffff; }
.card-id span { font-size: 0.75rem; color: #7b8299; font-weight: 400; margin-left: 4px; }

.status-badge {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    padding: 0.3rem 0.75rem;
    border-radius: 100px;
    text-transform: uppercase;
}
.status-low    { background: rgba(0,229,160,0.15); color: #00e5a0; }
.status-medium { background: rgba(245,158,11,0.15); color: #f59e0b; }
.status-high   { background: rgba(239,68,68,0.15);  color: #ef4444; }

.card-stat { margin-bottom: 0.7rem; }
.card-stat-label {
    font-size: 0.7rem;
    font-weight: 500;
    color: #7b8299;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.15rem;
}
.card-stat-value { font-size: 1.1rem; font-weight: 700; color: #e8eaf0; }

.avail-bar-bg {
    background: #1e2535;
    border-radius: 100px;
    height: 6px;
    margin-top: 0.4rem;
    overflow: hidden;
}
.avail-bar-fill { height: 100%; border-radius: 100px; }

.section-title {
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #7b8299;
    margin-bottom: 0.5rem;
    margin-top: 2rem;
}
</style>
""", unsafe_allow_html=True)


# ── Data ──
@st.cache_data
def load_data():
    stations    = pd.read_csv("datasets/stations.csv")
    predictions = pd.read_csv("results/predictions.csv")
    data        = stations.head(len(predictions)).copy()
    data["predicted_demand"] = predictions["predicted_demand"].values
    return data

data = load_data()
data = data[data["count"] > 0].reset_index(drop=True)


# ── Helpers ──
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def pincode_to_latlon(pincode):
    from geopy.geocoders import Nominatim
    geolocator = Nominatim(user_agent="chargesz_app")
    location   = geolocator.geocode({"postalcode": pincode, "country": "China"}, timeout=10)
    if location is None:
        raise ValueError(f"Postcode '{pincode}' not found. Try 518000, 518048 or 518055.")
    return location.latitude, location.longitude

def recommend(user_lat, user_lon):
    df = data.copy()
    df["dist_km"] = df.apply(
        lambda r: haversine_km(user_lat, user_lon, r["latitude"], r["longitude"]), axis=1
    )
    nearby          = df.sort_values("dist_km").head(10).copy()
    # nearby["score"] = nearby["predicted_demand"] / nearby["count"]
    nearby["score"] = nearby["predicted_demand"]
    return nearby.sort_values(["score", "dist_km"]).head(3).reset_index(drop=True)

def status(occupancy):
    if occupancy < 0.30:   return "Low Demand",   "low",    "#00e5a0"
    elif occupancy < 0.65: return "Moderate",     "medium", "#f59e0b"
    else:                  return "High Demand",  "high",   "#ef4444"


# ── Hero ──
st.markdown("""
<div class="hero">
    <div class="hero-badge">⚡ Shenzhen · STGAT Powered</div>
    <h1>Find Your Nearest<br>Charging Station</h1>
    <p>Enter your postcode and we'll find the best available charging spots near you.</p>
</div>
<hr class="divider"/>
""", unsafe_allow_html=True)


# ── Search ──
st.markdown('<div class="search-wrap">', unsafe_allow_html=True)
st.markdown('<div class="search-label">Enter your postcode</div>', unsafe_allow_html=True)

col_in, col_btn = st.columns([4, 1])
with col_in:
    pincode = st.text_input("", placeholder="e.g. 518000, 518048, 518055 …", label_visibility="collapsed")
with col_btn:
    search = st.button("Search ⚡")

st.markdown("</div>", unsafe_allow_html=True)


# ── Results ──
if search and pincode.strip():
    with st.spinner("Finding stations near you…"):
        try:
            user_lat, user_lon = pincode_to_latlon(pincode.strip())
            result = recommend(user_lat, user_lon)

            st.success(f"📍 Postcode **{pincode}** → {user_lat:.4f}°N, {user_lon:.4f}°E")
            st.markdown('<div class="section-title">Top 3 Recommended Stations</div>', unsafe_allow_html=True)

            card_html = '<div class="cards-row">'
            for i, row in result.iterrows():
                rank = i + 1
                #status_txt, status_cls, bar_clr = status(row["score"])
                status_txt, status_cls, bar_clr = status(row["predicted_demand"])
                demand_pct = int(row["predicted_demand"] * 100)

                card_html += f"""
                <div class="station-card card-rank-{rank}">
                    <div class="card-top">
                        <div class="card-id">#{int(row['station_id'])}<span>Station</span></div>
                        <span class="status-badge status-{status_cls}">{status_txt}</span>
                    </div>
                    <div class="card-stat">
                        <div class="card-stat-label">Distance</div>
                        <div class="card-stat-value">{row['dist_km']:.2f} km away</div>
                    </div>
                    <div class="card-stat">
                        <div class="card-stat-label">Total Slots</div>
                        <div class="card-stat-value">{int(row['count'])} chargers</div>
                    </div>
                    <div class="card-stat">
                        <div class="card-stat-label">Predicted Demand</div>
                        <div class="card-stat-value">{demand_pct}%</div>
                        <div class="avail-bar-bg">
                            <div class="avail-bar-fill" style="width:{demand_pct}%; background:{bar_clr};"></div>
                        </div>
                    </div>
                </div>"""
            card_html += "</div>"
            st.markdown(card_html, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ {e}")

elif search and not pincode.strip():
    st.warning("Please enter a postcode first.")