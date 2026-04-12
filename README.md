# EV-Charging-Station-Recommendation
This project presents a spatio-temporal EV charging station recommendation system that predicts future charging demand and suggests optimal nearby stations.

The system combines Graph Attention Networks (GAT) for spatial relationships and LSTM with temporal attention for time-based pattern learning, enabling accurate short-term demand prediction and real-time recommendations.

---

## Features

- Demand Prediction  
  Predicts EV charging demand for the next 30 minutes using past 1-hour data.

- Spatio-Temporal Modeling  
  - GAT captures spatial relationships between stations  
  - LSTM captures temporal patterns  
  - Temporal attention highlights important time steps  

- Smart Recommendation  
  Recommends top 3 nearby stations based on:
  - Predicted demand  
  - Distance from user  
  - Station capacity  

- User Interface  
  Built using Streamlit with pincode input and ranked station display.

---

## Model Architecture

- Input: Past 1 hour data (12 time steps)  
- Output: Next 30 minutes prediction (6 time steps)

Components:
- Graph Attention Network (GAT) for spatial dependency modeling  
- Long Short-Term Memory (LSTM) for temporal sequence learning  
- Temporal Attention mechanism  

---

## Evaluation Metrics

The model is evaluated using:
- Mean Absolute Error (MAE)  
- Root Mean Squared Error (RMSE)  
- Mean Absolute Percentage Error (MAPE)  
- R² Score  

---

## Tech Stack

- Programming Language: Python  
- Libraries:
  - PyTorch  
  - Pandas  
  - NumPy  
  - Scikit-learn  
  - Streamlit  


---

## How to Run

1. Install dependencies:
   pip install -r requirements.txt
2. Run the model:
   python main.py
3. Run the user interface:
   streamlit run app.py


---

## How Recommendation Works

1. Predict demand for all stations  
2. Convert user pincode to latitude and longitude  
3. Identify nearby stations  
4. Compute score: score = predicted_demand / capacity


5. Recommend stations with:
- Low demand  
- High availability  
- Minimum distance  

---

## Use Case

- Helps EV users find less crowded charging stations  
- Reduces waiting time  
- Improves efficiency of EV infrastructure  

---

## Novelty

- Combines spatial and temporal learning  
- Uses Graph Attention Networks to model station relationships  
- Provides recommendation along with prediction  

---

## Future Work

- Integration of real-time traffic data  
- API integration for live station data  
- Mobile application development  

---



