"""
scripts/make_dataset.py
Generates or loads Kaggle Food_Delivery_Times.csv and outputs data/orders.csv
with promised_time_min and is_late flags.
"""

import os
import random
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
OUTPUT_FILE = os.path.join(DATA_DIR, "orders.csv")
KAGGLE_FILE = os.path.join(DATA_DIR, "Food_Delivery_Times.csv")

SAMPLE_CUSTOMERS = [
    ("Aarav Mehta", "+91 98201 12345", "Andheri West, Mumbai"),
    ("Priya Sharma", "+91 98334 56789", "Bandra East, Mumbai"),
    ("Rohan Kulkarni", "+91 97654 32109", "Kothrud, Pune"),
    ("Ananya Iyer", "+91 98451 23456", "Indiranagar, Bengaluru"),
    ("Vikram Singh", "+91 98111 67890", "Cyber Hub, Gurugram"),
    ("Neha Deshmukh", "+91 98230 45678", "Viman Nagar, Pune"),
    ("Kabir Sen", "+91 98301 98765", "Salt Lake, Kolkata"),
    ("Pooja Nair", "+91 98470 11223", "Kaloor, Kochi"),
    ("Aditya Verma", "+91 98102 33445", "Hauz Khas, New Delhi"),
    ("Ishita Roy", "+91 98312 44556", "Park Street, Kolkata"),
    ("Siddharth Rao", "+91 98490 55667", "Banjara Hills, Hyderabad"),
    ("Deepika Patel", "+91 98250 66778", "Navrangpura, Ahmedabad"),
]

def generate_synthetic_orders(num_rows: int = 5000, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    random.seed(seed)

    weathers = ["Clear", "Rainy", "Foggy", "Windy", "Stormy"]
    weather_probs = [0.50, 0.22, 0.12, 0.11, 0.05]

    traffic_levels = ["Low", "Medium", "High", "Jam"]
    traffic_probs = [0.25, 0.40, 0.25, 0.10]

    times_of_day = ["Morning", "Afternoon", "Evening", "Night"]
    time_probs = [0.25, 0.25, 0.35, 0.15]

    vehicle_types = ["Bicycle", "Scooter", "Motorcycle", "Electric_Bike"]
    vehicle_probs = [0.10, 0.45, 0.35, 0.10]

    vehicle_pace_min_per_km = {
        "Bicycle": 4.8,
        "Scooter": 2.7,
        "Motorcycle": 2.3,
        "Electric_Bike": 3.0,
    }

    traffic_multipliers = {
        "Low": 1.0,
        "Medium": 1.35,
        "High": 1.85,
        "Jam": 2.45,
    }

    weather_multipliers = {
        "Clear": 1.0,
        "Windy": 1.15,
        "Foggy": 1.30,
        "Rainy": 1.65,
        "Stormy": 2.15,
    }

    time_multipliers = {
        "Morning": 1.05,
        "Afternoon": 0.95,
        "Evening": 1.38,
        "Night": 0.90,
    }

    rows = []
    for i in range(1, num_rows + 1):
        order_id = f"DB-{10000 + i}"
        cust = random.choice(SAMPLE_CUSTOMERS)
        cust_name = cust[0]
        cust_phone = cust[1]
        delivery_location = cust[2]

        # Distance: typical city delivery radius 1.0 - 16.0 km
        dist_km = round(float(np.random.gamma(shape=2.5, scale=2.0) + 0.8), 2)
        dist_km = min(dist_km, 20.0)

        weather = np.random.choice(weathers, p=weather_probs)
        traffic = np.random.choice(traffic_levels, p=traffic_probs)
        time_of_day = np.random.choice(times_of_day, p=time_probs)
        vehicle = np.random.choice(vehicle_types, p=vehicle_probs)

        # Preparation time: 10 to 35 min
        prep_time = int(np.random.normal(loc=18, scale=5))
        prep_time = max(8, min(prep_time, 45))

        # Experience: 0.5 to 10 years
        courier_exp = round(float(np.random.exponential(scale=2.2) + 0.5), 1)
        courier_exp = min(courier_exp, 12.0)

        # Transit calculation
        base_pace = vehicle_pace_min_per_km[vehicle]
        t_mult = traffic_multipliers[traffic]
        w_mult = weather_multipliers[weather]
        tod_mult = time_multipliers[time_of_day]
        exp_discount = 1.0 - 0.025 * min(courier_exp, 8.0)

        transit_time = dist_km * base_pace * t_mult * w_mult * tod_mult * exp_discount
        noise = np.random.normal(0, 2.5)

        total_delivery = prep_time + transit_time + noise
        total_delivery = max(12.0, round(float(total_delivery), 1))

        # Promised time: typical tiffin/bakery commitment is 35-50 min
        promised = random.choice([35, 40, 45, 50])

        is_late = int(total_delivery > promised)

        rows.append({
            "Order_ID": order_id,
            "Customer_Name": cust_name,
            "Customer_Phone": cust_phone,
            "Delivery_Location": delivery_location,
            "Distance_km": dist_km,
            "Weather": weather,
            "Traffic_Level": traffic,
            "Time_of_Day": time_of_day,
            "Vehicle_Type": vehicle,
            "Preparation_Time_min": prep_time,
            "Courier_Experience_yrs": courier_exp,
            "promised_time_min": promised,
            "Delivery_Time_min": total_delivery,
            "is_late": is_late,
        })

    return pd.DataFrame(rows)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(KAGGLE_FILE):
        print(f"Loading existing Kaggle dataset from {KAGGLE_FILE}...")
        df = pd.read_csv(KAGGLE_FILE)
        if "promised_time_min" not in df.columns:
            df["promised_time_min"] = 40
        if "is_late" not in df.columns:
            df["is_late"] = (df["Delivery_Time_min"] > df["promised_time_min"]).astype(int)
    else:
        print(f"Kaggle file not found. Generating 5,000 synthetic rows at {OUTPUT_FILE}...")
        df = generate_synthetic_orders(num_rows=5000, seed=42)

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Successfully generated {len(df)} orders saved to {OUTPUT_FILE}.")
    print("Sample distribution:")
    print(df[["Distance_km", "Preparation_Time_min", "Delivery_Time_min", "is_late"]].describe())


if __name__ == "__main__":
    main()
