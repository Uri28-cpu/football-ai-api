from fastapi import FastAPI
import joblib

model = joblib.load("model.pkl")
team_power = joblib.load("team_power.pkl")
team_lookup = joblib.load("team_lookup.pkl")

app = FastAPI()

@app.get("/")
def home():
    return {"status": "API Running"}

@app.get("/predict")
def predict(home: str, away: str):

    home_id = team_lookup.get(home.lower())
    away_id = team_lookup.get(away.lower())

    if home_id is None or away_id is None:
        return {"error": "Team not found"}

    home_attack = team_power[home_id]["attack"]
    away_attack = team_power[away_id]["attack"]
    home_def = team_power[home_id]["defense"]
    away_def = team_power[away_id]["defense"]

    features = [[
        home_attack,
        away_attack,
        home_def,
        away_def,
        1,
        1
    ]]

    proba = model.predict_proba(features)[0]
    prediction = model.predict(features)[0]

    return {
        "Prediction": "Over 2.5" if prediction == 1 else "Under 2.5",
        "Over Probability": round(float(proba[1])*100,2),
        "Under Probability": round(float(proba[0])*100,2)
}
