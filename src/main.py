import io
import uvicorn
import pandas as pd
from fastapi.responses import JSONResponse
from prediction import predict, load_model
from fastapi import FastAPI, File, UploadFile, HTTPException

app = FastAPI(title="DON Concentration Prediction API")

# Load the trained model
model = load_model("models/xgboost_model.pkl")


# Endpoint for JSON-based prediction
@app.post("/predict/json/")
async def predict_json(data: dict):
    try:
        df = pd.DataFrame([data])
        predictions = predict(model, df)
        return {"predictions": predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing data: {e}")


# Endpoint for CSV-based prediction
@app.post("/predict/csv/")
async def predict_csv(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
        predictions = predict(model, df)
        return JSONResponse(content={"predictions": predictions.tolist()})
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing file: {e}")


# Health check endpoint
@app.get("/")
async def root():
    return {"message": "DON Concentration Prediction API is running!"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
