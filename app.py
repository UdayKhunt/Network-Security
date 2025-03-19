import os,sys
import certifi
from src.exception import NetworkSecurityException
from src.logger import logging
from src.pipeline.train_pipeline import TrainingPipeline

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile,Request
from uvicorn import run as app_run
from fastapi.responses import Response
from starlette.responses import RedirectResponse

import pandas as pd
from src.utils.main_utils import load_obj
from src.utils.estimators import NetworkEstimatior

ca = certifi.where()

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.templating import Jinja2Templates
templates = Jinja2Templates(directory="./templates")

@app.get('/' , tags = ['authentication'])
async def index():
    return RedirectResponse(url = '/docs')

@app.get('/train')
async def train_route():
    try:
        training_pipeline = TrainingPipeline()
        training_pipeline.run_pipeline()
        return Response('Training is successful')
    except Exception as e:
            raise NetworkSecurityException(e,sys)
    
@app.post('/predict')
async def predict_route(request: Request,file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        preprocessor = load_obj('final_objects/preprocessor.pkl')
        model = load_obj('final_objects/model.pkl')

        network_estimator = NetworkEstimatior(model , preprocessor)
        pred = network_estimator.predict(df)

        print(pred)

        df['Predicted'] = pred
        df.to_csv('prediction_output/output.csv')
        table_html = df.to_html(classes='table table-striped')
        #print(table_html)
        return templates.TemplateResponse("table.html", {"request": request, "table": table_html})

    except Exception as e:
            raise NetworkSecurityException(e,sys)  
if __name__=='__main__':
     app_run(app , host = '0.0.0.0' , port = 8000)    
    