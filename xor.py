"""
author: Nazmul 
email: md.nazmul.islam0087@gmail.com
"""


from utils.model import Perceptron
from utils.all_utils import prepare_data, save_model, save_plot
import pandas as pd
import numpy as np
import logging
import os 

logging_str = "[%(asctime)s: %(levelname)s: %(module)s] %(message)s"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename= os.path.join(log_dir,"running_logs.log"),level=logging.INFO, format=logging_str, filemode="a")
#logging.basicConfig(level=logging.INFO, format=logging_str)

def main(data, eta, epochs, modelfilename,plotfilename):
    df = pd.DataFrame(data)
    logging.info(df)
    X,y = prepare_data(df)
    model = Perceptron(eta=eta, epochs=epochs)
    model.fit(X, y)

    _ = model.total_loss()

    save_model(model, filename=modelfilename)
    save_plot(df, file_name=plotfilename, model=model)

if __name__=="__main__": # << entry point <<
    XOR = {
        "x1": [0,0,1,1],
        "x2": [0,1,0,1],
        "y": [0,1,1,0],
    }
    
    ETA = 0.3 # 0 and 1
    EPOCHS = 10
    
    #main(data=XOR, eta=ETA, epochs=EPOCHS, modelfilename="xor.model", plotfilename="xor.png")
    try:
        logging.info(">>>>> starting training >>>>>")
        main(data=XOR, eta=ETA, epochs=EPOCHS, modelfilename="xor.model", plotfilename="xor.png")
        logging.info("<<<<< training done successfully<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e    
    