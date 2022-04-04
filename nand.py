from utils.model import Perceptron
from utils.all_utils import prepare_data, save_model, save_plot
import pandas as pd
import numpy as np

def main(data, eta, epochs, modelfilename,plotfilename):
    df = pd.DataFrame(data)
    print(df)
    X,y = prepare_data(df)
    model = Perceptron(eta=eta, epochs=epochs)
    model.fit(X, y)

    _ = model.total_loss()

    save_model(model, filename=modelfilename)
    save_plot(df, file_name=plotfilename, model=model)

if __name__=="__main__": # << entry point <<
    NAND = {
        "x1": [0,0,1,1],
        "x2": [0,1,0,1],
        "y": [1,1,1,0],
    }
    
    ETA = 0.3 # 0 and 1
    EPOCHS = 10
    
    main(data=NAND, eta=ETA, epochs=EPOCHS, modelfilename="nand.model", plotfilename="nand.png")
    
    