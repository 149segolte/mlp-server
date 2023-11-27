import sys
import json
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    # load model pkl
    model_file = sys.argv[1]
    model = pickle.load(open(model_file, 'rb'))
    input = json.loads(sys.argv[2])
    '''
    structure of input is:
        input = {
            'data': [
                [1, 2, 3, 4],
                [5, 6, 7, 8]
            ]
        }
    '''

    # convert input to dataframe
    input_df = pd.DataFrame(input['data'])

    # predict
    pred = model.predict(input_df)

    # return prediction
    print(json.dumps(pred.tolist()))
