import numpy as np
import dill
from flask import Flask, render_template, request, flash
from model import FlatModel

# Initiate the app.
app = Flask(__name__)

# Set up secret key.
app.config['SECRET_KEY'] = 'many bytes'

@app.route('/', methods=('GET', 'POST'))
def index():
    result = ''
    if request.method == 'POST':
        flat = FlatModel(request.form)
        error = flat.validate_data()

        if error is None:
            X = flat.transform()
            file = open('regressor_model', 'rb')
            model = dill.load(file)
            file.close()
            result = f'{round(np.expm1(model.predict(X))[0])} руб.'

        flash(error)

    return render_template('index.html', result=result)
