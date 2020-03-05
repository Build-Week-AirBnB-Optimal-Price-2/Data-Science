"""
__author__ : Loc Huynh, Nick Cole, Neal Whitlock
__credits__ : Julie Wang
__license__ : MIT
__version__ : 1.0
"""

from . import utils 
from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import regularizers, metrics


""" Set up flask app """
application = app = Flask(__name__)

""" Get encoder and predictive model ready """
scaler = pickle.load(open('Model/scaler.pkl', 'rb'))
# model = pickle.load(open('model.pkl', 'rb'))

# This code is because the model was giving trouble when pickling
from tensorflow.keras.models import model_from_json
# load json and create model
json_file = open('Model/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("Model/model.h5")

model.compile(
    loss='mean_absolute_error',
    optimizer='nadam',
    metrics=[metrics.mae])


@app.route('/', methods=['GET', 'POST'])
def home():
    """ Home page of site """
    message = "You are now home"
    return render_template('base.html', message=message)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """ Gets data to run through model. Returns prediction in JSON. """
    data = request.get_json()

    """ Input from user. Each field is required for the model. """
    host_is_superhost = data['host_is_superhost']
    latitude = data['latitude']
    longitude = data['longitude']
    property_type = data['property_type']
    accommodates = data['accommodates']
    bathrooms = data['bathrooms']
    bedrooms = data['bedrooms']
    room_type = data['room_type']
    bed_type = data['bed_type']
    size = data['size']
    distance = data['distance']
    security_deposit = data['security_deposit']
    cleaning_fee = data['cleaning_fee']
    guests_included = data['guests_included']
    extra_people = data['extra_people']
    minimum_nights = data['minimum_nights']
    cancellation_policy = data['cancellation_policy']
    tv = data['tv']
    wifi = data['wifi']
    washer = data['washer']
    dryer = data['dryer']
    kitchen = data['kitchen']
    heating = data['heating']
    free_parking = data['free_parking']
    smoking_allowed = data['smoking_allowed']
    neighbourhood = data['neighbourhood']
    instant_bookable = data['instant_bookable']
    is_business_travel_ready = data['is_business_travel_ready']

    """ Place for default values if any are used. """

    """ Features dictionary for model """
    features = {'host_is_superhost': host_is_superhost,
                'latitude': latitude,
                'longitude': longitude,
                'property_type': property_type,
                'accommodates': accommodates,
                'bathrooms': bathrooms,
                'bedrooms': bedrooms,
                'room_type': room_type,
                'bed_type': bed_type,
                'size': size,
                'distance': distance,
                'security_deposit': security_deposit,
                'cleaning_fee': cleaning_fee,
                'guests_included': guests_included,
                'extra_people': extra_people,
                'minimum_nights': minimum_nights,
                'cancellation_policy': cancellation_policy,
                'tv': tv,
                'wifi': wifi,
                'washer': washer,
                'dryer': dryer,
                'kitchen': kitchen,
                'heating': heating,
                'free_parking': free_parking,
                'smoking_allowed': smoking_allowed,
                'neighbourhood': neighbourhood,
                'instant_bookable': instant_bookable,
                'is_business_travel_ready': is_business_travel_ready}

    """ Converts data into DataFrame """
    predict_data = pd.DataFrame(features, index=[1])
    features_scaler = scaler.transform(predict_data)

    """ Feed the model the data """
    prediction = model.predict(features_scaler)

    """ Return prediction in JSON """
    return jsonify({'features': features,
                    'prediction': prediction[0]})


if __name__ == '__main__':
    app.run(debug=True)
