"""
This is the Flask app for serving the regression model for 
retrieving a player's overall crikcet rating and ranking 
"""

__date__ = "2023-05-03"
__author__ = "SamKemp"


# %% --------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
from flask import Flask, request 
import logging 
from waitress import serve
import retrieve_rating as rr


# %% --------------------------------------------------------------------------
# Set up logger
# -----------------------------------------------------------------------------

logging.basicConfig(
    format='[%(levelname)s %(name)s] %(asctime)s - %(message)s',
    level = logging.INFO,
    datefmt='%Y/%m/%d %I:%M:%S %p'
)

logger = logging.getLogger(__name__)

# %% --------------------------------------------------------------------------
# Flask App
# -----------------------------------------------------------------------------

app = Flask(__name__)

@app.route("/")
def hello():
    logger.info('Access to landing pge')
    """
    Landing page for Sam's cricking rating system
    """
    return('Hello this is the landing page for Sams cricking rating system')

@app.route('/retrieve_rating', methods=['POST'])
def retrieve_rating():
    logger.info('Access to player rating')
    json_data = request.get_json()
    response = rr.retrieve_rating(json_data)
    return response

serve(app, port=5050, host='0.0.0.0')
# %%
