from wsgiref import simple_server
from flask import Flaskrequestrender_template
from flask import Response
import os
from flask_cors import CORS cross_origin
from flask_monitoringdashboard as dashboard
import json

os.putenv('LANG''en_US.UTF-8')
os.putenv('LC_ALL''en_US.UTF-8')

#Here we are changing LANG & LC_ALL variable to english encoding

app= Flask(__name__)
dashboard.bind(app)
CORS(app)
