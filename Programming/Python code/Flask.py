# FLASK (PYTHON Web Framework)




# Step 0 Create App folder-tree
"""
FlaskApp
       \_ app.py
       \_ templates
                  \_ homepage.html
                  \_ page1.html
                  \_ page2.html
       \_ static
               \_ function1.css
               \_ js
                   \_ jQuery.js

"""

# Step 1 install Flask
$ pip install Flask





# ---------------------- in Python (app.py) --------------------------- #
from flask import Flask # main app
from flask import render_template # render html, css, js
from flask import request # get value from html UI into Python
from flask import json # use json in flask
app = Flask(__name__) # create an app using Flask


# > In Python process

...




# > UI render side method
# > Now define the basic route / and its corresponding request handler:
@app.route("/") # define the page route ex. "/" is homepage
def main():
	# return "Welcome!" # -- if not html, any result in Python
    return render_template('homepage.html')

@app.route('/page1')
def page1():
    return render_template('page1.html')

@app.route('/page2')
def page2():
    return render_template('page2.html')





# > server side method
@app.route('/page1',methods=['POST']) # using jQuery / request on js
def signUp():
 
    # read the posted values from the UI
    _name = request.form['inputName'] # - user input on page1.html / load into Python
    _email = request.form['inputEmail']
    _password = request.form['inputPassword']
 
    # validate the received values
    if _name and _email and _password: # - use those value to process in python and return back to UI
        return json.dumps({'html':'<span>All fields good !!</span>'}) # need to import json for this action
    else:
        return json.dumps({'html':'<span>Enter the required fields</span>'})






# > Next, check if the executed file is the main program and run the app:
if __name__ == "__main__":
    app.run()
# ------------------------- end of py file ---------------------------- #






# > Save the changes and execute app.py:
$ python app.py

# > point your browser to see the app:
http://localhost:5000/
































