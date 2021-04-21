from flask import Flask
import tensorflow as tf

app = Flask(__name__)

@app.route('/')
def hello():
    """Return a friendly HTTP greeting."""
    return 'Hello World!update by github'

if __name__ == "__main__":
    app.run(debug=True)