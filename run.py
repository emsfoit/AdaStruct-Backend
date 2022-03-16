from app import app


@app.route('/')
def health():
    return 'Hello, I am the healthy app ;)'

if __name__ == '__main__':
    app.run(port=5000, debug=True, host='0.0.0.0')