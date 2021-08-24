from oct import create_app

# Create the Flask app
application = app = create_app()

if __name__ == "__main__":
    app.run(debug=True)
