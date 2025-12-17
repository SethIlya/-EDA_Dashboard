from app import app
from layout import layout
from callbacks import register_callbacks

app.layout = layout

register_callbacks(app)

if __name__ == '__main__':
    app.run(debug=True)