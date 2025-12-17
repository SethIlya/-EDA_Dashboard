import dash
import dash_bootstrap_components as dbc

# Инициализация приложения с темной темой (Bootstrap Darkly)
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY], suppress_callback_exceptions=True)
app.title = "EDA System"