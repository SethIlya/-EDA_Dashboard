from dash import dcc, html
import dash_bootstrap_components as dbc

DARK_TABLE_HEADER = {'backgroundColor': 'rgb(40, 40, 40)', 'color': 'white', 'fontWeight': 'bold', 'border': '1px solid #555'}
DARK_TABLE_CELL = {'backgroundColor': 'rgb(60, 60, 60)', 'color': 'white', 'border': '1px solid #555', 'minWidth': '100px'}

confirm_modal = dbc.Modal([
    dbc.ModalHeader(dbc.ModalTitle("Подтверждение"), close_button=True),
    dbc.ModalBody("Вы изменяете данные. Это действие необратимо в текущей сессии."),
    dbc.ModalFooter([
        dbc.Button("Отмена", id="btn-cancel-action", className="ms-auto", n_clicks=0),
        dbc.Button("Да, выполнить", id="btn-confirm-action", color="danger", className="ms-2", n_clicks=0),
    ]),
], id="modal-confirm", is_open=False, centered=True)

layout = dbc.Container([
    dcc.Store(id='stored-data'), dcc.Store(id='filename-store'), dcc.Store(id='action-type-store'),
    
    # Компоненты для скачивания
    dcc.Download(id="download-dataframe-csv"),
    dcc.Download(id="download-report-html"),
    
    confirm_modal,

    dbc.Row([dbc.Col(html.H2("EDA: Разведочный анализ данных", className="text-center my-4 text-white"), width=12)]),

    dbc.Row([dbc.Col([
        dcc.Upload(id='upload-data', children=html.Div(['Перетащите файл или ', html.A('Выберите', style={'color': '#00bc8c'})]),
            style={'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px', 'textAlign': 'center', 
                   'backgroundColor': '#2b2b2b', 'color': '#ccc', 'height': '60px', 'lineHeight': '60px'}),
        html.Div(id='upload-status', className="text-center mb-3")
    ], width=8, className="offset-2")]),

    # Панель экспорта (НОВАЯ)
    dbc.Row([
        dbc.Col([
            dbc.ButtonGroup([
                dbc.Button("Скачать CSV (Обработанный)", id="btn-download-csv", color="success", className="me-2"),
                dbc.Button("Скачать HTML-отчет", id="btn-download-html", color="info")
            ], className="w-100")
        ], width=4, className="offset-4 mb-3")
    ]),

    # Панель Пропусков
    dbc.Collapse(dbc.Card([
        dbc.CardHeader("1. Обработка пропусков", className="text-white bg-primary"),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([html.Label("Столбец:"), dcc.Dropdown(id='clean-col-dropdown', style={'color': 'black'})], width=4),
                dbc.Col([html.Label("Метод:"), dcc.Dropdown(id='clean-action-dropdown', 
                    options=[{'label': 'Удалить строки', 'value': 'drop_rows'}, {'label': 'Удалить столбец', 'value': 'drop_col'},
                             {'label': 'Заполнить средним', 'value': 'fill_mean'}, {'label': 'Заполнить медианой', 'value': 'fill_median'},
                             {'label': 'Заполнить модой', 'value': 'fill_mode'}], style={'color': 'black'})], width=4),
                dbc.Col([html.Label("Действие:"), html.Br(), dbc.Button("Применить", id='btn-open-modal-missing', color="info", className="w-100")], width=4)
            ])
        ])
    ], color="dark", outline=True, className="mb-3"), id="cleaning-panel", is_open=False),

    # Панель Выбросов
    dbc.Collapse(dbc.Card([
        dbc.CardHeader("2. Устранение выбросов (IQR)", className="text-white bg-warning"),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([html.Label("Столбец (с выбросами):"), dcc.Dropdown(id='outlier-col-dropdown', style={'color': 'black'})], width=4),
                dbc.Col([html.Label("Метод:"), dcc.Dropdown(id='outlier-action-dropdown', 
                    options=[{'label': 'Удалить строки', 'value': 'remove_rows'}, {'label': 'Ограничить (Clip)', 'value': 'clip'}], 
                    style={'color': 'black'})], width=4),
                dbc.Col([html.Label("Действие:"), html.Br(), dbc.Button("Устранить", id='btn-open-modal-outliers', color="warning", className="w-100")], width=4)
            ])
        ])
    ], color="dark", outline=True, className="mb-4"), id="outliers-panel", is_open=False),

    html.Div(id='global-status-msg', className="text-center text-info font-italic mb-2"),
    html.Hr(className="bg-secondary"),
    dcc.Loading(id="loading-content", type="default", color="#00bc8c", children=html.Div(id='output-data-upload'))

], fluid=True, style={'minHeight': '100vh', 'backgroundColor': '#1e1e1e', 'paddingBottom': '50px'})