import base64
import io
import datetime

import dash
from dash import dcc, html, dash_table, Input, Output, State, ctx
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import numpy as np
from scipy import stats


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY], suppress_callback_exceptions=True)
app.title = "EDA Dashboard"

DARK_TABLE_HEADER = {
    'backgroundColor': 'rgb(30, 30, 30)',
    'color': 'white',
    'fontWeight': 'bold',
    'border': '1px solid #444'
}
DARK_TABLE_CELL = {
    'backgroundColor': 'rgb(50, 50, 50)',
    'color': 'white',
    'border': '1px solid #444'
}

def generate_insights(df):
    """Генерация текстовых инсайтов (без смайликов)"""
    insights = []
    
    # Пропуски
    missing_pct = df.isnull().mean() * 100
    high_missing = missing_pct[missing_pct > 50]
    if not high_missing.empty:
        cols = ", ".join(high_missing.index)
        insights.append(f"Внимание: В столбцах {cols} пропущено более 50% данных.")
    
    # Корреляции
    numeric_df = df.select_dtypes(include=np.number)
    if numeric_df.shape[1] > 1:
        corr_matrix = numeric_df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr = [column for column in upper.columns if any(upper[column] > 0.85)]
        
        for col in high_corr:
            related = upper.index[upper[col] > 0.85].tolist()
            for r in related:
                insights.append(f"Мультиколлинеарность: Очень сильная корреляция (>0.85) между '{col}' и '{r}'.")
                
    # Константы
    for col in df.columns:
        if df[col].nunique() == 1:
            insights.append(f"Столбец '{col}' содержит только одно уникальное значение.")

    return insights

def check_normality_and_outliers(df):
    """Проверка на нормальность и выбросы"""
    numeric_cols = df.select_dtypes(include=np.number).columns
    results = []
    
    for col in numeric_cols:
        clean_data = df[col].dropna()
        if len(clean_data) < 3: 
            continue
            
        stat, p_value = stats.shapiro(clean_data.sample(min(len(clean_data), 1000)))
        is_normal = p_value > 0.05
        
        Q1 = clean_data.quantile(0.25)
        Q3 = clean_data.quantile(0.75)
        IQR = Q3 - Q1
        outliers_count = ((clean_data < (Q1 - 1.5 * IQR)) | (clean_data > (Q3 + 1.5 * IQR))).sum()
        pct_outliers = round(outliers_count / len(clean_data) * 100, 2)
        
        results.append({
            'Колонка': col,
            'Нормальное распр.?': 'Да' if is_normal else 'Нет',
            'P-value': round(p_value, 4),
            'Выбросов (кол-во)': outliers_count,
            'Выбросов (%)': pct_outliers
        })
    
    if not results:
        return pd.DataFrame(columns=['Колонка', 'Результат'])
    
    return pd.DataFrame(results)

# (LAYOUT) 

app.layout = dbc.Container([
    dcc.Store(id='stored-data', storage_type='memory'),


    dbc.Row([
        dbc.Col(html.H1("Data Detective: EDA Dashboard", className="text-center my-4 text-light"), width=12)
    ]),

    # Блок загрузки 
    dbc.Row([
        dbc.Col([
            dcc.Upload(
                id='upload-data',
                children=html.Div(['Перетащите файл или ', html.A('Выберите (CSV/XLSX)', style={'color': '#00bc8c'})]),
                style={
                    'width': '100%', 'height': '60px', 'lineHeight': '60px',
                    'borderWidth': '2px', 'borderStyle': 'dashed', 'borderRadius': '10px',
                    'textAlign': 'center', 'margin': '10px', 
                    'backgroundColor': '#303030', 'color': 'white' 
                },
                multiple=False
            ),
            html.Div(id='upload-status', className="text-center")
        ], width=8, className="offset-2")
    ]),

    html.Hr(className="bg-light"),

    dcc.Loading(
        id="loading-content",
        type="circle",
        color="#00bc8c",
        children=html.Div(id='output-data-upload')
    )

], fluid=True, style={'minHeight': '100vh', 'backgroundColor': '#222'}) 




def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename.lower():
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename.lower():
            df = pd.read_excel(io.BytesIO(decoded))
        else:
            return None, "Формат не поддерживается"
    except Exception as e:
        return None, str(e)
    return df, None


# CALLBACKS 

@app.callback(
    [Output('stored-data', 'data'),
     Output('upload-status', 'children'),
     Output('output-data-upload', 'children')],
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename')]
)
def update_output(contents, filename):
    if contents is None:
        return None, "", html.Div([
            html.H4("Добро пожаловать", className="text-center text-light"),
            html.P("Загрузите датасет для начала анализа.", className="text-center text-muted")
        ])
    
    df, error = parse_contents(contents, filename)
    if error:
        return None, dbc.Alert(f"Ошибка: {error}", color="danger"), ""
    
    data_json = df.to_json(date_format='iso', orient='split')
    
    # Вкладки
    tabs_layout = dbc.Tabs([
        dbc.Tab(label="Обзор", tab_id="tab-overview"),
        dbc.Tab(label="Статистика", tab_id="tab-stats"),
        dbc.Tab(label="Графики", tab_id="tab-visuals"),
        dbc.Tab(label="Продвинутый анализ", tab_id="tab-advanced"),
    ], id="tabs", active_tab="tab-overview", className="mb-3")
    
    content = html.Div([
        html.H5(f"Файл: {filename} | Строк: {df.shape[0]} | Колонок: {df.shape[1]}", className="text-center mb-3 text-light"),
        tabs_layout,
        html.Div(id='tabs-content')
    ])
    
    return data_json, dbc.Alert("Данные успешно загружены", color="success", duration=2000), content


@app.callback(
    Output('tabs-content', 'children'),
    [Input('tabs', 'active_tab'),
     Input('stored-data', 'data')]
)
def render_tab_content(active_tab, data_json):
    if not data_json:
        return html.Div()
    
    df = pd.read_json(data_json, orient='split')
    
    # ОБЗОР 
    if active_tab == "tab-overview":
        missing = df.isnull().sum().reset_index()
        missing.columns = ['Feature', 'Missing']
        missing = missing[missing['Missing'] > 0]
        
        if not missing.empty:
            fig_missing = px.bar(missing, x='Feature', y='Missing', title="Количество пропусков", template='plotly_dark')
            missing_div = dcc.Graph(figure=fig_missing, style={'height': '500px'})
        else:
            missing_div = dbc.Alert("Пропущенных значений нет.", color="success")
            
        return dbc.Row([
            dbc.Col([
                html.H6("Пример данных (Head)", className="text-light"),
                dash_table.DataTable(
                    data=df.head().to_dict('records'),
                    columns=[{'name': i, 'id': i} for i in df.columns],
                    style_table={'overflowX': 'auto'},
                    style_cell=DARK_TABLE_CELL,    
                    style_header=DARK_TABLE_HEADER 
                )
            ], width=12, className="mb-4"),
            dbc.Col([
                html.H6("Типы данных", className="text-light"),
                html.Pre(df.dtypes.to_string(), className="text-white"),
            ], width=4),
            dbc.Col([
                html.H6("Пропущенные значения", className="text-light"),
                missing_div
            ], width=8)
        ])

    # СТАТИСТИКА 
    elif active_tab == "tab-stats":
        stats = df.describe().reset_index()
        return dbc.Row([
            dbc.Col([
                html.H5("Основные статистические показатели", className="text-light"),
                dash_table.DataTable(
                    data=stats.to_dict('records'),
                    columns=[{'name': i, 'id': i} for i in stats.columns],
                    style_table={'overflowX': 'auto'},
                    style_cell=DARK_TABLE_CELL,
                    style_header=DARK_TABLE_HEADER,
                    page_size=12
                )
            ])
        ])

    # ГРАФИКИ 
    elif active_tab == "tab-visuals":
        cols = df.columns.tolist()
        return dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Label("Тип графика", className="text-light"),
                        dcc.Dropdown(
                            id='plot-type',
                            options=[
                                {'label': 'Гистограмма', 'value': 'hist'},
                                {'label': 'Boxplot', 'value': 'box'},
                                {'label': 'Scatter (Точечный)', 'value': 'scatter'},
                                {'label': 'Heatmap (Корреляция)', 'value': 'corr'},
                                {'label': 'Pie Chart (Круговая)', 'value': 'pie'}
                            ],
                            value='hist',
                            style={'color': 'black'} 
                        ),
                        html.Hr(className="bg-light"),
                        html.Label("Ось X / Категория", className="text-light"),
                        dcc.Dropdown(id='axis-x', options=[{'label': i, 'value': i} for i in cols], value=cols[0], style={'color': 'black'}),
                        html.Label("Ось Y / Значение", className="mt-2 text-light"),
                        dcc.Dropdown(id='axis-y', options=[{'label': i, 'value': i} for i in cols], placeholder="Опционально", style={'color': 'black'}),
                        html.Label("Цвет (Группировка)", className="mt-2 text-light"),
                        dcc.Dropdown(id='axis-color', options=[{'label': i, 'value': i} for i in cols], placeholder="Опционально", style={'color': 'black'}),
                    ])
                ], className="h-100 bg-dark border-secondary") 
            ], width=3),
            dbc.Col([
                dcc.Graph(id='dynamic-graph', style={'height': '70vh', 'min-height': '500px'})
            ], width=9)
        ])

    # ПРОДВИНУТЫЙ АНАЛИЗ
    elif active_tab == "tab-advanced":
        insights = generate_insights(df)
        insights_div = [dbc.Alert(dcc.Markdown(i), color="info", className="mb-2") for i in insights]
        if not insights_div:
            insights_div = [dbc.Alert("Явных аномалий или сильных корреляций не найдено.", color="secondary")]

        adv_stats = check_normality_and_outliers(df)
        
        return dbc.Row([
            dbc.Col([
                html.H5("Автоматические инсайты", className="text-light"),
                html.Div(insights_div),
                html.Hr(className="bg-light"),
                html.H5("Тест на нормальность и выбросы", className="text-light"),
                dash_table.DataTable(
                    data=adv_stats.to_dict('records'),
                    columns=[{'name': i, 'id': i} for i in adv_stats.columns],
                    style_table={'overflowX': 'auto'},
                    style_cell=DARK_TABLE_CELL,
                    style_header=DARK_TABLE_HEADER,
                    sort_action="native"
                )
            ], width=12)
        ])

# Обновление графиков 
@app.callback(
    Output('dynamic-graph', 'figure'),
    [Input('plot-type', 'value'),
     Input('axis-x', 'value'),
     Input('axis-y', 'value'),
     Input('axis-color', 'value'),
     Input('stored-data', 'data')]
)
def update_graph(plot_type, x_col, y_col, color_col, data_json):
    if not data_json or not x_col:
        return {}
    
    df = pd.read_json(data_json, orient='split')
    template = 'plotly_dark'
    
    if plot_type == 'hist':
        return px.histogram(df, x=x_col, color=color_col, marginal="box", title=f"Распределение {x_col}", template=template)
    
    elif plot_type == 'box':
        return px.box(df, x=x_col, y=y_col, color=color_col, title=f"Boxplot {x_col}", template=template)
    
    elif plot_type == 'scatter':
        if not y_col: return px.scatter(title="Выберите Ось Y", template=template)
        return px.scatter(df, x=x_col, y=y_col, color=color_col, title=f"Scatter: {x_col} vs {y_col}", template=template)
    
    elif plot_type == 'pie':
        counts = df[x_col].value_counts().head(10).reset_index()
        counts.columns = [x_col, 'count']
        return px.pie(counts, names=x_col, values='count', title=f"Top-10 категорий: {x_col}", template=template)

    elif plot_type == 'corr':
        numeric_df = df.select_dtypes(include=np.number)
        if numeric_df.shape[1] < 2: return px.scatter(title="Недостаточно числовых данных", template=template)
        return px.imshow(numeric_df.corr(), text_auto=True, color_continuous_scale='RdBu_r', title="Матрица корреляций", template=template)
        
    return {}

if __name__ == '__main__':
    app.run(debug=True)