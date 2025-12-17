from dash import dcc, html, dash_table, Input, Output, State, ctx, no_update
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import numpy as np
import traceback 

from app import app
from logic import parse_contents, generate_insights, check_normality_and_outliers, clean_dataset, handle_outliers, get_outlier_columns, get_categorical_stats, get_group_stats, generate_report_html
from layout import DARK_TABLE_HEADER, DARK_TABLE_CELL

def register_callbacks(app):
    
    # 1. Модальное окно (Открытие/Закрытие)
    @app.callback(
        [Output("modal-confirm", "is_open"), Output("action-type-store", "data")],
        [Input("btn-open-modal-missing", "n_clicks"), Input("btn-open-modal-outliers", "n_clicks"),
         Input("btn-confirm-action", "n_clicks"), Input("btn-cancel-action", "n_clicks")],
        [State("modal-confirm", "is_open"), State("clean-col-dropdown", "value"), State("clean-action-dropdown", "value"),
         State("outlier-col-dropdown", "value"), State("outlier-action-dropdown", "value")]
    )
    def toggle_modal(n_m, n_o, n_c, n_cancel, is_open, m_col, m_act, o_col, o_act):
        trigger = ctx.triggered_id
        if trigger == "btn-open-modal-missing" and m_col and m_act: return True, "missing"
        if trigger == "btn-open-modal-outliers" and o_col and o_act: return True, "outliers"
        if trigger in ["btn-confirm-action", "btn-cancel-action"]: return False, no_update
        return is_open, no_update

    # 2. Менеджер данных (Загрузка, Очистка, Выбросы)
    @app.callback(
        [Output('stored-data', 'data'), Output('filename-store', 'data'), Output('upload-status', 'children'),
         Output('global-status-msg', 'children'), Output('cleaning-panel', 'is_open'), Output('clean-col-dropdown', 'options'),
         Output('outliers-panel', 'is_open'), Output('outlier-col-dropdown', 'options')],
        [Input('upload-data', 'contents'), Input('btn-confirm-action', 'n_clicks')],
        [State('upload-data', 'filename'), State('stored-data', 'data'), State('action-type-store', 'data'),
         State('clean-col-dropdown', 'value'), State('clean-action-dropdown', 'value'),
         State('outlier-col-dropdown', 'value'), State('outlier-action-dropdown', 'value')]
    )
    def manage_data(contents, n_conf, filename, current_data, action_type, m_col, m_act, o_col, o_act):
        trigger = ctx.triggered_id
        
        def get_opts(df):
            miss = [{'label': f"{c} ({df[c].isna().sum()})", 'value': c} for c in df.columns[df.isna().any()]]
            out = [{'label': f"{c} ({cnt})", 'value': c} for c, cnt in get_outlier_columns(df).items()]
            return miss, out

        try:
            if trigger == 'upload-data':
                if not contents: return tuple([no_update]*8)
                df, err = parse_contents(contents, filename)
                if err: return None, None, dbc.Alert(err, color="danger"), "", False, [], False, []
                m_opts, o_opts = get_opts(df)
                return df.to_json(orient='split', date_format='iso'), filename, dbc.Alert("Загружено", color="success"), "", True, m_opts, True, o_opts

            if trigger == 'btn-confirm-action' and current_data:
                df = pd.read_json(current_data, orient='split')
                if action_type == "missing":
                    df, err = clean_dataset(df, m_col, m_act)
                    msg = f"Пропуски в {m_col} обработаны."
                elif action_type == "outliers":
                    df, err = handle_outliers(df, o_col, o_act)
                    msg = f"Выбросы в {o_col} устранены."
                else: return tuple([no_update]*8)
                
                if err: return no_update, no_update, no_update, dbc.Alert(err, color="danger"), no_update, no_update, no_update, no_update
                m_opts, o_opts = get_opts(df)
                return df.to_json(orient='split', date_format='iso'), no_update, no_update, dbc.Alert(msg, color="success"), True, m_opts, True, o_opts
        except Exception:
            traceback.print_exc()
        return tuple([no_update]*8)

    # 3. Основной Layout (Вкладки)
    @app.callback(
        Output('output-data-upload', 'children'),
        [Input('stored-data', 'data'), Input('filename-store', 'data')]
    )
    def render_layout(data, filename):
        if not data: return html.Div(html.H4("Жду файл...", className="text-center text-muted mt-5"))
        df = pd.read_json(data, orient='split')
        return html.Div([
            html.H5(f"Файл: {filename} | Строк: {df.shape[0]} | Столбцов: {df.shape[1]}", className="text-center text-light mb-3"),
            dbc.Tabs([
                dbc.Tab(label="1. Обзор", tab_id="tab-overview"),
                dbc.Tab(label="2. Статистика", tab_id="tab-stats"),
                dbc.Tab(label="3. Визуализация", tab_id="tab-visuals"),
                dbc.Tab(label="4. Продвинутый анализ", tab_id="tab-insights"),
            ], id="tabs", active_tab="tab-overview", className="mb-3"),
            html.Div(id='tabs-content')
        ])

    # 4. Содержимое вкладок
    @app.callback(
        Output('tabs-content', 'children'),
        [Input('tabs', 'active_tab'), Input('stored-data', 'data')]
    )
    def render_content(tab, data):
        if not data: return html.Div()
        df = pd.read_json(data, orient='split')
        
        if tab == "tab-overview":
            dtypes = df.dtypes.astype(str).reset_index()
            dtypes.columns = ['Признак', 'Тип']
            dtypes['Уник.'] = [df[c].nunique() for c in df.columns]
            dtypes['Пропуски'] = [df[c].isna().sum() for c in df.columns]
            
            miss_fig = px.bar(x=df.columns, y=df.isna().sum(), title="Пропуски по столбцам", template='plotly_dark')
            
            return dbc.Row([
                dbc.Col([
                    html.H5("Типы данных", className="text-white"),
                    dash_table.DataTable(data=dtypes.to_dict('records'), columns=[{'name': i, 'id': i} for i in dtypes.columns],
                        style_cell=DARK_TABLE_CELL, style_header=DARK_TABLE_HEADER, page_size=10),
                    html.H5("Пример данных (Head)", className="text-white mt-3"),
                    dash_table.DataTable(data=df.head().to_dict('records'), columns=[{'name': i, 'id': i} for i in df.columns],
                        style_table={'overflowX': 'auto'}, style_cell=DARK_TABLE_CELL, style_header=DARK_TABLE_HEADER)
                ], width=6),
                dbc.Col([dcc.Graph(figure=miss_fig)], width=6)
            ])

        elif tab == "tab-stats":
            num_stats = df.describe().reset_index()
            cat_stats = get_categorical_stats(df)
            cat_cols = df.select_dtypes(include=['object', 'category']).columns
            
            return dbc.Row([
                dbc.Col([
                    html.H5("Числовая статистика", className="text-info"),
                    dash_table.DataTable(data=num_stats.to_dict('records'), columns=[{'name': i, 'id': i} for i in num_stats.columns],
                        style_table={'overflowX': 'auto'}, style_cell=DARK_TABLE_CELL, style_header=DARK_TABLE_HEADER, page_size=10),
                    
                    html.H5("Категориальная статистика", className="text-info mt-4"),
                    dash_table.DataTable(data=cat_stats.to_dict('records'), columns=[{'name': i, 'id': i} for i in cat_stats.columns],
                        style_table={'overflowX': 'auto'}, style_cell=DARK_TABLE_CELL, style_header=DARK_TABLE_HEADER) if not cat_stats.empty else html.P("Нет категориальных данных"),
                ], width=12),
                
                dbc.Col([
                    html.Hr(className="bg-secondary"),
                    html.H4("Группировка (Сводные таблицы)", className="text-warning"),
                    html.Label("Выберите категорию для группировки:", className="text-white"),
                    dcc.Dropdown(id='groupby-col', options=[{'label': c, 'value': c} for c in cat_cols], style={'color': 'black'}),
                    html.Br(),
                    html.Div(id='groupby-output')
                ], width=12, className="mt-3")
            ])

        elif tab == "tab-visuals":
            all_cols = df.columns
            num_cols = df.select_dtypes(include=np.number).columns
            
            return dbc.Row([
                dbc.Col([
                    dbc.Card([dbc.CardBody([
                        html.Label("Тип графика:", className="text-white"),
                        dcc.Dropdown(id='plot-type', options=[
                            {'label': 'Гистограмма (Распределение/Частоты)', 'value': 'hist'},
                            {'label': 'Boxplot (Сравнение категорий)', 'value': 'box'},
                            {'label': 'Scatter (Взаимосвязь)', 'value': 'scatter'},
                            {'label': 'Корреляция (Heatmap)', 'value': 'corr'},
                            {'label': 'Pie (Доли)', 'value': 'pie'}
                        ], value='hist', clearable=False, style={'color': 'black'}),
                        
                        html.Hr(),
                        html.Div(id='control-x', children=[
                            html.Label("Ось X:", className="text-white"),
                            dcc.Dropdown(id='axis-x', options=[{'label': c, 'value': c} for c in all_cols], value=all_cols[0], style={'color': 'black'})
                        ]),
                        html.Div(id='control-y', children=[
                            html.Label("Ось Y:", className="text-white mt-2"),
                            dcc.Dropdown(id='axis-y', options=[{'label': c, 'value': c} for c in num_cols], style={'color': 'black'})
                        ], style={'display': 'none'}),
                        html.Div(id='control-color', children=[
                            html.Label("Цвет (Группировка):", className="text-white mt-2"),
                            dcc.Dropdown(id='axis-color', options=[{'label': c, 'value': c} for c in all_cols], style={'color': 'black'})
                        ])
                    ])], className="bg-dark h-100")
                ], width=3),
                dbc.Col([dcc.Graph(id='dynamic-graph', style={'height': '70vh'})], width=9)
            ])

        elif tab == "tab-insights":
            insights = generate_insights(df)
            adv_stats = check_normality_and_outliers(df)
            outlier_cols = adv_stats[adv_stats['Выбросов (кол-во)'] > 0]['Признак'].tolist()
            
            return dbc.Row([
                dbc.Col([
                    html.H4("Текстовые выводы", className="text-info"),
                    html.Div([dbc.Alert(i, color="info") for i in insights]),
                    html.H5("Анализ распределений и выбросов", className="text-white mt-4"),
                    dash_table.DataTable(data=adv_stats.to_dict('records'), columns=[{'name': i, 'id': i} for i in adv_stats.columns],
                        style_cell=DARK_TABLE_CELL, style_header=DARK_TABLE_HEADER, sort_action="native")
                ], width=7),
                dbc.Col([
                    dbc.Card([dbc.CardHeader("Просмотр выбросов", className="bg-warning text-dark"), dbc.CardBody([
                        dcc.Dropdown(id='outlier-view-col', options=[{'label': c, 'value': c} for c in outlier_cols], placeholder="Выберите столбец", style={'color': 'black'}),
                        dcc.Graph(id='outlier-view-graph')
                    ])], color="dark", outline=True)
                ], width=5)
            ])

    # 5. Groupby Callback (Сводные таблицы)
    @app.callback(Output('groupby-output', 'children'), [Input('groupby-col', 'value'), State('stored-data', 'data')])
    def update_groupby(col, data):
        if not data or not col: return ""
        df = pd.read_json(data, orient='split')
        res = get_group_stats(df, col)
        if res.empty: return dbc.Alert("Не удалось сгруппировать (возможно нет числовых колонок)", color="warning")
        return dash_table.DataTable(data=res.round(2).to_dict('records'), columns=[{'name': i, 'id': i} for i in res.columns],
            style_table={'overflowX': 'auto'}, style_cell=DARK_TABLE_CELL, style_header=DARK_TABLE_HEADER, page_size=10)

    # 6. Управление видимостью настроек графика
    @app.callback(
        [Output('control-y', 'style'), Output('control-color', 'style')],
        [Input('plot-type', 'value')]
    )
    def update_controls(plot_type):
        show = {'display': 'block'}
        hide = {'display': 'none'}
        if plot_type == 'hist': return hide, show
        if plot_type == 'pie': return hide, hide
        if plot_type == 'corr': return hide, hide
        return show, show

    # 7. Построение графика (С ИСПРАВЛЕНИЕМ ГИСТОГРАММЫ)
    @app.callback(
        Output('dynamic-graph', 'figure'),
        [Input('plot-type', 'value'), Input('axis-x', 'value'), Input('axis-y', 'value'), Input('axis-color', 'value'), State('stored-data', 'data')]
    )
    def update_graph(ptype, x, y, color, data):
        if not data: return {}
        df = pd.read_json(data, orient='split')
        tpl = 'plotly_dark'
        
        try:
            # --- ИСПРАВЛЕНИЕ: Превращаем малое кол-во чисел в строки ---
            if x and ptype in ['hist', 'box']:
                if pd.api.types.is_numeric_dtype(df[x]) and df[x].nunique() < 15:
                    df = df.sort_values(by=x)
                    df[x] = df[x].astype(str)

            if color:
                if pd.api.types.is_numeric_dtype(df[color]) and df[color].nunique() < 15:
                    df = df.sort_values(by=color)
                    df[color] = df[color].astype(str)
            # -----------------------------------------------------------

            if ptype == 'hist':
                if not x: return px.scatter(title="Выберите Ось X")
                return px.histogram(df, x=x, color=color, marginal="box", title=f"Распределение: {x}", template=tpl)
            elif ptype == 'box':
                if not x or not y: return px.box(title="Выберите Ось X (Категория) и Ось Y (Число)")
                return px.box(df, x=x, y=y, color=color, title=f"Сравнение: {y} по группам {x}", template=tpl)
            elif ptype == 'scatter':
                if not x or not y: return px.scatter(title="Выберите Ось X и Ось Y")
                return px.scatter(df, x=x, y=y, color=color, title=f"Корреляция: {x} vs {y}", template=tpl)
            elif ptype == 'pie':
                if not x: return px.pie(title="Выберите категорию")
                cts = df[x].value_counts().head(10).reset_index()
                cts.columns = [x, 'count']
                return px.pie(cts, names=x, values='count', title=f"Топ-10 значений: {x}", template=tpl)
            elif ptype == 'corr':
                return px.imshow(df.select_dtypes(include=np.number).corr(), text_auto=True, color_continuous_scale='RdBu_r', title="Матрица Корреляции", template=tpl)
        except Exception as e:
            return px.scatter(title=f"Ошибка: {e}", template=tpl)
        return {}

    # 8. Превью выброса
    @app.callback(Output('outlier-view-graph', 'figure'), [Input('outlier-view-col', 'value'), State('stored-data', 'data')])
    def view_outlier(col, data):
        if not data or not col: return {}
        df = pd.read_json(data, orient='split')
        return px.box(df, y=col, title=f"Выбросы в {col}", template='plotly_dark', points='outliers')

    # 9. СКАЧИВАНИЕ (CSV + HTML)
    @app.callback(
        Output("download-dataframe-csv", "data"),
        [Input("btn-download-csv", "n_clicks")],
        [State("stored-data", "data"), State("filename-store", "data")]
    )
    def download_csv(n_clicks, data, filename):
        if not n_clicks or not data: return no_update
        df = pd.read_json(data, orient='split')
        name = "processed_" + filename if filename else "processed_data.csv"
        return dcc.send_data_frame(df.to_csv, name, index=False)

    @app.callback(
        Output("download-report-html", "data"),
        [Input("btn-download-html", "n_clicks")],
        [State("stored-data", "data"), State("filename-store", "data")]
    )
    def download_html(n_clicks, data, filename):
        if not n_clicks or not data: return no_update
        df = pd.read_json(data, orient='split')
        name = "report_" + (filename.split('.')[0] if filename else "data") + ".html"
        html_str = generate_report_html(df, filename or "Dataset")
        return dict(content=html_str, filename=name)