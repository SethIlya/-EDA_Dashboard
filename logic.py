import base64
import io
import pandas as pd
import numpy as np
from scipy import stats

def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename.lower():
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename.lower():
            df = pd.read_excel(io.BytesIO(decoded))
        else:
            return None, "Формат файла не поддерживается. Используйте CSV или Excel."
    except Exception as e:
        return None, str(e)
    return df, None

def clean_dataset(df, column, action):
    df_copy = df.copy()
    if column not in df_copy.columns: return df_copy, "Столбец не найден."
    try:
        if action == 'drop_rows':
            df_copy = df_copy.dropna(subset=[column])
        elif action == 'drop_col':
            df_copy = df_copy.drop(columns=[column])
        elif action == 'fill_mean':
            if pd.api.types.is_numeric_dtype(df_copy[column]):
                df_copy[column] = df_copy[column].fillna(df_copy[column].mean())
        elif action == 'fill_median':
            if pd.api.types.is_numeric_dtype(df_copy[column]):
                df_copy[column] = df_copy[column].fillna(df_copy[column].median())
        elif action == 'fill_mode':
            mode_val = df_copy[column].mode()[0]
            df_copy[column] = df_copy[column].fillna(mode_val)
    except Exception as e:
        return df, f"Ошибка: {str(e)}"
    return df_copy, None

def handle_outliers(df, column, action):
    df_copy = df.copy()
    if column not in df_copy.columns or not pd.api.types.is_numeric_dtype(df_copy[column]):
        return df_copy, "Ошибка столбца."
    try:
        Q1 = df_copy[column].quantile(0.25)
        Q3 = df_copy[column].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        
        if action == 'remove_rows':
            df_copy = df_copy[(df_copy[column] >= lower) & (df_copy[column] <= upper)]
        elif action == 'clip':
            df_copy[column] = df_copy[column].clip(lower=lower, upper=upper)
    except Exception as e:
        return df, str(e)
    return df_copy, None

def get_outlier_columns(df):
    outlier_info = {}
    for col in df.select_dtypes(include=np.number).columns:
        clean = df[col].dropna()
        if len(clean) < 3: continue
        Q1 = clean.quantile(0.25)
        Q3 = clean.quantile(0.75)
        IQR = Q3 - Q1
        cnt = clean[(clean < (Q1 - 1.5 * IQR)) | (clean > (Q3 + 1.5 * IQR))].count()
        if cnt > 0: outlier_info[col] = cnt
    return outlier_info

def generate_insights(df):
    insights = []
    miss = df.isnull().mean() * 100
    if not miss[miss > 50].empty:
        insights.append(f"КРИТИЧНО: Столбцы {list(miss[miss>50].index)} имеют >50% пропусков. Удалите их.")
    
    num_df = df.select_dtypes(include=np.number)
    if num_df.shape[1] > 1:
        corr = num_df.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        high_corr = [c for c in upper.columns if any(upper[c] > 0.9)]
        if high_corr:
            insights.append(f"ВНИМАНИЕ: Найдена сильная корреляция (>0.9) в столбцах: {high_corr}. Это мультиколлинеарность.")
            
    for col in df.columns:
        if df[col].nunique() == 1:
            insights.append(f"ИНФО: Столбец '{col}' имеет 1 значение. Удалите его.")
            
    if not insights: insights.append("ОК: Явных структурных проблем не обнаружено.")
    return insights

def check_normality_and_outliers(df):
    results = []
    for col in df.select_dtypes(include=np.number).columns:
        clean = df[col].dropna()
        if len(clean) < 3: continue
        
        try:
            stat, p_val = stats.shapiro(clean.sample(min(len(clean), 500)))
            dist_verdict = "Да" if p_val > 0.05 else "Нет"
        except:
            p_val = 0
            dist_verdict = "-"
            
        Q1 = clean.quantile(0.25)
        Q3 = clean.quantile(0.75)
        IQR = Q3 - Q1
        outliers_count = clean[(clean < Q1 - 1.5*IQR) | (clean > Q3 + 1.5*IQR)].count()
        outliers_pct = round((outliers_count / len(clean)) * 100, 2)
        
        results.append({
            'Признак': col,
            'Нормальное распределение': dist_verdict,
            'p-value': round(p_val, 5),
            'Выбросов (кол-во)': outliers_count,
            'Выбросов (%)': outliers_pct
        })
    
    if not results:
        return pd.DataFrame(columns=['Признак', 'Нормальное распределение', 'p-value', 'Выбросов (кол-во)', 'Выбросов (%)'])
    return pd.DataFrame(results)

def get_categorical_stats(df):
    cat_df = df.select_dtypes(include=['object', 'category'])
    if cat_df.empty: return pd.DataFrame()
    return cat_df.describe().reset_index()

def get_group_stats(df, group_col):
    if group_col not in df.columns: return pd.DataFrame()
    grouped = df.groupby(group_col).mean(numeric_only=True).reset_index()
    return grouped

def generate_report_html(df, filename):
    """
    Генерирует HTML-отчет на основе текущего состояния датасета.
    """
    # 1. Сбор данных
    insights = generate_insights(df)
    adv_stats = check_normality_and_outliers(df)
    desc_stats = df.describe().T.reset_index().round(2)
    
    # 2. Формирование HTML
    html_content = f"""
    <html>
    <head>
        <title>EDA Report: {filename}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f4f4f9; }}
            h1, h2, h3 {{ color: #333; }}
            .container {{ max-width: 1000px; margin: auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
            table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #00bc8c; color: white; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            .alert {{ padding: 15px; background-color: #e7f3fe; border-left: 6px solid #2196F3; margin-bottom: 15px; }}
            .footer {{ margin-top: 30px; font-size: 0.8em; color: #777; text-align: center; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Отчет EDA: {filename}</h1>
            <p>Дата формирования: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}</p>
            <p>Размер датасета: {df.shape[0]} строк, {df.shape[1]} столбцов.</p>
            
            <h2>1. Аналитические выводы</h2>
            {''.join([f'<div class="alert">{i}</div>' for i in insights])}
            
            <h2>2. Основная статистика (Числовые данные)</h2>
            {desc_stats.to_html(index=False, classes='table table-striped', border=0)}
            
            <h2>3. Продвинутый анализ (Нормальность и Выбросы)</h2>
            {adv_stats.to_html(index=False, classes='table table-striped', border=0)}
            
            <div class="footer">Сгенерировано модулем автоматизации EDA</div>
        </div>
    </body>
    </html>
    """
    return html_content