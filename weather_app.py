# weather_app.py
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd

# Import 'app' từ file app.py
from app import app

# --- 1. Tải và Xử lý Dữ liệu Thời tiết ---
file_path_weather = r'D:\Trực quan hóa\Trực quan cuối kỳ\demo_final\europe_weather_2019_2025_sample_extended.csv'

try:
    df = pd.read_csv(file_path_weather)
except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy tệp tại đường dẫn: {file_path_weather}")
    df = pd.DataFrame() # Tạo dataframe rỗng để tránh lỗi

if not df.empty:
    df['Date'] = pd.to_datetime(df[['Year', 'Month']].assign(DAY=1))
    df = df.sort_values('Date')
    ALL_COUNTRIES = df['Country'].unique()
else:
    ALL_COUNTRIES = []

METRIC_OPTIONS = {
    'AvgTemp_C': 'Nhiệt độ trung bình (°C)',
    'Precip_mm': 'Lượng mưa (mm)',
    'Humidity_%': 'Độ ẩm (%)',
    'Snowfall_mm': 'Lượng tuyết rơi (mm)',
    'US_AQI': 'Chỉ số chất lượng không khí (AQI)'
}

# --- 3. Định nghĩa Bố cục (Layout) cho trang này ---
# Lưu ý: đổi tên 'app.layout' thành 'layout'
layout = html.Div(className='main-container', children=[
    
    # Panel điều khiển (Bên trái)
    html.Div(className='control-panel', children=[
        html.H3("Bảng điều khiển - Thời tiết"),
        
        # (Copy y hệt phần control-panel từ app.py cũ)
        html.Label("Chọn chế độ xem:", className="control-label"),
        dcc.RadioItems(id='mode-selector', options=[
            {'label': 'So sánh Thành phố (trong 1 Nước)', 'value': 'cities'},
            {'label': 'So sánh Quốc gia', 'value': 'countries'}
        ], value='cities', labelStyle={'display': 'block', 'marginBottom': '5px'}),
        
        html.Hr(),
        
        html.Div(id='city-controls', children=[
            html.Label("1. Chọn Quốc gia:", className="control-label"),
            dcc.Dropdown(id='country-dropdown', options=[{'label': c, 'value': c} for c in ALL_COUNTRIES], value=ALL_COUNTRIES[0] if len(ALL_COUNTRIES) > 0 else None),
            html.Br(),
            html.Label("2. Chọn (tối đa 3) Thành phố:", className="control-label"),
            dcc.Dropdown(id='city-dropdown', multi=True, value=[]),
        ]),
        
        html.Div(id='country-controls', style={'display': 'none'}, children=[
            html.Label("Chọn các Quốc gia để so sánh:", className="control-label"),
            dcc.Dropdown(id='country-dropdown-multi', options=[{'label': c, 'value': c} for c in ALL_COUNTRIES], multi=True, value=list(ALL_COUNTRIES[:2]) if len(ALL_COUNTRIES) > 1 else []),
        ]),
        
        html.Hr(),
        
        html.Label("3. Chọn Tiêu chí:", className="control-label"),
        dcc.Dropdown(id='metric-dropdown', options=[{'label': v, 'value': k} for k, v in METRIC_OPTIONS.items()], value='AvgTemp_C'),
        
        html.Br(),
        html.Label("4. Chọn Khoảng thời gian:", className="control-label"),
        dcc.DatePickerRange(
            id='date-picker-range',
            min_date_allowed=df['Date'].min() if not df.empty else None,
            max_date_allowed=df['Date'].max() if not df.empty else None,
            start_date=df['Date'].min() if not df.empty else None,
            end_date=df['Date'].max() if not df.empty else None,
            display_format='DD/MM/YYYY',
            style={'width': '100%'}
        ),
        
        html.Br(), html.Br(),
        
        html.Label("5. Chọn Dạng biểu đồ:", className="control-label"),
        dcc.RadioItems(
            id='chart-type-selector',
            options=[
                {'label': 'Biểu đồ đường (Theo thời gian)', 'value': 'line'},
                {'label': 'Biểu đồ cột (Theo thời gian)', 'value': 'bar'}
            ],
            value='line',
            labelStyle={'display': 'block', 'marginBottom': '5px'}
        ),
    ]),
    
    # Panel nội dung (Bên phải)
    html.Div(className='content-panel', children=[
        dcc.Graph(id='main-chart', config={'responsive': True}),
        html.Div(id='stats-output', className='stats-panel')
    ])
])

# --- 4. Callbacks (Copy y hệt toàn bộ các callback của app.py cũ) ---
# Các callback này vẫn dùng @app.callback
@app.callback(
    [Output('city-controls', 'style'),
     Output('country-controls', 'style')],
    [Input('mode-selector', 'value')]
)
def toggle_controls(mode):
    if mode == 'cities':
        return {'display': 'block'}, {'display': 'none'}
    else:
        return {'display': 'none'}, {'display': 'block'}

@app.callback(
    Output('city-dropdown', 'options'),
    [Input('country-dropdown', 'value')]
)
def update_city_options(selected_country):
    if not selected_country or df.empty:
        return []
    cities_in_country = df[df['Country'] == selected_country]['City'].unique()
    return [{'label': c, 'value': c} for c in cities_in_country]

@app.callback(
    [Output('main-chart', 'figure'),
     Output('stats-output', 'children')],
    [Input('mode-selector', 'value'),
     Input('country-dropdown', 'value'),
     Input('city-dropdown', 'value'),
     Input('country-dropdown-multi', 'value'),
     Input('metric-dropdown', 'value'),
     Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date'),
     Input('chart-type-selector', 'value')]   
)
def update_graph_and_stats(mode, city_country, cities, countries, metric, start_date, end_date, chart_type):
    if df.empty:
        return go.Figure(layout={'title': 'Lỗi tải dữ liệu thời tiết'}), html.P("Không thể tải tệp dữ liệu.")

    dff = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    
    fig = go.Figure()
    stats_data = []
    metric_label = METRIC_OPTIONS.get(metric, metric)
    
    if mode == 'cities':
        # (Toàn bộ logic của callback cũ copy vào đây)
        # ... (phần này dài, tôi giả định bạn copy y hệt vào) ...
        if not city_country or not cities:
            fig.update_layout(title="Vui lòng chọn Quốc gia và ít nhất 1 Thành phố")
            return fig, html.P("Chưa chọn dữ liệu.")
        
        selected_cities = cities[:3] 
        city_plot_data = [] 

        for city in selected_cities:
            city_data = dff[(dff['Country'] == city_country) & (dff['City'] == city)]
            if not city_data.empty:
                stats = city_data[metric].describe()
                stats_data.append({'Tên': city, **stats})
                city_plot_data.append({'name': city, 'data': city_data})

        if chart_type == 'line':
            for item in city_plot_data:
                fig.add_trace(go.Scatter(x=item['data']['Date'], y=item['data'][metric], name=item['name'], mode='lines+markers'))
            fig.update_layout(title=f"So sánh {metric_label} tại {city_country}", xaxis_title="Thời gian", hovermode="x unified")
        
        elif chart_type == 'bar':
            for item in city_plot_data:
                fig.add_trace(go.Bar(x=item['data']['Date'], y=item['data'][metric], name=item['name']))
            fig.update_layout(title=f"So sánh {metric_label} tại {city_country}", xaxis_title="Thời gian", barmode='group', hovermode="x")

    elif mode == 'countries':
        # (Toàn bộ logic của callback cũ copy vào đây)
        # ... (phần này dài, tôi giả định bạn copy y hệt vào) ...
        if not countries:
            fig.update_layout(title="Vui lòng chọn ít nhất 1 Quốc gia")
            return fig, html.P("Chưa chọn dữ liệu.")
            
        dff_country = dff[dff['Country'].isin(countries)]
        df_agg_line = dff_country.groupby(['Country', 'Date'])[metric].mean().reset_index()
        
        for country in countries:
            country_data_line = df_agg_line[df_agg_line['Country'] == country]
            if not country_data_line.empty:
                stats = country_data_line[metric].describe()
                stats_data.append({'Tên': country, **stats})

        if chart_type == 'line':
            for country in countries:
                country_data_line = df_agg_line[df_agg_line['Country'] == country]
                if not country_data_line.empty:
                    fig.add_trace(go.Scatter(x=country_data_line['Date'], y=country_data_line[metric], name=country, mode='lines+markers'))
            fig.update_layout(title=f"So sánh {metric_label} (TB) giữa các Quốc gia", xaxis_title="Thời gian", hovermode="x unified")

        elif chart_type == 'bar':
            for country in countries:
                country_data_line = df_agg_line[df_agg_line['Country'] == country]
                if not country_data_line.empty:
                    fig.add_trace(go.Bar(x=country_data_line['Date'], y=country_data_line[metric], name=country))
            fig.update_layout(title=f"So sánh {metric_label} (TB) giữa các Quốc gia", xaxis_title="Thời gian", barmode='group', hovermode="x")

    # Chung
    fig.update_layout(yaxis_title=metric_label, legend_title="Địa điểm", margin=dict(l=50, r=50, t=80, b=50), xaxis=dict(tickformat='%m/%Y'))
    
    if not stats_data:
        return fig, html.P("Không có dữ liệu thống kê.")
    header = [html.Th("Địa điểm"), html.Th("Trung bình"), html.Th("Độ lệch chuẩn"), html.Th("Min"), html.Th("Max")]
    rows = []
    for item in stats_data:
        rows.append(html.Tr([
            html.Td(item['Tên'], className='stats-label'),
            html.Td(f"{item.get('mean', 'N/A'):.2f}"),
            html.Td(f"{item.get('std', 'N/A'):.2f}"),
            html.Td(f"{item.get('min', 'N/A'):.2f}"),
            html.Td(f"{item.get('max', 'N/A'):.2f}"),
        ]))
    stats_table = html.Table([html.Thead(html.Tr(header)), html.Tbody(rows)], className='stats-table')
    
    return fig, stats_table