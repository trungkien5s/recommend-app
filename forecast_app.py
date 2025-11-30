# forecast_app.py
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
# --- THAY ĐỔI MỚI: Import make_subplots ---
from plotly.subplots import make_subplots 
import plotly.graph_objects as go
import pandas as pd
import requests

from app import app

# ***************************************************************
API_KEY = "cae7dc5a4513108f90858b008b334a43" 
# ***************************************************************

# --- Tải Dữ liệu (chỉ để lấy danh sách thành phố) ---
try:
    file_path_weather = r'C:\Users\admin\Downloads\europe_weather_2019_2025_sample_extended (3).csv'
    df_weather = pd.read_csv(file_path_weather)
    ALL_CITIES = sorted(df_weather['City'].unique())
except FileNotFoundError:
    ALL_CITIES = ["Berlin", "Paris", "London"]

# --- THAY ĐỔI MỚI: Tạo hàm và bảng màu cho AQI ---
aqi_color_map = {
    1: 'blue',
    2: 'green',
    3: 'orange',
    4: 'red',
    5: 'purple',
    'N/A': 'grey'
}

def get_aqi_description(aqi_value):
    if aqi_value == 1: return "Tốt"
    elif aqi_value == 2: return "Trung bình"
    elif aqi_value == 3: return "Kém"
    elif aqi_value == 4: return "Xấu"
    elif aqi_value == 5: return "Rất xấu"
    return "N/A"

# --- Bố cục (Layout) ---
layout = html.Div(className='main-container', children=[
    html.Div(className='control-panel', children=[
        html.H3("Bảng điều khiển - Dự báo"),
        dcc.Dropdown(
            id='forecast-city-dropdown',
            options=[{'label': city, 'value': city} for city in ALL_CITIES],
            value=ALL_CITIES[0]
        ),
        html.P("Dữ liệu được cung cấp bởi OpenWeatherMap.", 
               style={'fontSize': '0.9em', 'color': '#777', 'marginTop': '15px'}),
        
        # Bảng chú giải AQI (Giữ nguyên)
        html.H5("Chú giải Chất lượng Không khí (AQI):", style={'marginTop': '20px'}),
        html.Table([
            html.Tr([html.Td("1: Tốt"), html.Td("🔵", style={'color': 'blue'})]),
            html.Tr([html.Td("2: Trung bình"), html.Td("🟢", style={'color': 'green'})]),
            html.Tr([html.Td("3: Kém"), html.Td("🟡", style={'color': 'orange'})]),
            html.Tr([html.Td("4. Xấu"), html.Td("🔴", style={'color': 'red'})]),
            html.Tr([html.Td("5: Rất xấu"), html.Td("🟣", style={'color': 'purple'})]),
        ], className='aqi-legend-table')
    ]),
    
    html.Div(className='content-panel', children=[
        html.Div(id='forecast-title'),
        dcc.Graph(id='forecast-chart', config={'responsive': True}, style={'height': '700px'}), # Tăng chiều cao
        html.H4("Chi tiết các mốc thời gian:", style={'marginTop': '20px'}),
        html.Div(id='forecast-table-container')
    ])
])

# --- Callbacks ---
@app.callback(
    [Output('forecast-title', 'children'),
     Output('forecast-chart', 'figure'),
     Output('forecast-table-container', 'children')],
    [Input('forecast-city-dropdown', 'value')]
)
def update_forecast(selected_city):
    if not selected_city:
        return html.H3("Vui lòng chọn một thành phố"), go.Figure(layout={'height': 700}), "Không có dữ liệu"

    # --- CUỘC GỌI API 1: LẤY THỜI TIẾT ---
    url_weather = f"https://api.openweathermap.org/data/2.5/forecast?q={selected_city}&appid={API_KEY}&units=metric&lang=vi"
    
    try:
        response_weather = requests.get(url_weather)
        if response_weather.status_code != 200:
            error_msg = response_weather.json().get('message', 'Lỗi không xác định')
            return html.H3(f"Lỗi API Thời tiết: {error_msg}"), go.Figure(layout={'height': 700}), f"Không thể lấy dữ liệu cho {selected_city}."
            
        data_weather = response_weather.json()
        coords = data_weather.get('city', {}).get('coord', {})
        lat, lon = coords.get('lat'), coords.get('lon')

        if lat is None or lon is None:
             return html.H3("Không tìm thấy tọa độ cho thành phố này."), go.Figure(layout={'height': 700}), ""

        # --- CUỘC GỌI API 2: LẤY AQI ---
        url_aqi = f"http://api.openweathermap.org/data/2.5/air_pollution/forecast?lat={lat}&lon={lon}&appid={API_KEY}"
        response_aqi = requests.get(url_aqi)
        
        aqi_list = []
        if response_aqi.status_code == 200:
            aqi_list = response_aqi.json().get('list', [])
        else:
            print(f"Lỗi API AQI: {response_aqi.json().get('message')}")

        # --- 1. Xử lý dữ liệu Thời tiết ---
        weather_list = data_weather.get('list', [])
        if not weather_list:
            return html.H3(f"Không có dữ liệu dự báo cho {selected_city}"), go.Figure(layout={'height': 700}), ""

        processed_weather = []
        for item in weather_list:
            processed_weather.append({
                'Thời gian': pd.to_datetime(item['dt'], unit='s'),
                'Nhiệt độ (°C)': item['main']['temp'],
                'Cảm giác như': item['main']['feels_like'],
                'Độ ẩm (%)': item['main']['humidity'],
                'Mô tả': item['weather'][0]['description'].capitalize(),
                'Lượng mưa (mm)': item.get('rain', {}).get('3h', 0),
                'Lượng tuyết (mm)': item.get('snow', {}).get('3h', 0)
            })
        df_forecast = pd.DataFrame(processed_weather)

        # --- 2. Xử lý dữ liệu AQI ---
        processed_aqi = []
        for item in aqi_list:
            processed_aqi.append({
                'Thời gian': pd.to_datetime(item['dt'], unit='s'),
                'AQI (1-5)': item['main']['aqi']
            })
        
        # --- 3. Ghép 2 DataFrame ---
        if processed_aqi:
            df_aqi = pd.DataFrame(processed_aqi)
            df_forecast = df_forecast.sort_values('Thời gian')
            df_aqi = df_aqi.sort_values('Thời gian')
            df_merged = pd.merge_asof(df_forecast, df_aqi, on='Thời gian', direction='nearest')
        else:
            df_merged = df_forecast
            df_merged['AQI (1-5)'] = 'N/A'
        
        df_merged['AQI (Mô tả)'] = df_merged['AQI (1-5)'].apply(get_aqi_description)
        df_merged['Thời gian (Hiển thị)'] = df_merged['Thời gian'].dt.strftime('%Y-%m-%d %H:%M')
        # --- THAY ĐỔI MỚI: Thêm cột màu ---
        df_merged['AQI Color'] = df_merged['AQI (1-5)'].map(aqi_color_map)


        # --- 4. TẠO BIỂU ĐỒ SUBPLOTS ---
        
        # --- THAY ĐỔI MỚI: Tạo fig với 2 hàng và 2 trục Y ở hàng 1 ---
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True, # Dùng chung trục X
            vertical_spacing=0.1, # Khoảng cách giữa 2 biểu đồ
            row_heights=[0.7, 0.3], # Biểu đồ trên 70%, dưới 30%
            specs=[[{"secondary_y": True}],  # Hàng 1 có 2 trục Y
                   [{"secondary_y": False}]] # Hàng 2 có 1 trục Y
        )

        # --- Biểu đồ con HÀNG 1: Thời tiết ---
        # Nhiệt độ (Trục Y1 - trái)
        fig.add_trace(go.Scatter(
            x=df_merged['Thời gian'], y=df_merged['Nhiệt độ (°C)'],
            name='Nhiệt độ (°C)', mode='lines+markers'
        ), row=1, col=1, secondary_y=False) # secondary_y=False là trục Y1

        # Lượng mưa (Trục Y2 - phải)
        fig.add_trace(go.Bar(
            x=df_merged['Thời gian'], y=df_merged['Lượng mưa (mm)'],
            name='Lượng mưa (mm)', opacity=0.7, marker_color='blue'
        ), row=1, col=1, secondary_y=True) # secondary_y=True là trục Y2

        # Lượng tuyết (Trục Y2 - phải)
        fig.add_trace(go.Bar(
            x=df_merged['Thời gian'], y=df_merged['Lượng tuyết (mm)'],
            name='Lượng tuyết (mm)', opacity=0.7, marker_color='lightblue'
        ), row=1, col=1, secondary_y=True)

        # --- Biểu đồ con HÀNG 2: AQI ---
        # --- THAY ĐỔI MỚI: Thêm biểu đồ AQI ---
        fig.add_trace(go.Bar(
            x=df_merged['Thời gian'],
            y=df_merged['AQI (1-5)'],
            name='AQI (1-5)',
            marker_color=df_merged['AQI Color'] # Tô màu các cột
        ), row=2, col=1) # Thêm vào hàng 2

        # --- 5. Cập nhật Layout chung ---
        fig.update_layout(
            title_text=f"Dự báo 5 ngày cho {selected_city}",
            height=700, # Đặt chiều cao tổng thể
            barmode='stack', # Áp dụng cho cả 2 biểu đồ (Mưa/Tuyết xếp chồng)
            legend_title="Chú giải",
            hovermode="x unified", # Hiển thị hover cho tất cả
            xaxis_showticklabels=True, # Đảm bảo trục X hàng 1 hiện
            xaxis2_showticklabels=True, # Đảm bảo trục X hàng 2 hiện
        )
        
        # Cập nhật tên các trục Y
        fig.update_yaxes(title_text="Nhiệt độ (°C)", row=1, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Lượng mưa/tuyết (mm)", row=1, col=1, secondary_y=True)
        fig.update_yaxes(title_text="AQI (1-5)", range=[0.5, 5.5], row=2, col=1) # Đặt trục Y cho hàng 2
        
        
        # --- 6. Tạo Bảng chi tiết (Cập nhật cột) ---
        table_columns = [
            {'name': 'Thời gian', 'id': 'Thời gian (Hiển thị)'}, 
            {'name': 'Mô tả', 'id': 'Mô tả'},
            {'name': 'Nhiệt độ (°C)', 'id': 'Nhiệt độ (°C)'},
            {'name': 'Cảm giác như', 'id': 'Cảm giác như'},
            {'name': 'Lượng mưa (mm)', 'id': 'Lượng mưa (mm)'},
            {'name': 'Lượng tuyết (mm)', 'id': 'Lượng tuyết (mm)'},
            {'name': 'Độ ẩm (%)', 'id': 'Độ ẩm (%)'},
            {'name': 'AQI (1-5)', 'id': 'AQI (1-5)'},
            {'name': 'AQI (Mô tả)', 'id': 'AQI (Mô tả)'}
        ]
        
        table = dash.dash_table.DataTable(
            data=df_merged.to_dict('records'),
            columns=table_columns,
            page_size=10,
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left', 'minWidth': '100px'},
            style_header={'fontWeight': 'bold'},
        )
        
        title_component = html.H3(f"Dự báo cho: {data_weather['city']['name']}, {data_weather['city']['country']}")
        
        return title_component, fig, table

    except Exception as e:
        import traceback
        traceback.print_exc() # In lỗi chi tiết ra terminal
        return html.H3(f"Đã xảy ra lỗi: {str(e)}"), go.Figure(layout={'height': 700}), "Lỗi xử lý dữ liệu."