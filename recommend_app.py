# recommend_app.py - Gợi ý Điểm đến Dựa trên Cost + Weather + Review
# Version 2.0 - Improved with better scoring and visualization

import pandas as pd
import numpy as np
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app import app

# -----------------------------
# 1. LOAD & TIỀN XỬ LÝ DỮ LIỆU
# -----------------------------

# 1.1. Cost of Living
file_id = "1q5qP66oBoxCEXtGrvE5jtMT5aTATfpn6"
file_path_cost = f"https://drive.google.com/uc?export=download&id={file_id}"
print(file_path_cost)


try:
    df_cost = pd.read_csv(file_path_cost)
    df_cost = df_cost.dropna(subset=['Country'])
    cost_indices_columns = [
        'Cost of Living Index',
        'Rent Index',
        'Cost of Living Plus Rent Index',
        'Groceries Index',
        'Restaurant Price Index',
    ]
    df_cost['Average Cost Index'] = df_cost[cost_indices_columns].mean(axis=1)
    df_cost_cost = df_cost[['Country', 'Average Cost Index']].drop_duplicates()
except Exception as e:
    print(f"Lỗi tải dữ liệu cost: {e}")
    df_cost_cost = pd.DataFrame(columns=['Country', 'Average Cost Index'])

# 1.2. Weather (thời tiết theo tháng)
file_id_weather = "12moZNfbEpVNO39HxQXnIPSoM1ItAR-sE"
file_path_weather = f"https://drive.google.com/uc?export=download&id={file_id_weather}"

try:
    df_weather = pd.read_csv(file_path_weather)
    if not df_weather.empty:
        df_weather['Date'] = pd.to_datetime(
            df_weather[['Year', 'Month']].assign(DAY=1)
        )
    else:
        df_weather['Date'] = pd.to_datetime([])
except Exception as e:
    print(f"Lỗi tải dữ liệu thời tiết: {e}")
    df_weather = pd.DataFrame(columns=['Country', 'Date', 'AvgTemp_C', 'Precip_mm'])
    df_weather['Date'] = pd.to_datetime([])

# 1.3. Review Rating
file_id_review = "1tcsQodOIGlroMDDdfYowl1OHJUSHYrRB"
file_path_review = f"https://drive.google.com/uc?export=download&id={file_id_review}"

try:
    df_raw_review = pd.read_csv(file_path_review)
    id_vars = ['User', 'Country']
    category_columns = [c for c in df_raw_review.columns if c not in id_vars]
    df_review_unpivot = df_raw_review.melt(
        id_vars=id_vars,
        value_vars=category_columns,
        var_name="Category_Name",
        value_name="Rating"
    )
    df_review_clean = df_review_unpivot[df_review_unpivot['Rating'] > 0]
    
    # Tính cả average và count
    df_review_country = df_review_clean.groupby('Country').agg({
        'Rating': ['mean', 'count']
    }).reset_index()
    df_review_country.columns = ['Country', 'Review_Avg', 'Review_Count']
except Exception as e:
    print(f"Lỗi tải dữ liệu review: {e}")
    df_review_country = pd.DataFrame(columns=['Country', 'Review_Avg', 'Review_Count'])

# 1.4. Danh sách country giao nhau trong cả 3 nguồn
ALL_COUNTRIES = sorted(
    set(df_cost_cost['Country'])
    & set(df_weather['Country'])
    & set(df_review_country['Country'])
)


# -----------------------------
# 2. HÀM HỖ TRỢ SCORING
# -----------------------------

def min_max_normalize(series):
    """Chuẩn hóa về [0,1]. Nếu constant thì trả về 0.5."""
    if series.empty:
        return series
    s_min = series.min()
    s_max = series.max()
    if s_max == s_min:
        return pd.Series([0.5] * len(series), index=series.index)
    return (series - s_min) / (s_max - s_min)
def normalize_cost_index(cost_series, low=30, high=90):
    """
    Chuẩn hóa cost_index về [0, 1] theo khoảng cố định.
    30 = rẻ, 90 = rất đắt.
    """
    if cost_series.empty:
        return pd.Series([0.5] * len(cost_series), index=cost_series.index)

    s = cost_series.clip(lower=low, upper=high)
    norm = (s - low) / (high - low)  # 0 = rẻ, 1 = đắt
    return 1 - norm                  # 1 = rẻ, 0 = đắt

def get_cost_category(cost_idx):
    """Phân loại chi phí với emoji"""
    if pd.isna(cost_idx):
        return "Không có dữ liệu"
    if cost_idx < 40:
        return "Rất rẻ 💰"
    elif cost_idx < 60:
        return "Rẻ 💰💰"
    elif cost_idx < 80:
        return "Trung bình 💰💰💰"
    else:
        return "Đắt 💰💰💰💰"


def get_weather_description(details):
    """Tạo mô tả thời tiết dễ hiểu"""
    if not details:
        return "Không có dữ liệu"
    
    temp_avg = details.get('temp_avg', 0)
    stability = details.get('stability', 'Trung bình')
    
    if temp_avg < 10:
        temp_desc = "Lạnh ❄️"
    elif temp_avg < 18:
        temp_desc = "Mát 🌤️"
    elif temp_avg < 25:
        temp_desc = "Ấm ☀️"
    else:
        temp_desc = "Nóng 🔥"
    
    return f"{temp_desc} ({temp_avg:.1f}°C) - Độ ổn định: {stability}"


def get_review_quality(avg_rating, count):
    """Đánh giá chất lượng review"""
    if pd.isna(avg_rating):
        return "Không có dữ liệu"
    
    if avg_rating >= 4.5:
        rating_desc = "Xuất sắc ⭐⭐⭐⭐⭐"
    elif avg_rating >= 4.0:
        rating_desc = "Rất tốt ⭐⭐⭐⭐"
    elif avg_rating >= 3.5:
        rating_desc = "Tốt ⭐⭐⭐"
    elif avg_rating >= 3.0:
        rating_desc = "Khá ⭐⭐"
    else:
        rating_desc = "Trung bình ⭐"
    
    confidence = "Cao" if count >= 100 else "Trung bình" if count >= 50 else "Thấp"
    return f"{rating_desc} ({avg_rating:.2f}/5) - Độ tin cậy: {confidence}"


def compute_weather_score_v2(df_weather_filtered, climate_pref):
    """
    Phiên bản cải tiến với stability + seasonal variance
    """
    if df_weather_filtered.empty:
        return pd.DataFrame(columns=['Country', 'Weather_Score', 'Weather_Details'])
    
    # Target temperature theo gu
    target_temps = {'cool': 12, 'warm': 22, 'neutral': 18}
    target_temp = target_temps.get(climate_pref, 18)
    
    # Aggregate by country
    df_agg = df_weather_filtered.groupby('Country').agg({
        'AvgTemp_C': ['mean', 'std', 'min', 'max'],
        'Precip_mm': ['mean', 'sum']
    }).reset_index()
    
    df_agg.columns = ['Country', 'temp_mean', 'temp_std', 'temp_min', 'temp_max', 
                      'precip_mean', 'precip_total']
    
    # 1. Temperature fit (Gaussian)
    temp_diff = (df_agg['temp_mean'] - target_temp).abs()
    sigma = 7  # Độ khoan dung
    temp_fit_score = np.exp(-(temp_diff ** 2) / (2 * sigma ** 2))
    
    # 2. Temperature stability (std thấp = tốt)
    temp_std_norm = min_max_normalize(df_agg['temp_std'].fillna(0))
    temp_stability_score = 1 - temp_std_norm
    
    # 3. Rain score (ít mưa = tốt)
    rain_norm = min_max_normalize(df_agg['precip_mean'].fillna(0))
    rain_score = 1 - rain_norm
    
    # 4. Seasonal variation (low range = tốt)
    temp_range = df_agg['temp_max'] - df_agg['temp_min']
    variation_norm = min_max_normalize(temp_range.fillna(0))
    variation_score = 1 - variation_norm
    
    # Combined score với trọng số
    df_agg['Weather_Score'] = (
        temp_fit_score * 0.4 +        # Fit với preference: 40%
        temp_stability_score * 0.2 +  # Ổn định: 20%
        rain_score * 0.3 +            # Ít mưa: 30%
        variation_score * 0.1         # Ít biến động: 10%
    )
    
    # Thêm details để hiển thị
    df_agg['Weather_Details'] = df_agg.apply(
        lambda row: {
            'temp_avg': row['temp_mean'],
            'temp_range': f"{row['temp_min']:.1f}–{row['temp_max']:.1f}°C",
            'rain_avg': row['precip_mean'],
            'stability': 'Cao' if row['temp_std'] < 3 else 'Trung bình' if row['temp_std'] < 6 else 'Thấp'
        },
        axis=1
    )
    
    return df_agg[['Country', 'Weather_Score', 'Weather_Details']]


# -----------------------------
# 3. LAYOUT
# -----------------------------

layout = html.Div(className='main-container', children=[

    # --- CỘT TRÁI: CONTROL PANEL ---
    html.Div(className='control-panel', children=[
        html.H3("⚙️ Bảng điều khiển", style={'marginBottom': '20px', 'color': '#667eea'}),

        # 1. Khoảng thời gian du lịch
        html.Label("📅 1. Chọn khoảng thời gian đi du lịch:", className="control-label"),
        dcc.DatePickerRange(
            id='rec-date-range',
            min_date_allowed=df_weather['Date'].min() if not df_weather.empty else None,
            max_date_allowed=df_weather['Date'].max() if not df_weather.empty else None,
            start_date=df_weather['Date'].min() if not df_weather.empty else None,
            end_date=df_weather['Date'].max() if not df_weather.empty else None,
            display_format='MM/YYYY',
            style={'width': '100%'}
        ),
        html.Div(id='selected-date-info', style={
            'fontSize': '0.85em', 
            'color': '#6b7280', 
            'marginTop': '8px',
            'fontStyle': 'italic'
        }),

        html.Br(), html.Br(),

        # 2. Gu thời tiết
        html.Label("🌡️ 2. Bạn thích kiểu thời tiết nào?", className="control-label"),
        dcc.RadioItems(
            id='rec-climate-pref',
            options=[
                {'label': '❄️ Mát (10–15°C)', 'value': 'cool'},
                {'label': '🌤️ Ôn hòa (16–20°C)', 'value': 'neutral'},
                {'label': '☀️ Ấm (21–26°C)', 'value': 'warm'},
            ],
            value='neutral',
            labelStyle={'display': 'block', 'marginBottom': '8px'}
        ),

        html.Hr(style={'margin': '20px 0'}),

        html.Label("⚖️ 3. Mức độ quan trọng từng yếu tố (0–10):", className="control-label"),
        html.Div(style={'marginTop': '10px'}, children=[
            html.Div("💰 Chi phí / Giá cả", style={'fontSize': '0.95em', 'fontWeight': '500'}),
            dcc.Slider(id='rec-weight-cost', min=0, max=10, step=1, value=5,
                       marks={0: '0', 5: '5', 10: '10'}),
            html.Div("💡 10 = rất quan trọng, 0 = không quan trọng", 
                     style={'fontSize': '0.75em', 'color': '#9ca3af', 'fontStyle': 'italic', 'marginBottom': '15px'}),
            
            html.Div("🌤️ Thời tiết dễ chịu", style={'fontSize': '0.95em', 'fontWeight': '500'}),
            dcc.Slider(id='rec-weight-weather', min=0, max=10, step=1, value=5,
                       marks={0: '0', 5: '5', 10: '10'}),
            html.Div("💡 Bao gồm: nhiệt độ, mưa, độ ổn định", 
                     style={'fontSize': '0.75em', 'color': '#9ca3af', 'fontStyle': 'italic', 'marginBottom': '15px'}),
            
            html.Div("⭐ Trải nghiệm & đánh giá", style={'fontSize': '0.95em', 'fontWeight': '500'}),
            dcc.Slider(id='rec-weight-review', min=0, max=10, step=1, value=5,
                       marks={0: '0', 5: '5', 10: '10'}),
            html.Div("💡 Dựa trên đánh giá của người dùng", 
                     style={'fontSize': '0.75em', 'color': '#9ca3af', 'fontStyle': 'italic'}),
        ]),

        html.Hr(style={'margin': '20px 0'}),

        html.Label("🔢 4. Số lượng gợi ý hiển thị:", className="control-label"),
        dcc.Slider(
            id='rec-top-k',
            min=3, max=10, step=1, value=5,
            marks={3: '3', 5: '5', 8: '8', 10: '10'}
        ),

        html.Div([
            html.Div("ℹ️ Cách tính điểm:", style={'fontWeight': '600', 'marginTop': '20px', 'fontSize': '0.9em'}),
            html.Div(
                "Hệ thống kết hợp Cost + Weather + Review theo trọng số bạn chọn, chuẩn hóa về thang 0-1, sau đó xếp hạng các điểm đến.",
                style={'fontSize': '0.8em', 'color': '#6b7280', 'marginTop': '5px', 'lineHeight': '1.5'}
            )
        ])
    ]),

    # --- CỘT PHẢI: CONTENT PANEL ---
    html.Div(className='content-panel', children=[
        html.H2("🎯 Gợi ý Điểm đến Tốt nhất", 
                style={'textAlign': 'center', 'marginBottom': '25px'}),

        # Chart chính
        dcc.Graph(id='rec-recommend-chart', config={'responsive': True},
                  style={'height': '400px', 'marginBottom': '30px'}),
        
        # Breakdown chart
        html.H4("📊 Phân tích Chi tiết Điểm số", 
                style={'marginTop': '30px', 'marginBottom': '15px', 'color': '#374151'}),
        dcc.Graph(id='rec-breakdown-chart', config={'responsive': True},
                  style={'height': '400px', 'marginBottom': '30px'}),

        # Explanation cards
        html.H4("📝 Giải thích Chi tiết", 
                style={'marginTop': '30px', 'marginBottom': '15px', 'color': '#374151'}),
        html.Div(id='rec-explanation-list')
    ])
])


# -----------------------------
# 4. CALLBACK GỢI Ý
# -----------------------------

@app.callback(
    [Output('rec-recommend-chart', 'figure'),
     Output('rec-breakdown-chart', 'figure'),
     Output('rec-explanation-list', 'children'),
     Output('selected-date-info', 'children')],
    [Input('rec-date-range', 'start_date'),
     Input('rec-date-range', 'end_date'),
     Input('rec-climate-pref', 'value'),
     Input('rec-weight-cost', 'value'),
     Input('rec-weight-weather', 'value'),
     Input('rec-weight-review', 'value'),
     Input('rec-top-k', 'value')]
)


def update_recommendations(start_date, end_date,
                           climate_pref,
                           w_cost, w_weather, w_review,
                           top_k):
    # Date info
    if start_date and end_date:
        start_str = pd.to_datetime(start_date).strftime('%m/%Y')
        end_str = pd.to_datetime(end_date).strftime('%m/%Y')
        date_info = f"📌 Đang phân tích: {start_str} → {end_str}"
    else:
        date_info = "⚠️ Vui lòng chọn khoảng thời gian"
    
    # Nếu không có dữ liệu đủ 3 nguồn → báo lỗi
    if len(ALL_COUNTRIES) == 0:
        fig_empty = px.bar(title="❌ Không có đủ dữ liệu để gợi ý")
        return fig_empty, fig_empty, html.P("⚠️ Vui lòng kiểm tra lại các file dữ liệu."), date_info

    # 1. Lọc WEATHER theo khoảng thời gian
    if df_weather.empty:
        df_weather_filtered = df_weather
    else:
        dff_w = df_weather.copy()
        if start_date:
            dff_w = dff_w[dff_w['Date'] >= start_date]
        if end_date:
            dff_w = dff_w[dff_w['Date'] <= end_date]
        df_weather_filtered = dff_w[dff_w['Country'].isin(ALL_COUNTRIES)]

    # 2. Tính weather score v2 (improved)
    df_weather_score = compute_weather_score_v2(df_weather_filtered, climate_pref)

    # 3. Chuẩn bị cost score (cost thấp ⇒ score cao)
    df_cost_sub = df_cost_cost[df_cost_cost['Country'].isin(ALL_COUNTRIES)].copy()
    if df_cost_sub.empty:
        df_cost_sub['Cost_Score'] = 0.5
    else:
        cost_score = normalize_cost_index(df_cost_sub['Average Cost Index'])
        df_cost_sub['Cost_Score'] = cost_score


    # 4. Chuẩn bị review score (với confidence weighting)
    df_review_sub = df_review_country[df_review_country['Country'].isin(ALL_COUNTRIES)].copy()
    if df_review_sub.empty:
        df_review_sub['Review_Score'] = 0.5
    else:
        # Normalize rating
        rating_norm = min_max_normalize(df_review_sub['Review_Avg'])
        
        # Confidence factor dựa trên số lượng reviews
        # Nhiều review = tin cậy hơn
        count_norm = min_max_normalize(df_review_sub['Review_Count'])
        confidence_factor = 0.7 + (0.3 * count_norm)  # 0.7 - 1.0
        
        df_review_sub['Review_Score'] = rating_norm * confidence_factor

    # 5. Merge tất cả về cùng 1 bảng
    df_merge = pd.DataFrame({'Country': ALL_COUNTRIES})

    df_merge = df_merge.merge(df_cost_sub[['Country', 'Average Cost Index', 'Cost_Score']],
                              on='Country', how='left')
    df_merge = df_merge.merge(df_weather_score, on='Country', how='left')
    df_merge = df_merge.merge(df_review_sub[['Country', 'Review_Avg', 'Review_Count', 'Review_Score']],
                              on='Country', how='left')

    # Fill NaN score bằng 0.5 (trung tính) nếu thiếu
    for col in ['Cost_Score', 'Weather_Score', 'Review_Score']:
        if col not in df_merge.columns:
            df_merge[col] = 0.5
        else:
            df_merge[col] = df_merge[col].fillna(0.5)

    # 6. Tính tổng điểm theo trọng số
    weight_sum = (w_cost + w_weather + w_review)
    if weight_sum == 0:
        w_cost = w_weather = w_review = 1
        weight_sum = 3

    df_merge['Total_Score'] = (
        df_merge['Cost_Score'] * w_cost +
        df_merge['Weather_Score'] * w_weather +
        df_merge['Review_Score'] * w_review
    ) / weight_sum

    # 7. Sắp xếp & lấy top_k
    df_top = df_merge.sort_values('Total_Score', ascending=False).head(top_k)

    # 8. Vẽ biểu đồ chính - Total Score
    fig_main = px.bar(
        df_top,
        x='Country',
        y='Total_Score',
        title="🏆 Top Điểm đến Đề xuất",
        labels={'Country': 'Quốc gia', 'Total_Score': 'Điểm Tổng hợp'},
        text=df_top['Total_Score'].round(3),
        color='Total_Score',
        color_continuous_scale='Viridis'
    )
    fig_main.update_traces(textposition='outside', textfont_size=12)
    fig_main.update_layout(
        template='plotly_white',
        yaxis=dict(range=[0, 1.05]),
        showlegend=False,
        title_font_size=18,
        title_x=0.5
    )

    # 9. Vẽ breakdown chart - Grouped Bar với text labels
    # Tạo text cho từng bar - thêm * nếu là giá trị default (thiếu data)
    cost_text = []
    weather_text = []
    review_text = []
    
    for _, row in df_top.iterrows():
        country = row['Country']
        
        # Cost text
        if pd.isna(row.get('Average Cost Index')):
            cost_text.append(f"{row['Cost_Score']:.2f}*")
        else:
            cost_text.append(f"{row['Cost_Score']:.2f}")
        
        # Weather text - luôn có dữ liệu
        weather_text.append(f"{row['Weather_Score']:.2f}")
        
        # Review text
        if pd.isna(row.get('Review_Avg')):
            review_text.append(f"{row['Review_Score']:.2f}*")
        else:
            review_text.append(f"{row['Review_Score']:.2f}")
    
    fig_breakdown = go.Figure()
    
    fig_breakdown.add_trace(go.Bar(
        name='💰 Chi phí',
        x=df_top['Country'],
        y=df_top['Cost_Score'],
        marker_color='#f5576c',
        text=cost_text,
        textposition='outside',
        textfont=dict(size=10)
    ))
    
    fig_breakdown.add_trace(go.Bar(
        name='🌤️ Thời tiết',
        x=df_top['Country'],
        y=df_top['Weather_Score'],
        marker_color='#4facfe',
        text=weather_text,
        textposition='outside',
        textfont=dict(size=10)
    ))
    
    fig_breakdown.add_trace(go.Bar(
        name='⭐ Đánh giá',
        x=df_top['Country'],
        y=df_top['Review_Score'],
        marker_color='#f093fb',
        text=review_text,
        textposition='outside',
        textfont=dict(size=10)
    ))
    
    fig_breakdown.update_layout(
        title='📊 So sánh Điểm số Từng Tiêu chí (* = dữ liệu mặc định)',
        barmode='group',
        yaxis_title='Điểm số (0-1)',
        xaxis_title='Quốc gia',
        template='plotly_white',
        yaxis=dict(range=[0, 1.05]),  # Ensure y-axis shows full range
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        title_font_size=16,
        title_x=0.5,
        annotations=[
            dict(
                text="* Giá trị 0.5 là điểm mặc định khi thiếu dữ liệu",
                xref="paper",
                yref="paper",
                x=0.5,
                y=-0.15,
                showarrow=False,
                font=dict(size=10, color="#6b7280"),
                xanchor="center"
            )
        ]
    )

    # 10. Tạo explanation cards
    explanation_children = []
    for idx, row in df_top.iterrows():
        country = row['Country']
        cost_idx = row.get('Average Cost Index', np.nan)
        weather_details = row.get('Weather_Details', {})
        review_avg = row.get('Review_Avg', np.nan)
        review_count = row.get('Review_Count', 0)
        
        # Tạo card với styling đẹp
        card = html.Div(style={
            'backgroundColor': '#f9fafb',
            'padding': '20px',
            'borderRadius': '12px',
            'marginBottom': '15px',
            'border': '2px solid #e5e7eb',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.05)'
        }, children=[
            html.H4(f"🌍 {country}", style={
                'color': '#667eea',
                'marginBottom': '12px',
                'fontSize': '1.3em'
            }),
            html.Div(style={'marginLeft': '10px'}, children=[
                html.P([
                    html.Strong("💰 Chi phí: "),
                    html.Span(get_cost_category(cost_idx)),
                    html.Span(f" (Index: {cost_idx:.1f})" if not pd.isna(cost_idx) else " ⚠️ Dữ liệu thiếu - dùng mặc định",
                             style={'fontSize': '0.9em', 'color': '#f59e0b' if pd.isna(cost_idx) else '#6b7280'})
                ], style={'marginBottom': '8px'}),
                
                html.P([
                    html.Strong("🌤️ Thời tiết: "),
                    html.Span(get_weather_description(weather_details))
                ], style={'marginBottom': '8px'}),
                
                html.P([
                    html.Strong("⭐ Đánh giá: "),
                    html.Span(get_review_quality(review_avg, review_count)),
                    html.Span(" ⚠️ Dữ liệu thiếu - dùng mặc định" if pd.isna(review_avg) else "",
                             style={'fontSize': '0.9em', 'color': '#f59e0b'})
                ], style={'marginBottom': '8px'}),
                
                html.Div(style={
                    'marginTop': '12px',
                    'padding': '10px',
                    'backgroundColor': '#fff',
                    'borderRadius': '8px',
                    'border': '1px solid #e5e7eb'
                }, children=[
                    html.Strong("📈 Điểm chi tiết:", style={'color': '#374151'}),
                    html.Div(style={'marginTop': '8px', 'fontSize': '0.9em'}, children=[
                        html.Span([
                            "💰 ",
                            f"{row['Cost_Score']:.3f}",
                            "*" if pd.isna(cost_idx) else "",
                            "  "
                        ], style={'marginRight': '15px'}),
                        html.Span(f"🌤️ {row['Weather_Score']:.3f}  ", style={'marginRight': '15px'}),
                        html.Span([
                            "⭐ ",
                            f"{row['Review_Score']:.3f}",
                            "*" if pd.isna(review_avg) else "",
                            "  "
                        ], style={'marginRight': '15px'}),
                        html.Strong(f"→ Tổng: {row['Total_Score']:.3f}", 
                                  style={'color': '#667eea', 'fontSize': '1.05em'}),
                        html.Br() if (pd.isna(cost_idx) or pd.isna(review_avg)) else "",
                        html.Span("* = Giá trị mặc định (0.5)" if (pd.isna(cost_idx) or pd.isna(review_avg)) else "",
                                 style={'fontSize': '0.8em', 'color': '#9ca3af', 'fontStyle': 'italic'})
                    ])
                ])
            ])
        ])
        
        explanation_children.append(card)

    return fig_main, fig_breakdown, explanation_children, date_info
