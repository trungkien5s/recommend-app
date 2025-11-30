# recommend_app.py - Gợi ý Điểm đến Dựa trên Cost + Weather + Review

import pandas as pd
import numpy as np
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px

from app import app

# -----------------------------
# 1. LOAD & TIỀN XỬ LÝ DỮ LIỆU
# -----------------------------

# 1.1. Cost of Living
file_path_cost = r'C:\Users\admin\Downloads\wikipedia_cost_of_living_indices4 (1).csv'
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
except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy tệp chi phí tại: {file_path_cost}")
    df_cost_cost = pd.DataFrame(columns=['Country', 'Average Cost Index'])

# 1.2. Weather (thời tiết theo tháng)
file_path_weather = r'C:\Users\admin\Downloads\europe_weather_2019_2025_sample_extended (3).csv'
try:
    df_weather = pd.read_csv(file_path_weather)
    if not df_weather.empty:
        df_weather['Date'] = pd.to_datetime(
            df_weather[['Year', 'Month']].assign(DAY=1)
        )
    else:
        df_weather['Date'] = pd.to_datetime([])
except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy tệp thời tiết tại: {file_path_weather}")
    df_weather = pd.DataFrame(columns=['Country', 'Date', 'AvgTemp_C', 'Precip_mm'])
    df_weather['Date'] = pd.to_datetime([])

# 1.3. Review Rating
file_path_review = r"C:\Users\admin\Downloads\google_review_ratings_rounded (2).csv"
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
    df_review_country = (
        df_review_clean
        .groupby('Country')['Rating']
        .mean()
        .reset_index()
        .rename(columns={'Rating': 'Review_Avg'})
    )
except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy tệp review tại: {file_path_review}")
    df_review_country = pd.DataFrame(columns=['Country', 'Review_Avg'])

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


def compute_weather_score(df_weather_filtered, climate_pref):
    """
    Tính điểm thời tiết cho từng Country dựa trên:
    - Nhiệt độ trung bình (gần với 'target' theo gu)
    - Lượng mưa (càng ít càng tốt)
    """
    if df_weather_filtered.empty:
        return pd.DataFrame(columns=['Country', 'Weather_Score'])

    # Chọn target nhiệt độ theo gu
    if climate_pref == 'cool':      # mát
        target_temp = 12
    elif climate_pref == 'warm':    # ấm
        target_temp = 22
    else:                           # neutral / ôn hòa
        target_temp = 18

    df_agg = df_weather_filtered.groupby('Country').agg({
        'AvgTemp_C': 'mean',
        'Precip_mm': 'mean'
    }).reset_index()

    # 1) Độ phù hợp nhiệt độ: càng gần target càng tốt
    #   dùng hàm Gaussian đơn giản
    temp_diff = (df_agg['AvgTemp_C'] - target_temp).abs()
    temp_score = np.exp(- (temp_diff ** 2) / (2 * (5 ** 2)))  # sigma ~ 5°C
    # normalize (0-1) đề phòng trường hợp kỳ quặc
    temp_score = min_max_normalize(pd.Series(temp_score, index=df_agg.index))

    # 2) Lượng mưa: ít mưa hơn => điểm cao
    #   Nên đảo chiều Precip_mm
    precip_norm = min_max_normalize(df_agg['Precip_mm'])
    rain_score = 1 - precip_norm

    df_agg['Weather_Score'] = (temp_score + rain_score) / 2
    return df_agg[['Country', 'Weather_Score']]


# -----------------------------
# 3. LAYOUT
# -----------------------------

layout = html.Div(className='main-container', children=[

    # --- CỘT TRÁI: CONTROL PANEL ---
    html.Div(className='control-panel', children=[
        html.H3("Bảng điều khiển - Gợi ý Điểm đến", style={'marginBottom': '20px'}),

        # 1. Khoảng thời gian du lịch
        html.Label("1. Chọn khoảng thời gian đi du lịch:", className="control-label"),
        dcc.DatePickerRange(
            id='rec-date-range',
            min_date_allowed=df_weather['Date'].min() if not df_weather.empty else None,
            max_date_allowed=df_weather['Date'].max() if not df_weather.empty else None,
            start_date=df_weather['Date'].min() if not df_weather.empty else None,
            end_date=df_weather['Date'].max() if not df_weather.empty else None,
            display_format='MM/YYYY',
            style={'width': '100%'}
        ),

        html.Br(), html.Br(),

        # 2. Gu thời tiết
        html.Label("2. Bạn thích kiểu thời tiết nào?", className="control-label"),
        dcc.RadioItems(
            id='rec-climate-pref',
            options=[
                {'label': 'Mát (10–15°C)', 'value': 'cool'},
                {'label': 'Ôn hòa (16–20°C)', 'value': 'neutral'},
                {'label': 'Ấm (21–26°C)', 'value': 'warm'},
            ],
            value='neutral',
            labelStyle={'display': 'block', 'marginBottom': '5px'}
        ),

        html.Hr(),

        html.Label("3. Mức độ quan trọng từng yếu tố (0–10):", className="control-label"),
        html.Div(style={'marginTop': '5px'}, children=[
            html.Div("Chi phí / Giá cả", style={'fontSize': '0.9em'}),
            dcc.Slider(id='rec-weight-cost', min=0, max=10, step=1, value=5,
                       marks={0: '0', 5: '5', 10: '10'}),
            html.Br(),
            html.Div("Thời tiết dễ chịu", style={'fontSize': '0.9em'}),
            dcc.Slider(id='rec-weight-weather', min=0, max=10, step=1, value=5,
                       marks={0: '0', 5: '5', 10: '10'}),
            html.Br(),
            html.Div("Trải nghiệm & đánh giá", style={'fontSize': '0.9em'}),
            dcc.Slider(id='rec-weight-review', min=0, max=10, step=1, value=5,
                       marks={0: '0', 5: '5', 10: '10'}),
        ]),

        html.Hr(),

        html.Label("4. Giới hạn số gợi ý:", className="control-label"),
        dcc.Slider(
            id='rec-top-k',
            min=3, max=10, step=1, value=5,
            marks={3: '3', 5: '5', 8: '8', 10: '10'}
        ),

        html.Div(
            "Hệ thống sẽ kết hợp Cost + Weather + Review thành 1 điểm tổng hợp để xếp hạng.",
            style={'fontSize': '0.8em', 'color': '#666', 'marginTop': '15px'}
        )
    ]),

    # --- CỘT PHẢI: CONTENT PANEL ---
    html.Div(className='content-panel', children=[
        html.H2("Gợi ý Điểm đến Tổng hợp", style={'textAlign': 'center'}),

        dcc.Graph(id='rec-recommend-chart', config={'responsive': True},
                  style={'height': '550px', 'marginTop': '10px'}),

        html.H4("Giải thích gợi ý:", style={'marginTop': '20px'}),
        html.Div(id='rec-explanation-list')
    ])
])


# -----------------------------
# 4. CALLBACK GỢI Ý
# -----------------------------

@app.callback(
    [Output('rec-recommend-chart', 'figure'),
     Output('rec-explanation-list', 'children')],
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
    # Nếu không có dữ liệu đủ 3 nguồn → báo lỗi
    if len(ALL_COUNTRIES) == 0:
        fig_empty = px.bar(title="Không có đủ dữ liệu để gợi ý (thiếu Cost/Weather/Review)")
        return fig_empty, html.P("Vui lòng kiểm tra lại các file dữ liệu.")

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

    # 2. Tính weather score
    df_weather_score = compute_weather_score(df_weather_filtered, climate_pref)

    # 3. Chuẩn bị cost score (cost thấp ⇒ score cao)
    df_cost_sub = df_cost_cost[df_cost_cost['Country'].isin(ALL_COUNTRIES)].copy()
    if df_cost_sub.empty:
        df_cost_sub['Cost_Score'] = 0.5
    else:
        # normalize Average Cost Index, rồi đảo chiều
        cost_norm = min_max_normalize(df_cost_sub['Average Cost Index'])
        df_cost_sub['Cost_Score'] = 1 - cost_norm

    # 4. Chuẩn bị review score
    df_review_sub = df_review_country[df_review_country['Country'].isin(ALL_COUNTRIES)].copy()
    if df_review_sub.empty:
        df_review_sub['Review_Score'] = 0.5
    else:
        df_review_sub['Review_Score'] = min_max_normalize(df_review_sub['Review_Avg'])

    # 5. Merge tất cả về cùng 1 bảng
    df_merge = pd.DataFrame({'Country': ALL_COUNTRIES})

    df_merge = df_merge.merge(df_cost_sub[['Country', 'Average Cost Index', 'Cost_Score']],
                              on='Country', how='left')
    df_merge = df_merge.merge(df_weather_score, on='Country', how='left')
    df_merge = df_merge.merge(df_review_sub[['Country', 'Review_Avg', 'Review_Score']],
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
        # Nếu user cho tất cả = 0 thì gán đều
        w_cost = w_weather = w_review = 1
        weight_sum = 3

    df_merge['Total_Score'] = (
        df_merge['Cost_Score'] * w_cost +
        df_merge['Weather_Score'] * w_weather +
        df_merge['Review_Score'] * w_review
    ) / weight_sum

    # 7. Sắp xếp & lấy top_k
    df_top = df_merge.sort_values('Total_Score', ascending=False).head(top_k)

    # 8. Vẽ biểu đồ
    fig = px.bar(
        df_top,
        x='Country',
        y='Total_Score',
        title="Top điểm đến đề xuất",
        labels={'Country': 'Quốc gia', 'Total_Score': 'Điểm gợi ý (0–1)'},
        text=df_top['Total_Score'].round(2)
    )
    fig.update_traces(textposition='auto')
    fig.update_layout(template='plotly_white', yaxis=dict(range=[0, 1]))

    # 9. Tạo explanation list
    explanation_children = []
    for _, row in df_top.iterrows():
        country = row['Country']
        cost_idx = row.get('Average Cost Index', np.nan)
        weather_sc = row['Weather_Score']
        review_avg = row.get('Review_Avg', np.nan)

        explanation_children.append(
            html.Div(style={'marginBottom': '10px'}, children=[
                html.Strong(country),
                html.Ul([
                    html.Li(f"Chi phí: {cost_idx:.1f} (Cost_Score ~ {row['Cost_Score']:.2f})"
                            if not np.isnan(cost_idx) else "Chi phí: không có dữ liệu"),
                    html.Li(f"Thời tiết phù hợp: Weather_Score ~ {weather_sc:.2f}"),
                    html.Li(f"Đánh giá trung bình: {review_avg:.2f} (Review_Score ~ {row['Review_Score']:.2f})"
                            if not np.isnan(review_avg) else "Đánh giá: không có dữ liệu"),
                ])
            ])
        )

    return fig, explanation_children
