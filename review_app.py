# review_app.py - Dashboard Đánh giá (Google Reviews)
import pandas as pd
import plotly.express as px
from dash import dcc, html
from dash.dependencies import Input, Output

from app import app

# --- 1. Load & Chuẩn bị dữ liệu ---

# Đường dẫn file (sửa lại nếu cần)
file_id_review = "1tcsQodOIGlroMDDdfYowl1OHJUSHYrRB"
file_path_review = f"https://drive.google.com/uc?export=download&id={file_id_review}"

try:
    df_raw = pd.read_csv(file_path_review)
except Exception as e:
    print("Lỗi tải dữ liệu review:", e)
    df_raw = pd.DataFrame()


try:
    df_raw = pd.read_csv(file_path_review)

    # Unpivot: User, Country, Category_Name, Rating
    id_vars = ['User', 'Country']
    category_columns = [col for col in df_raw.columns if col not in id_vars]

    df_unpivoted = df_raw.melt(
        id_vars=id_vars,
        value_vars=category_columns,
        var_name="Category_Name",
        value_name="Rating"
    )

    # Chỉ giữ Rating > 0
    df_reviews = df_unpivoted[df_unpivoted["Rating"] > 0]

    all_categories = sorted(df_reviews['Category_Name'].unique())
    all_countries = sorted(df_reviews['Country'].unique())

    # --- Cấp 1: Xếp hạng chung ---
    df_overall_rating = df_reviews.groupby('Country')['Rating'].mean().reset_index()
    if not df_overall_rating.empty:
        max_overall_rating = df_overall_rating['Rating'].max()
        df_overall_rating['highlight'] = df_overall_rating['Rating'].apply(
            lambda x: x == max_overall_rating
        )
    else:
        df_overall_rating['highlight'] = []

    # --- Cấp 3: Heatmap ---
    if not df_reviews.empty:
        df_heatmap_data = df_reviews.groupby(
            ['Country', 'Category_Name']
        )['Rating'].mean().reset_index()

        df_heatmap_pivot = df_heatmap_data.pivot(
            index='Country',
            columns='Category_Name',
            values='Rating'
        )
    else:
        df_heatmap_pivot = pd.DataFrame()

    print("Đã tải & xử lý dữ liệu review thành công.")

except FileNotFoundError:
    print(f"LỖI: Không tìm thấy file tại {file_path_review}")
    # Tạo dữ liệu rỗng để app không bị crash
    df_reviews = pd.DataFrame(columns=["User", "Country", "Category_Name", "Rating"])
    all_categories = []
    all_countries = []
    df_overall_rating = pd.DataFrame(columns=["Country", "Rating", "highlight"])
    df_heatmap_pivot = pd.DataFrame()


# --- 2. Layout (giống cấu trúc cost_app: main-container / control-panel / content-panel) ---

layout = html.Div(className='main-container', children=[

    # -------- CỘT TRÁI: CONTROL PANEL --------
    html.Div(className='control-panel', children=[
        html.H3("Bảng điều khiển - Đánh giá", style={'marginBottom': '20px'}),

        # Bộ lọc hạng mục (cho dynamic bar chart)
        html.Label("1. Chọn các hạng mục bạn quan tâm:",
                   className="control-label"),
        dcc.Dropdown(
            id='review-category-filter',
            options=[{'label': cat, 'value': cat} for cat in all_categories],
            value=[all_categories[0]] if all_categories else [],
            multi=True,
            placeholder="Chọn 1 hoặc nhiều hạng mục"
        ),
        html.P(
            "Biểu đồ 'Gợi ý theo Sở thích' sẽ tìm quốc gia phù hợp nhất với lựa chọn của bạn.",
            style={'fontSize': '0.85em', 'color': '#666', 'marginTop': '5px'}
        ),

        html.Hr(),

        # Bộ lọc quốc gia (cho radar + top 10)
        html.Label("2. Chọn 1 Quốc gia để xem hồ sơ chi tiết:",
                   className="control-label"),
        dcc.Dropdown(
            id='review-country-filter',
            options=[{'label': c, 'value': c} for c in all_countries],
            value=all_countries[0] if all_countries else None,
            clearable=False,
            placeholder="Chọn quốc gia"
        ),
        html.P(
            "Biểu đồ Radar và Top 10 hạng mục sẽ được cập nhật theo quốc gia này.",
            style={'fontSize': '0.85em', 'color': '#666', 'marginTop': '5px'}
        ),

        html.Hr(),

        html.Div(
            "Phần 'Xếp hạng chung' và 'Heatmap' nằm ở cột phải, dùng cho bối cảnh tổng quan.",
            style={
                'fontSize': '0.85em',
                'color': '#555',
                'background': '#f8f9fa',
                'padding': '10px',
                'borderRadius': '6px'
            }
        ),
    ]),

    # -------- CỘT PHẢI: CONTENT PANEL --------
    html.Div(className='content-panel', children=[

        html.H2(
            "Phân tích Trải nghiệm (Dashboard Review)",
            style={'textAlign': 'center', 'marginBottom': '30px'}
        ),

        # 1. Dynamic Bar Chart - Gợi ý theo sở thích
        html.Div(style={'marginBottom': '40px'}, children=[
            html.H3(" Gợi ý theo Sở thích"),
            dcc.Graph(id='review-dynamic-bar-chart',
                      config={'responsive': True})
        ]),

        html.Hr(),

        # 2. Radar + Top 10 (2 biểu đồ, xếp dọc cho mobile-friendly)
        html.Div(style={'marginTop': '20px', 'marginBottom': '40px'}, children=[
            html.H3("🎨 Hồ sơ chi tiết Quốc gia"),
            html.P(
                "Radar thể hiện diện mạo tổng thể theo hạng mục; biểu đồ Top 10 cho biết các hạng mục nổi bật nhất.",
                style={'fontSize': '0.9em', 'color': '#555'}
            ),
            dcc.Graph(id='review-radar-chart',
                      config={'responsive': True},
                      style={'marginBottom': '30px'}),
            dcc.Graph(id='review-top-10-bar-chart',
                      config={'responsive': True})
        ]),

        html.Hr(),

        # 3. Overall Rating
        html.Div(style={'marginTop': '20px', 'marginBottom': '40px'}, children=[
            html.H3("🗺️ Xếp hạng Hài lòng Chung theo Quốc gia"),
            dcc.Graph(
                id='review-overall-bar-chart',
                config={'responsive': True},
                figure=px.bar(
                    df_overall_rating,
                    x='Country',
                    y='Rating',
                    title="Quốc gia nào được đánh giá hài lòng nhất (Tổng thể)",
                    labels={'Country': 'Quốc gia', 'Rating': 'Rating Trung bình'},
                    color='highlight',
                    color_discrete_map={
                        True: "#f49eaa",   # đỏ cho quốc gia cao nhất
                        False: "#7eb3d8"   # xanh cho các quốc gia còn lại
                    }
                ).update_layout(
                    xaxis={'categoryorder': 'total descending'},
                    yaxis=dict(range=[2.2, 2.6], dtick=0.05),
                    showlegend=False,
                    bargap=0.3,
                    template='plotly_white'
                )
            )
        ]),

        html.Hr(),

        # 4. Heatmap
        html.Div(style={'marginTop': '20px'}, children=[
            html.H3("🔥 Ma trận So sánh Hạng mục giữa các Quốc gia"),
            dcc.Graph(
                id='review-category-heatmap',
                config={'responsive': True},
                figure=px.imshow(
                    df_heatmap_pivot,
                    labels=dict(
                        x="Hạng mục",
                        y="Quốc gia",
                        color="Rating trung bình"
                    ),
                    title="So sánh chi tiết điểm mạnh/yếu của các Quốc gia",
                    aspect="auto"
                ).update_layout(template='plotly_white')
            )
        ])
    ])
])


# --- 3. Callbacks ---

# Callback 1: Dynamic Bar Chart (Gợi ý theo sở thích)
@app.callback(
    Output('review-dynamic-bar-chart', 'figure'),
    [Input('review-category-filter', 'value')]
)
def update_review_dynamic_bar(selected_categories):
    if df_reviews is None or df_reviews.empty:
        return px.bar(title="Không có dữ liệu", labels={'x': 'Quốc gia', 'y': 'Rating Trung bình'})

    if not selected_categories:
        selected_categories = []

    filtered_df = df_reviews[df_reviews['Category_Name'].isin(selected_categories)]

    if filtered_df.empty:
        return px.bar(
            title="Vui lòng chọn ít nhất 1 hạng mục",
            labels={'x': 'Quốc gia', 'y': 'Rating Trung bình'}
        )

    avg_rating_df = filtered_df.groupby('Country')['Rating'].mean().reset_index()

    # Tô màu quốc gia có rating cao nhất
    max_rating = avg_rating_df['Rating'].max()
    avg_rating_df['highlight'] = avg_rating_df['Rating'].apply(lambda x: x == max_rating)

    fig = px.bar(
        avg_rating_df,
        x='Country',
        y='Rating',
        title="Quốc gia phù hợp nhất dựa trên lựa chọn của bạn",
        labels={'Country': 'Quốc gia', 'Rating': 'Rating Trung bình'},
        color='highlight',
        color_discrete_map={
            True: "#f69292",   # đỏ
            False: "#80b8e0"   # xanh
        }
    )
    fig.update_layout(
        xaxis={'categoryorder': 'total descending'},
        yaxis=dict(range=[2.0, 3.0], dtick=0.08),
        showlegend=False,
        bargap=0.3,
        template='plotly_white'
    )
    return fig


# Callback 2: Radar Chart
@app.callback(
    Output('review-radar-chart', 'figure'),
    [Input('review-country-filter', 'value')]
)
def update_review_radar(selected_country):
    if df_reviews is None or df_reviews.empty or not selected_country:
        return px.line_polar(title="Không có dữ liệu")

    filtered_df = df_reviews[df_reviews['Country'] == selected_country]

    avg_rating_df = filtered_df.groupby('Category_Name')['Rating'].mean().reset_index()

    fig = px.line_polar(
        avg_rating_df,
        r='Rating',
        theta='Category_Name',
        line_close=True,
        title=f"Hồ sơ tổng thể của {selected_country}",
        labels={'Rating': 'Rating TB', 'Category_Name': 'Hạng mục'}
    )
    fig.update_traces(fill='toself')
    fig.update_layout(template='plotly_white', height=450)
    return fig


# Callback 3: Top 10 Category Chart
@app.callback(
    Output('review-top-10-bar-chart', 'figure'),
    [Input('review-country-filter', 'value')]
)
def update_review_top10(selected_country):
    if df_reviews is None or df_reviews.empty or not selected_country:
        return px.bar(title="Không có dữ liệu")

    filtered_df = df_reviews[df_reviews['Country'] == selected_country]
    avg_rating_df = filtered_df.groupby('Category_Name')['Rating'].mean().reset_index()

    top_10_df = avg_rating_df.sort_values(by='Rating', ascending=False).head(10)

    fig = px.bar(
        top_10_df,
        y='Category_Name',
        x='Rating',
        orientation='h',
        title=f"Top 10 Hạng mục tại {selected_country}",
        labels={'Category_Name': 'Hạng mục', 'Rating': 'Rating Trung bình'}
    )
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        bargap=0.3,
        template='plotly_white',
        height=450
    )
    return fig
