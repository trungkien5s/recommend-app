# cost_app.py - Enhanced Version
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

from app import app

# --- 0. Mapping tiếng Việt ---
METRIC_LABELS_VI = {
    'Cost of Living Index': 'Chi phí Sinh hoạt',
    'Rent Index': 'Giá Thuê Nhà',
    'Cost of Living Plus Rent Index': 'Chi phí Sinh hoạt + Thuê Nhà',
    'Groceries Index': 'Giá Thực Phẩm',
    'Restaurant Price Index': 'Giá Nhà hàng',
    'Local Purchasing Power Index': 'Sức Mua Địa Phương',
    'Average Cost Index': 'Chi phí Trung bình'
}

def get_metric_label(metric_name: str) -> str:
    return METRIC_LABELS_VI.get(metric_name, metric_name)

# --- 1. Load Data ---
file_id = "1q5qP66oBoxCEXtGrvE5jtMT5aTATfpn6"
file_path_cost = f"https://drive.google.com/uc?export=download&id={file_id}"



try:
    df_cost = pd.read_csv(file_path_cost)
except Exception as e:
    print("Lỗi tải dữ liệu cost:", e)
    df_cost = pd.DataFrame()

BUDGET_LABELS = ['Thấp', 'Trung bình', 'Cao']
BUDGET_BINS = [0, 40, 80, float('inf')]
BUDGET_COLORS = {
    'Thấp': '#4A90E2',      # Xanh dương nhạt
    'Trung bình': '#7B68EE', # Tím nhạt
    'Cao': '#E67E73'         # Đỏ cam nhạt
}

try:
    df_cost = pd.read_csv(file_path_cost)
    df_cost = df_cost.dropna(subset=['Country'])
    ALL_COUNTRIES_COST = sorted(df_cost['Country'].unique())

    cost_indices_columns = [
        'Cost of Living Index', 
        'Rent Index', 
        'Cost of Living Plus Rent Index', 
        'Groceries Index', 
        'Restaurant Price Index',
    
    ]
    
    df_cost['Average Cost Index'] = df_cost[cost_indices_columns].mean(axis=1).round(2)
    df_cost['Budget_Category'] = pd.cut(
        df_cost['Average Cost Index'],
        bins=BUDGET_BINS,
        labels=BUDGET_LABELS,
        right=False
)

    METRIC_OPTIONS_COST = [
        {"label": get_metric_label(col), "value": col}
        for col in df_cost.columns
        if col not in ['Country', 'Budget_Category']
    ]
    METRIC_NAMES_COST = [col['value'] for col in METRIC_OPTIONS_COST]

except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy tệp tại đường dẫn: {file_path_cost}")
    df_cost = pd.DataFrame()
    ALL_COUNTRIES_COST = []
    METRIC_OPTIONS_COST = []
    METRIC_NAMES_COST = []

# --- 2. Layout ---
layout = html.Div(className='main-container', children=[
    
    # Control Panel
    html.Div(className='control-panel', children=[
        html.H3(" Bảng điều khiển", style={'marginBottom': '20px'}),
        
        # Visualization Mode
        html.Label("Chế độ Trực quan hóa:", className="control-label"),
        dcc.RadioItems(
            id='cost-mode-selector',
            options=[
                {'label': 'So sánh Cơ bản (Biểu đồ cột)', 'value': 'bar'},
                # {'label': 'Phân tích Tương quan', 'value': 'scatter'},
                # {'label': ' Biểu đồ Radar (Đa chiều)', 'value': 'radar'},
                {'label': ' Biểu đồ Nhiệt (Heatmap)', 'value': 'heatmap'},
                # {'label': ' Biểu đồ Hộp (Box Plot)', 'value': 'box'},
                {'label': ' Phân bố Ngân sách (Pie)', 'value': 'pie'}
            ],
            value='bar',
            labelStyle={'display': 'block', 'marginBottom': '8px'}
        ),
        
        html.Hr(),
        
        # Budget Filter
        html.Label("Lọc theo Ngân sách:", className="control-label"),
dcc.Dropdown(
    id='cost-budget-filter',
    options=[
        {'label': ' Tất cả', 'value': 'All'}
    ] + [
        {
            'label': (
                # Nếu là bin cuối (vô cùng) → dùng dạng "> 85"
                f'{b} (> {BUDGET_BINS[i]})'
                if BUDGET_BINS[i+1] == float("inf")
                else 
                # Còn lại thì dùng dạng "10 - 30"
                f'{b} ({BUDGET_BINS[i]} - {BUDGET_BINS[i+1]})'
            ),
            'value': b
        }
        for i, b in enumerate(BUDGET_LABELS)
    ],
    value='All',
    clearable=False
),
html.Br(),


        
        # Country Selection
        html.Div(id='cost-country-controls', children=[
            html.Label("Chọn Quốc gia:", className="control-label"),
            dcc.Dropdown(
                id='cost-country-multi-dropdown',
                options=[{'label': c, 'value': c} for c in ALL_COUNTRIES_COST],
                multi=True,
                value=['Vietnam', 'Singapore', 'United States', 'Japan', 'Thailand']
            ),
        ]),
        html.Br(),
        
        # Metric Selection (for bar/scatter/box)
        html.Div(id='cost-metric-controls', children=[
            html.Label(" Chọn Chỉ số chính:", className="control-label"),
            dcc.Dropdown(
                id='cost-metric-dropdown',
                options=METRIC_OPTIONS_COST,
                value='Cost of Living Index'
            ),
        ]),
        
        # Scatter Specific Controls
        html.Div(id='cost-scatter-controls', style={'display': 'none'}, children=[
            html.Label(" Chỉ số trục X:", className="control-label"),
            dcc.Dropdown(
                id='cost-scatter-x-metric',
                options=METRIC_OPTIONS_COST,
                value='Cost of Living Index'
            ),
            html.Br(),
            html.Label(" Chỉ số trục Y:", className="control-label"),
            dcc.Dropdown(
                id='cost-scatter-y-metric',
                options=METRIC_OPTIONS_COST,
                value='Local Purchasing Power Index'
            ),
        ]),
        
        html.Hr(),
        
        # Quick Stats
        html.Div(id='quick-stats', style={
            'padding': '15px',
            'background': '#f8f9fa',
            'borderRadius': '8px',
            'marginTop': '20px'
        })
    ]),
    
    # Content Panel
    html.Div(className='content-panel', children=[
        dcc.Graph(id='cost-chart', config={'responsive': True}, style={'height': '600px'}),
        
        # Secondary Chart (for comparison)
        html.Div(id='secondary-chart-container', style={'marginTop': '30px'}),
        
        # html.H4(" Bảng dữ liệu chi tiết", style={'marginTop': '30px'}),
        # dash_table.DataTable(
        #     id='cost-table',
        #     columns=[{"name": "Quốc gia", "id": "Country"},
        #              {"name": "Ngân sách", "id": "Budget_Category"}] + 
        #             [{"name": get_metric_label(i), "id": i} for i in METRIC_NAMES_COST],
        #     data=df_cost.to_dict('records'),
        #     page_size=15,
        #     sort_action="native",
        #     filter_action="native",
        #     style_table={'overflowX': 'auto'},
        #     style_cell={'textAlign': 'left', 'padding': '10px'},
        #     style_header={
        #         'backgroundColor': '#34495e',
        #         'color': 'white',
        #         'fontWeight': 'bold'
        #     },
        #     style_data_conditional=[
        #         {
        #             'if': {'column_id': 'Budget_Category', 'filter_query': '{Budget_Category} = "Thấp"'},
        #             'backgroundColor': '#d5f4e6',
        #         },
        #         {
        #             'if': {'column_id': 'Budget_Category', 'filter_query': '{Budget_Category} = "Trung bình"'},
        #             'backgroundColor': '#fff3cd',
        #         },
        #         {
        #             'if': {'column_id': 'Budget_Category', 'filter_query': '{Budget_Category} = "Cao"'},
        #             'backgroundColor': '#f8d7da',
        #         }
        #     ]
        # )
    ])
])

# --- 3. Callbacks ---

# Toggle controls based on mode
@app.callback(
    [Output('cost-metric-controls', 'style'),
     Output('cost-scatter-controls', 'style'),
     Output('cost-country-controls', 'style')],
    [Input('cost-mode-selector', 'value')]
)
def toggle_cost_controls(mode):
    if mode == 'bar':
        return {'display': 'block'}, {'display': 'none'}, {'display': 'block'}
    elif mode == 'scatter':
        return {'display': 'none'}, {'display': 'block'}, {'display': 'block'}
    elif mode in ['radar', 'box']:
        return {'display': 'block'}, {'display': 'none'}, {'display': 'block'}
    else:  # heatmap, pie
        return {'display': 'none'}, {'display': 'none'}, {'display': 'none'}

# Update country dropdown based on budget
@app.callback(
    [Output('cost-country-multi-dropdown', 'options'),
     Output('cost-country-multi-dropdown', 'value')],
    [Input('cost-budget-filter', 'value')]
)
def update_country_dropdown(selected_budget):
    if selected_budget == 'All':
        filtered_countries_list = ALL_COUNTRIES_COST
    else:
        filtered_df = df_cost[df_cost['Budget_Category'] == selected_budget]
        filtered_countries_list = sorted(filtered_df['Country'].unique())
    
    new_options = [{'label': c, 'value': c} for c in filtered_countries_list]
    new_value = filtered_countries_list[:7] if len(filtered_countries_list) >= 7 else filtered_countries_list
    
    return new_options, new_value

# Update quick stats
@app.callback(
    Output('quick-stats', 'children'),
    [Input('cost-budget-filter', 'value'),
     Input('cost-country-multi-dropdown', 'value')]
)
def update_quick_stats(budget_filter, selected_countries):
    if df_cost.empty:
        return html.Div("Không có dữ liệu")
    
    if budget_filter == 'All':
        filtered_df = df_cost
    else:
        filtered_df = df_cost[df_cost['Budget_Category'] == budget_filter]
    
    if selected_countries:
        filtered_df = filtered_df[filtered_df['Country'].isin(selected_countries)]
    
    if filtered_df.empty:
        return html.Div("Không có dữ liệu cho bộ lọc này")
    
    avg_cost = filtered_df['Cost of Living Index'].mean()
    min_country = filtered_df.loc[filtered_df['Cost of Living Index'].idxmin(), 'Country']
    max_country = filtered_df.loc[filtered_df['Cost of Living Index'].idxmax(), 'Country']
    
    return html.Div([
        html.H4(" Thống kê Nhanh", style={'marginBottom': '10px'}),
        html.P(f" Số quốc gia: {len(filtered_df)}"),
        html.P(f" Chi phí TB: {avg_cost:.1f}"),
        html.P(f"⬇ Thấp nhất: {min_country}"),
        html.P(f"⬆ Cao nhất: {max_country}")
    ])

# Main chart update
@app.callback(
    Output('cost-chart', 'figure'),
    [Input('cost-mode-selector', 'value'),
     Input('cost-country-multi-dropdown', 'value'),
     Input('cost-metric-dropdown', 'value'),
     Input('cost-scatter-x-metric', 'value'),
     Input('cost-scatter-y-metric', 'value')]
)
def update_cost_chart(mode, selected_countries, bar_metric, scatter_x, scatter_y):
    if df_cost.empty:
        return go.Figure(layout={'title': 'Lỗi tải dữ liệu'})

    fig = go.Figure()
    
    # BAR CHART
    if mode == 'bar':
        if not selected_countries or not bar_metric:
            fig.update_layout(title="Vui lòng chọn Quốc gia và Chỉ số")
            return fig
        
        metric_label = get_metric_label(bar_metric)
        dff = df_cost[df_cost['Country'].isin(selected_countries)].sort_values(by=bar_metric, ascending=False)
        
        colors = [BUDGET_COLORS.get(cat, '#95a5a6') for cat in dff['Budget_Category']]
        
        fig.add_trace(go.Bar(
            x=dff['Country'],
            y=dff[bar_metric],
            text=dff[bar_metric].round(1),
            textposition='auto',
            marker=dict(color=colors),
            hovertemplate='<b>%{x}</b><br>' + metric_label + ': %{y:.1f}<extra></extra>'
        ))
        
        fig.update_layout(
            # title=f"So sánh {metric_label}",
            xaxis_title="Quốc gia",
            yaxis_title=metric_label,
            height=600,
            template='plotly_white'
        )
    
    # SCATTER PLOT
    elif mode == 'scatter':
        if not scatter_x or not scatter_y:
            fig.update_layout(title="Vui lòng chọn chỉ số X và Y")
            return fig
        
        x_label = get_metric_label(scatter_x)
        y_label = get_metric_label(scatter_y)
        
        colors_map = {    'Thấp': '#4A90E2',      # Xanh dương nhạt
    'Trung bình': '#7B68EE', # Tím nhạt
    'Cao': '#E67E73'         # Đỏ cam nhạt
}
        
        for category in BUDGET_LABELS:
            dff = df_cost[df_cost['Budget_Category'] == category]
            fig.add_trace(go.Scatter(
                x=dff[scatter_x],
                y=dff[scatter_y],
                mode='markers',
                name=category,
                text=dff['Country'],
                marker=dict(size=10, color=colors_map[category], opacity=0.7),
                hovertemplate='<b>%{text}</b><br>' + x_label + ': %{x:.1f}<br>' + y_label + ': %{y:.1f}<extra></extra>'
            ))
        
        fig.update_layout(
            # title=f"Tương quan: {x_label} vs {y_label}",
            xaxis_title=x_label,
            yaxis_title=y_label,
            height=600,
            template='plotly_white',
            hovermode='closest'
        )
    
    # RADAR CHART
    elif mode == 'radar':
        if not selected_countries:
            fig.update_layout(title="Vui lòng chọn ít nhất một quốc gia")
            return fig
        
        metrics = ['Cost of Living Index', 'Rent Index', 'Groceries Index', 
                   'Restaurant Price Index', 'Local Purchasing Power Index']
        
        for country in selected_countries[:7]:  # Limit to 5 countries
            country_data = df_cost[df_cost['Country'] == country]
            if not country_data.empty:
                values = [country_data[m].values[0] for m in metrics]
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=[get_metric_label(m) for m in metrics],
                    fill='toself',
                    name=country
                ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 160])),
            # title=" Phân tích Đa chiều (Radar)",
            height=600,
            template='plotly_white'
        )
    
    # HEATMAP
    elif mode == 'heatmap':
        metrics = ['Cost of Living Index', 'Rent Index', 'Groceries Index', 
                   'Restaurant Price Index', 'Local Purchasing Power Index']
        
        # Top 20 countries by Cost of Living
        top_countries = df_cost.nlargest(20, 'Cost of Living Index')['Country'].tolist()
        dff = df_cost[df_cost['Country'].isin(top_countries)]
        
        heatmap_data = dff[metrics].T
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=dff['Country'].values,
            y=[get_metric_label(m) for m in metrics],
            colorscale='RdYlGn_r',
            text=heatmap_data.values.round(1),
            texttemplate='%{text}',
            textfont={"size": 10},
            hovertemplate='%{y}<br>%{x}<br>Giá trị: %{z:.1f}<extra></extra>'
        ))
        
        fig.update_layout(
            # title=" Bản đồ Nhiệt - Top 20 Quốc gia Chi phí Cao",
            height=600,
            template='plotly_white'
        )
    
    # BOX PLOT
    elif mode == 'box':
        if not bar_metric:
            fig.update_layout(title="Vui lòng chọn Chỉ số")
            return fig
        
        metric_label = get_metric_label(bar_metric)
        
        for category in BUDGET_LABELS:
            dff = df_cost[df_cost['Budget_Category'] == category]
            fig.add_trace(go.Box(
                y=dff[bar_metric],
                name=category,
                marker_color=BUDGET_COLORS[category],
                boxmean='sd'
            ))
        
        fig.update_layout(
            title=f" Phân bố {metric_label} theo Ngân sách",
            yaxis_title=metric_label,
            height=600,
            template='plotly_white'
        )
    
    # PIE CHART
    elif mode == 'pie':
        budget_counts = df_cost['Budget_Category'].value_counts()
        
        fig = go.Figure(data=[go.Pie(
            labels=budget_counts.index,
            values=budget_counts.values,
            marker=dict(colors=[BUDGET_COLORS[cat] for cat in budget_counts.index]),
            textinfo='label+percent',
            hovertemplate='<b>%{label}</b><br>Số quốc gia: %{value}<br>Tỷ lệ: %{percent}<extra></extra>'
        )])
        
        fig.update_layout(
            # title=" Phân bố Quốc gia theo Mức Ngân sách",
            height=600,
            template='plotly_white'
        )
    
    return fig

# Secondary chart (Comparison)
@app.callback(
    Output('secondary-chart-container', 'children'),
    [Input('cost-mode-selector', 'value'),
     Input('cost-country-multi-dropdown', 'value')]
)
def update_secondary_chart(mode, selected_countries):
    if mode not in ['bar', 'radar'] or not selected_countries:
        return html.Div()
    
    # Create a multi-metric comparison
    metrics = ['Cost of Living Index', 'Rent Index', 'Groceries Index', 
               'Restaurant Price Index', 'Local Purchasing Power Index']
    
    dff = df_cost[df_cost['Country'].isin(selected_countries)]
    
    fig = go.Figure()
    
    for metric in metrics:
        fig.add_trace(go.Bar(
            name=get_metric_label(metric),
            x=dff['Country'],
            y=dff[metric],
            text=dff[metric].round(1),
            textposition='auto'
        ))
    
    fig.update_layout(
        # title=" So sánh Tổng hợp Tất cả Chỉ số",
        barmode='group',
        height=400,
        template='plotly_white',
        xaxis_title="Quốc gia",
        yaxis_title="Giá trị Chỉ số"
    )
    
    return dcc.Graph(figure=fig, config={'responsive': True})