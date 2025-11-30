# index.py
from dash import dcc, html
from dash.dependencies import Input, Output

# Import app và các trang con
from app import app
import weather_app
import cost_app
import forecast_app
import review_app
import recommend_app   # <-- THÊM MỚI: Import trang Gợi ý

server = app.server
# --- Layout chính ---
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    
    html.Div(className='header', children=[
        html.H1("Dashboard Phân tích Tổng hợp", className="header-title")
    ]),
    
    # Thanh điều hướng
    html.Div(className='navigation-bar', children=[
        dcc.Link('Phân tích Lịch sử', href='/weather', className='nav-link'),
        dcc.Link('Phân tích Giá cả', href='/cost', className='nav-link'),
        dcc.Link('Đánh giá Điểm đến', href='/review', className='nav-link'),
        dcc.Link('Dự báo Du lịch', href='/forecast', className='nav-link'),
        dcc.Link('Gợi ý Điểm đến', href='/recommend', className='nav-link'),  # <-- LINK MỚI
    ]),
    
    html.Div(id='page-content', className='app-container')
])

# --- Bộ định tuyến giữa các trang ---
@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')]
)
def display_page(pathname):
    if pathname in ['/', '/weather']:
        return weather_app.layout
    elif pathname == '/cost':
        return cost_app.layout
    elif pathname == '/forecast':
        return forecast_app.layout
    elif pathname == '/review':
        return review_app.layout
    elif pathname == '/recommend':      # <-- ROUTE MỚI
        return recommend_app.layout
    else:
        # Path lạ → quay về trang thời tiết (mặc định)
        return weather_app.layout


if __name__ == '__main__':
    app.run(debug=True)
