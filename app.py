# app.py
import dash

# Khởi tạo ứng dụng Dash
# suppress_callback_exceptions=True là BẮT BUỘC cho ứng dụng đa trang
app = dash.Dash(__name__, 
                external_stylesheets=['styles.css'], 
                suppress_callback_exceptions=True)

app.title = "Dashboard Phân tích Tổng hợp"
server = app.server