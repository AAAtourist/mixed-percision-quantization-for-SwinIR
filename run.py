from flask import Flask, render_template_string, send_from_directory
import os

app = Flask(__name__)

# HTML 模板，用于展示图片
html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>图片展示</title>
    <style>
        .container {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            justify-content: center;
            padding: 20px;
        }
        .image-item {
            text-align: center;
        }
        .image-item img {
            max-width: 200px;
            height: auto;
            border: 1px solid #ccc;
            border-radius: 5px;
        }
    </style>
</head>
<body>
    <h1 style="text-align:center;">图片展示</h1>
    <div class="container">
        {% for filename in images %}
            <div class="image-item">
                <img src="{{ url_for('serve_image', filename=filename) }}" alt="{{ filename }}">
                <p>{{ filename }}</p>
            </div>
        {% endfor %}
    </div>
</body>
</html>
"""

# 这里设置图片存储的文件夹路径
IMAGE_FOLDER = "/data/user/tourist/mixed-percision-quantization-for-SwinIR/results/search_quant_x2_4bit/visualization/Urban100"

@app.route('/')
def index():
    # 获取文件夹中的所有图片文件（jpg, png, jpeg等）
    images = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    return render_template_string(html_template, images=images)

@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory(IMAGE_FOLDER, filename)

if __name__ == '__main__':
    # 启动Flask应用
    app.run(host='0.0.0.0', port=2418, debug=True)
