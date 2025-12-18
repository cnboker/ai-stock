from config import COLORS

def generate_tail_label(future_index, median, high, index, name):
    return {
        "x": future_index[-1],
        "y": high[-1],
        "name": name,
        "price": median[-1],
        "color": COLORS[index],
    }

def create_annotation(prediction_tails):
    # ==================== 2×2 子图专用尾部标签（完美不乱） ====================
    annotations = []

    # 子图坐标映射
    subplot_refs = [
        ("x1", "y1"),  # 第1只：左上
        ("x2", "y2"),  # 第2只：右上
        ("x3", "y3"),  # 第3只：左下
        ("x4", "y4"),  # 第4只：右下
    ]

    for i, tail in enumerate(prediction_tails):
        name = tail["name"]
        price = tail["price"]
        color = tail["color"]
        x = tail["x"]
        y = tail["y"]

        xref, yref = subplot_refs[i]  # 每只股票用自己的子图坐标系

        annotations.append(
            dict(
                x=x,
                y=y,
                xref=xref,
                yref=yref,
                text=f" → <b>{name} {price:.2f}</b>",
                font=dict(size=15, color="white", family="Consolas"),
                arrowhead=2,
                arrowcolor=color,
                arrowwidth=3,
                ax=-80,  # 箭头指向左，指向预测终点
                ay=0,
                opacity=0.96,
                xanchor="left",
                yanchor="middle",
            )
        )
    return annotations
