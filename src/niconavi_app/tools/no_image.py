# %%
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from matplotlib.pyplot import Figure, Axes
import numpy as np

def get_no_image() -> Figure:
    # FigureとAxesを作成

    # rng = np.random.default_rng(10)
    # data = rng.random((10, 10))
    # print(data)

    # fig, ax = plt.subplots()
    # ax.imshow(data, aspect="equal")

    fig, ax = plt.subplots(figsize=(4, 4))
    # ax.scatter(np.linspace(0, 1, 100), np.linspace(0, 1, 100))

    # 背景を白に設定（必要に応じて）
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    # 外側のリング（部分円弧）
    # theta1=30, theta2=330 で上側に30度分の“欠け”を作る
    # outer_arc = patches.Arc(
    #     xy=(0, 0),  # 中心
    #     width=2.0,  # 横径
    #     height=2.0,  # 縦径
    #     angle=0,  # 回転角度
    #     theta1=0,  # 開始角度
    #     theta2=360,  # 終了角度
    #     linewidth=3,
    #     color="#66C3AE",
    # )
    # ax.add_patch(outer_arc)

    # 内側のリング（部分円弧）
    # theta1=60, theta2=360 でやや左上が欠けるように調整
    # inner_arc = patches.Arc(
    #     xy=(0, 0),
    #     width=1.5,
    #     height=1.5,
    #     angle=0,
    #     theta1=60,
    #     theta2=360,
    #     linewidth=3,
    #     color="#66C3AE",
    # )
    # ax.add_patch(inner_arc)

    # 中央のプラス記号（直線を2本重ねる）
    # プラスの大きさは適宜調整
    # ax.plot([-0.05, 0.05], [0, 0], color="#66C3AE", linewidth=2)
    # ax.plot([0, 0], [-0.05, 0.05], color="#66C3AE", linewidth=2)

    # ロゴ下部のテキスト
    # 位置やフォントサイズは好みに合わせて調整
    # ax.text(-0.05, -1.3, "SUMA", color="#66C3AE", fontsize=10, ha="center", va="center")
    # ax.text(
    #     0.7, -1.35, "v0.1.0", color="#66C3AE", fontsize=4, ha="center", va="center"
    # )

    # fig.subplots_adjust(left=0.4, right=0.6, bottom=0.4, top=0.6)
    # # 軸を消して円がゆがまないよう比率を固定
    # ax.set_aspect("equal")
    # ax.axis("off")


    # plt.show()

    return fig


if __name__ == "__main__":
    get_no_image()

# %%
