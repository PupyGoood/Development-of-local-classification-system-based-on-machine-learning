import warnings
from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import QMainWindow, QApplication, QLineEdit, QHBoxLayout, QPushButton, QTabWidget, QLabel
import SVMRunner, DecisionTreeRunner, MLPRunner, RandomForestRunner, KNNRunner, LRRunner


class mainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.title = "ML Application"
        self.top = 200
        self.left = 500
        self.width = 410
        self.height = 600
        self.iconName = r"C:\Users\LWY520\Desktop\weixiang_word\预备图\character26.png"
        self.backgroundImagePath = r"background.png"  # 更改为你的背景图片路径

        self.initUI()

    def initUI(self):
        # 设置背景颜色为淡绿色
        self.setStyleSheet("background-color: lightgreen;")

        self.setWindowTitle(self.title)
        self.setWindowIcon(QtGui.QIcon(self.iconName))
        self.setGeometry(self.left, self.top, self.width, self.height)

        # 设置背景图片
        self.setBackgroundImage()

        tab_widget = QTabWidget(self)
        tab_widget.setGeometry(QtCore.QRect(0, 0, self.width, self.height))

        svm_runner = SVMRunner.Main()
        dt_runner = DecisionTreeRunner.Main()
        mlp_runner = MLPRunner.Main()
        RF_runner = RandomForestRunner.Main()
        KNN_runner = KNNRunner.Main()
        LR_runner = LRRunner.Main()

        tab_widget.addTab(svm_runner, "SVM")
        tab_widget.addTab(dt_runner, "Decision Tree")
        tab_widget.addTab(mlp_runner, "MLP")
        tab_widget.addTab(RF_runner, "Random Forest")
        tab_widget.addTab(KNN_runner, "KNN")
        tab_widget.addTab(LR_runner, "LR")

        self.show()

    def setBackgroundImage(self):
        # 创建 QLabel 以显示背景图片
        self.background_label = QLabel(self)
        self.background_label.setGeometry(410, 10, self.width, self.height)

        # 加载背景图片
        background_pixmap = QtGui.QPixmap(self.backgroundImagePath)
        # 检查图片是否加载成功
        if background_pixmap.isNull():
            print(f"Failed to load background image from {self.backgroundImagePath}")
            return
        # 设置背景图片
        self.background_label.setPixmap(background_pixmap)
        self.background_label.setScaledContents(True)  # 确保图片适应标签大小
        # 设置透明度
        self.background_label.setWindowOpacity(0.2)  # 调整透明度值，范围为0.0到1.0

        # 确保背景图片在窗口最底层
        self.background_label.lower()

if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    mWin = mainWindow()
    sys.exit(app.exec_())
