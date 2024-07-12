import warnings

# ignore all future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")
# from  MainRunner import  mainWindow
from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import *
import LR

class mainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.title = "Logistic Regression application"
        self.top = 300
        self.left = 600
        self.width = 410
        self.height = 600
        self.result_label = QLabel(self)
        self.iconName = "C:/Users/user/Documents/pythonprog/ML/MLGUI/assets/python.png"
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setWindowIcon(QtGui.QIcon(self.iconName))
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.splitSize = 20
        self.regularization = 1.0
        self.solver = "lbfgs"

        self.drawBrowser()
        self.drawSplit()
        self.drawRegularization()
        self.drawSolver()
        self.setDefault()
        self.drawFeatureScaling()
        self.drawPCAOption()
        self.drawPCAComponents()

        self.runButton = self.createButton("Run", self.runLR, 340, 280, 60, 30)

        self.result_label.setGeometry(10, 320, 400, 220)  # 设置结果显示的位置

        self.show()

    def setDefault(self):
        self.reg_button1.setChecked(True)
        self.featureScaling = "None"
        self.applyPCA = False
        self.pcaComponents = 2

    def drawBrowser(self):
        self.centralwidget = QWidget(self)
        self.csv_label = QLabel(self.centralwidget)
        self.csv_label.setGeometry(QtCore.QRect(10, 10, 80, 20))
        self.csv_label.setText("csv file: ")

        self.csv_lineEdit = QLineEdit(self)
        self.csv_lineEdit.setGeometry(QtCore.QRect(90, 10, 310, 30))
        self.browseButton = self.createButton("Browse", self.getFileName, 340, 50, 60, 30)

    def drawSplit(self):
        self.split_label = QLabel("test_data size(%): ", self)
        self.split_label.setStyleSheet('background-color: yellow')
        self.split_label.setGeometry(QtCore.QRect(40, 80, 110, 30))

        self.split_lineEdit = QLineEdit(self)
        self.split_lineEdit.setGeometry(QtCore.QRect(160, 80, 50, 30))
        self.split_lineEdit.setText(str(self.splitSize))

    def drawRegularization(self):
        self.reg_label = QLabel("Regularization: ", self)
        self.reg_label.setStyleSheet('background-color: yellow')
        self.reg_label.setGeometry(QtCore.QRect(40, 120, 110, 30))

        self.reg_group = QButtonGroup(self)
        self.reg_button1 = QRadioButton("1.0", self)
        self.reg_button1.setGeometry(QtCore.QRect(160, 120, 60, 30))
        self.reg_group.addButton(self.reg_button1)
        self.reg_button2 = QRadioButton("0.1", self)
        self.reg_button2.setGeometry(QtCore.QRect(230, 120, 60, 30))
        self.reg_group.addButton(self.reg_button2)
        self.reg_button3 = QRadioButton("0.01", self)
        self.reg_button3.setGeometry(QtCore.QRect(300, 120, 60, 30))
        self.reg_group.addButton(self.reg_button3)

    def drawSolver(self):
        self.solver_label = QLabel("Solver: ", self)
        self.solver_label.setStyleSheet('background-color: yellow')
        self.solver_label.setGeometry(QtCore.QRect(40, 160, 110, 30))

        self.solver_cb = QComboBox(self)
        self.solver_cb.setGeometry(QtCore.QRect(160, 160, 120, 30))
        self.solver_cb.addItems(["lbfgs", "liblinear", "sag", "saga"])
        self.solver_cb.currentIndexChanged.connect(self.solverChange)

    def drawFeatureScaling(self):
        self.scaling_label = QLabel("Feature Scaling: ", self)
        self.scaling_label.setStyleSheet('background-color: yellow')
        self.scaling_label.setGeometry(QtCore.QRect(40, 200, 110, 30))

        self.scaling_cb = QComboBox(self)
        self.scaling_cb.setGeometry(QtCore.QRect(160, 200, 120, 30))
        self.scaling_cb.addItems(["None", "StandardScaler", "MinMaxScaler"])
        self.scaling_cb.currentIndexChanged.connect(self.scalingChange)

    def drawPCAOption(self):
        self.pca_label = QLabel("Apply PCA: ", self)
        self.pca_label.setStyleSheet('background-color: yellow')
        self.pca_label.setGeometry(QtCore.QRect(40, 240, 110, 30))

        self.pca_cb = QCheckBox(self)
        self.pca_cb.setGeometry(QtCore.QRect(160, 240, 20, 30))
        self.pca_cb.stateChanged.connect(self.pcaChange)

    def drawPCAComponents(self):
        self.pca_components_label = QLabel("PCA Components: ", self)
        self.pca_components_label.setStyleSheet('background-color: yellow')
        self.pca_components_label.setGeometry(QtCore.QRect(40, 280, 110, 30))
        self.pca_components_label.setVisible(False)  # Initially hidden

        self.pca_components_lineEdit = QLineEdit(self)
        self.pca_components_lineEdit.setGeometry(QtCore.QRect(160, 280, 50, 30))
        self.pca_components_lineEdit.setText(str(self.pcaComponents))
        self.pca_components_lineEdit.setVisible(False)

    def scalingChange(self):
        self.featureScaling = self.scaling_cb.currentText()

    def solverChange(self):
        self.solver = self.solver_cb.currentText()

    def pcaChange(self):
        self.applyPCA = self.pca_cb.isChecked()
        self.pca_components_label.setVisible(self.applyPCA)
        self.pca_components_lineEdit.setVisible(self.applyPCA)

    def getFileName(self):
        fileName, _ = QFileDialog.getOpenFileName(self, 'Single File',
                                                  'C:/Users/user/Documents/pythonprog/ML/MLGUI/scripts', '*.csv')
        self.csv_lineEdit.setText(fileName)
        self.fileName = self.csv_lineEdit.text()

    def runLR(self):
        if self.fileName != "":
            self.splitSize = int(self.split_lineEdit.text())
            self.featureScaling = self.scaling_cb.currentText()
            self.applyPCA = self.pca_cb.isChecked()
            self.pcaComponents = int(self.pca_components_lineEdit.text())
            self.regularization = float(self.reg_group.checkedButton().text())

            if self.splitSize <= 40:
                self.results = LR.run(self.fileName, self.splitSize, self.regularization,
                                                           self.solver, self.featureScaling, self.applyPCA, self.pcaComponents)
            else:
                pass
        else:
            pass

        self.result_label.setText("Result: " + str(self.results))

    def createButton(self, text, fun, x, y, l, w):
        pushButton = QPushButton(text, self)
        pushButton.setGeometry(QtCore.QRect(x, y, l, w))
        pushButton.clicked.connect(fun)
        return pushButton


def Main():
    m = mainWindow()
    m.show()
    return m


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    mWin = mainWindow()
    sys.exit(app.exec_())
