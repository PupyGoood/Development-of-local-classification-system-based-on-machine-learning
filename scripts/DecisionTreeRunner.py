import warnings
# ignore all future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")


from PyQt5 import QtGui,QtCore
from PyQt5.QtWidgets import *
import DecesionTree

class mainWindow(QMainWindow):
    
    def __init__(self):
        super().__init__()
        self.title = "Decision Tree application"
        self.top = 300
        self.left = 600
        self.width = 410
        self.height = 600
        self.result_label = QLabel(self)
        self.iconName = "C:/Users/user/Documents/pythonprog/ML/MLGUI/assets/python.png"
        # self.backgroundImagePath = r"background.png"  # 更改为你的背景图片路径
        self.initUI()
        
        
    def initUI(self):
        # # 设置背景颜色为淡绿色
        # self.setStyleSheet("background-color: lightgreen;")
        # # 设置背景图片
        # self.setBackgroundImage()

        self.setWindowTitle(self.title)
        self.setWindowIcon(QtGui.QIcon(self.iconName))
        self.setGeometry(self.left, self.top, self.width, self.height)
        
        self.splitSize = 20

        self.drawBrowser()
        self.drawSplit()                
        self.drawCriterion()
        self.drawSplitter()        
        self.setDefault()
        self.drawFeatureScaling()
        self.drawPCAOption()
        self.drawPCAComponents()

        self.svmButton = self.createButton("Run",self.runSVM,340, 295,60, 30)

        self.result_label.setGeometry(10, 320, 400, 220)  # 设置结果显示的位置

        self.show()
        
        
    def setDefault(self):
        # self.fileName = ""
        self.crit_button1.setChecked(True)
        self.splitter_button1.setChecked(True)
        self.splitter = 'best'
        self.criterion = 'gini'
        self.featureScaling = "None"
        self.applyPCA = False
        self.pcaComponents = 2
    
    def drawBrowser(self):
        self.centralwidget = QWidget(self) 
        self.csv_label = QLabel(self.centralwidget) 
        self.csv_label.setGeometry(QtCore.QRect(10, 10, 80, 20))
        self.csv_label.setText("csv file: ")
        
        self.csv_lineEdit = QLineEdit(self)
        self.csv_lineEdit.setGeometry(QtCore.QRect(90,10,310,30))
        self.svmButton = self.createButton("Browse",self.getFileName,340, 50,60, 30)

    def drawSplit(self):
        self.split_label = QLabel("test_data size(%): ",self)
        self.split_label.setStyleSheet('background-color: yellow')              
        self.split_label.setGeometry(QtCore.QRect(40,80, 110, 30))
        
        self.split_lineEdit = QLineEdit(self)
        self.split_lineEdit.setGeometry(QtCore.QRect(160,80,50,30))
        self.split_lineEdit.setText(str(self.splitSize))
        
    def drawCriterion(self):
        self.crit_label = QLabel("Criterion: ",self)
        self.crit_label.setStyleSheet('background-color: yellow')              
        self.crit_label.setGeometry(QtCore.QRect(40,120, 80, 30))
        

        self.crit_group = QButtonGroup(self)
        self.crit_button1 = QRadioButton("gini",self)
        self.crit_button1.setGeometry(QtCore.QRect(160,125,80,20))
        self.crit_group.addButton(self.crit_button1)
        self.crit_button2 = QRadioButton("entropy",self)
        self.crit_button2.setGeometry(QtCore.QRect(250,125,80,20))
        self.crit_group.addButton(self.crit_button2)
        
    def drawSplitter(self):
        self.splitter_label = QLabel("Splitter: ",self)
        self.splitter_label.setStyleSheet('background-color: yellow')              
        self.splitter_label.setGeometry(QtCore.QRect(40,160, 80, 30))
        
        self.splitter_group = QButtonGroup(self)
        self.splitter_button1 = QRadioButton("best",self)
        self.splitter_button1.setGeometry(QtCore.QRect(160,165,80,20))
        self.splitter_group.addButton(self.splitter_button1)
        self.splitter_button2 = QRadioButton("random",self)
        self.splitter_button2.setGeometry(QtCore.QRect(250,165,80,20))
        self.splitter_group.addButton(self.splitter_button2)

    def drawFeatureScaling(self):
        self.scaling_label = QLabel("Feature Scaling: ", self)
        self.scaling_label.setStyleSheet('background-color: yellow')
        self.scaling_label.setGeometry(QtCore.QRect(40, 205, 110, 30))

        self.scaling_cb = QComboBox(self)
        self.scaling_cb.setGeometry(QtCore.QRect(160, 205, 120, 30))
        self.scaling_cb.addItems(["None", "StandardScaler", "MinMaxScaler"])
        self.scaling_cb.currentIndexChanged.connect(self.scalingChange)

    def drawPCAOption(self):
        self.pca_label = QLabel("Apply PCA: ", self)
        self.pca_label.setStyleSheet('background-color: yellow')
        self.pca_label.setGeometry(QtCore.QRect(40, 255, 110, 30))

        self.pca_cb = QCheckBox(self)
        self.pca_cb.setGeometry(QtCore.QRect(160, 255, 20, 30))
        self.pca_cb.stateChanged.connect(self.pcaChange)

    def drawPCAComponents(self):
        self.pca_components_label = QLabel("PCA Components: ", self)
        self.pca_components_label.setStyleSheet('background-color: yellow')
        self.pca_components_label.setGeometry(QtCore.QRect(40, 295, 110, 30))
        self.pca_components_label.setVisible(False)  # Initially hidden

        self.pca_components_lineEdit = QLineEdit(self)
        self.pca_components_lineEdit.setGeometry(QtCore.QRect(160, 295, 50, 30))
        self.pca_components_lineEdit.setText(str(self.pcaComponents))
        self.pca_components_lineEdit.setVisible(False)

    def scalingChange(self):
        self.featureScaling = self.scaling_cb.currentText()

    def pcaChange(self):
        self.applyPCA = self.pca_cb.isChecked()
        self.pca_components_label.setVisible(self.applyPCA)
        self.pca_components_lineEdit.setVisible(self.applyPCA)
    
    def getFileName(self):
        fileName, _ = QFileDialog.getOpenFileName(self, 'Single File', 'C:/Users/user/Documents/pythonprog/ML/MLGUI/scripts' , '*.csv')
        self.csv_lineEdit.setText(fileName)
        self.fileName = self.csv_lineEdit.text()
        #print(self.fileName)

    def runSVM(self):
        # print("--------TRAINING--------")
        if self.fileName != "":
            self.splitSize = int(self.split_lineEdit.text())
            self.featureScaling = self.scaling_cb.currentText()
            self.applyPCA = self.pca_cb.isChecked()
            self.pcaComponents = int(self.pca_components_lineEdit.text())

            if self.splitSize <=40:
                if self.splitter_button1.isChecked() is False:
                    self.splitter = 'random'
                if self.crit_button1.isChecked() is False:
                    self.criterion = 'entropy'
                # print("Test percentage: ",self.splitSize)
                # print("Criterion: ",self.criterion)
                # print("Splitter: ",self.splitter)
                self.results = DecesionTree.run(self.fileName,self.splitSize,self.criterion,self.splitter, self.featureScaling, self.applyPCA, self.pcaComponents)
            else:
                pass# print("cannot train on such small dataset")
        else: 
            pass# print("incorrect file name!")            
        # print("--------SUCCESSFUL--------")
        

        # QMessageBox.about(self,"Results:", self.results)
        self.result_label.setText("Result: " + str(self.results))

    def createButton(self,text,fun,x,y,l,w):
        pushButton = QPushButton(text,self) 
        pushButton.setGeometry(QtCore.QRect(x,y,l,w))
        pushButton.clicked.connect(fun)
        return pushButton
    # def setBackgroundImage(self):
    #     # 创建 QLabel 以显示背景图片
    #     self.background_label = QLabel(self)
    #     self.background_label.setGeometry(410, 0, self.width, self.height)
    #
    #     # 加载背景图片
    #     background_pixmap = QtGui.QPixmap(self.backgroundImagePath)
    #     # 检查图片是否加载成功
    #     if background_pixmap.isNull():
    #         print(f"Failed to load background image from {self.backgroundImagePath}")
    #         return
    #     # 设置背景图片
    #     self.background_label.setPixmap(background_pixmap)
    #     self.background_label.setScaledContents(True)  # 确保图片适应标签大小
    #     # 设置透明度
    #     self.background_label.setWindowOpacity(0.2)  # 调整透明度值，范围为0.0到1.0
    #
    #     # 确保背景图片在窗口最底层
    #     self.background_label.lower()
        
def Main():
    m = mainWindow()
    m.show()
    return m
    
if __name__=="__main__":   
    import sys
    app = QApplication(sys.argv)
    mWin = mainWindow()
    sys.exit(app.exec_())