import functools
import json
import sys
import random
import numpy as np

import joblib
from PyQt6.QtCore import Qt, QCoreApplication
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QHBoxLayout, QListWidgetItem, QWidget, QPushButton, QLabel, QApplication, QMainWindow, \
    QListWidget, QLineEdit, QVBoxLayout

from training import bag_of_words

data = json.loads(open('intents.json').read())
words = joblib.load('words.joblib')
classes = joblib.load('classes.joblib')

def pred_class(text, vocab, labels):
    loaded_model = joblib.load('fifatmodel.model')
    #model = load_model('fifatmodel.model')
    bow = bag_of_words(text, vocab)
    result = loaded_model.predict(np.array([bow]))[0]  # Extracting probabilities
    thresh = 0.5
    y_pred = [[indx, res] for indx, res in enumerate(result) if res > thresh]
    y_pred.sort(key=lambda x: x[1], reverse=True)  # sorting by values of probability in decreasing order
    return_list = []
    print(f"bag of word = {bow}")
    print(f"Predicted result = {result}")
    print(f"Probability = {y_pred}")

    for r in y_pred:
        return_list.append(labels[r[0]])  # contains labels(tags) for highest probability
    return return_list


def get_response(intents_list, intents_json):
    if len(intents_list) == 0:
        result = "Sorry! I don't understand."
        options = []
    else:
        tag = intents_list[0]
        list_of_intents = intents_json["intents"]
        for i in list_of_intents:
            if i["tag"] == tag:
                result = random.choice(i["responses"])
                options = []
                if i["options"] is not None:
                    options = i["options"]
                break
    return result, options


# Subclass QMainWindow to customize your application's main window pyqt6
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("FIFAT")

        app = QApplication([])
        self.label = QLabel("FIFAT Chat bot")
        self.text_area = QListWidget()
        self.button = QPushButton("Send")
        #self.button.resize(120, 150)
        self.button.clicked.connect(self.update)
        self.text_area.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.message = QLineEdit()
        #self.message.setStyleSheet("height: 27pt")
        self.message.resize(120,150)
        self.message.returnPressed.connect(self.button.click)
        userinputHorizontalLayout = QHBoxLayout()
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.text_area)

        userinputHorizontalLayout.addWidget(self.message)
        userinputHorizontalLayout.addWidget(self.button)

        layout.addLayout(userinputHorizontalLayout)
        #self.text_area.addItem("FIFAT : Hello I am FIFAT, How can i help you?")
        self.setBotResponse("FIFAT : Hello I am FIFAT, How can i help you?")
        window = QWidget()
        window.setLayout(layout)
        window.show()
        window.setGeometry(10,6,600, 600)
        app.exec()

    # action method
    def handle_input(self, message):

        self.set_user_message(message)

        intents = pred_class(message.lower(), words, classes)
        result, options = get_response(intents, data)

        self.setBotResponse(result)

        #self.text_area.setItemWidget(self,botHBox)
        #self.text_area.addItem("FIFAT : " + result)

        if options is not None:
            widgetLayout = QHBoxLayout()
            self.widget = QWidget()
            self.itemN = QListWidgetItem()
            for items in options:
                # create QListWidget and add the buttons into it

                Button1 = QPushButton(self)
                Button1.setText(items)
                # Button1.clicked.connect(lambda: print(Button1.text()))
                Button1.clicked.connect(functools.partial(self.select_option, items))

                widgetLayout.addWidget(Button1)

            self.widget.setLayout(widgetLayout)
            self.text_area.addItem(self.itemN)
            self.itemN.setSizeHint(self.widget.sizeHint())
            self.text_area.setItemWidget(self.itemN, self.widget)


        self.text_area.addItem("")
        self.message.setText("")

        self.text_area.scrollToBottom()

        if message == "Bye":
            #close the application
            print("Bye")
            #self.destroy()
            #self.close()
            #time.sleep(1000)
            sys.exit(0)


    def setBotResponse(self,result):
        botResponseBox = QHBoxLayout()

        botHBox = QVBoxLayout()
        botWidget = QWidget()
        botItemN = QListWidgetItem()

        label = QLabel(self)
        label.resize(50, 50)

        w = label.width();
        h = label.height();

        pixmap = QPixmap('images/bot.png')

        # label.setPixmap(pixmap)
        label.setPixmap(pixmap.scaled(w, h));

        botHBox.addWidget(label)
        BotNameLabel = QLabel(self)
        BotNameLabel.setText("FIFAT")
        BotNameLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        botHBox.addWidget(BotNameLabel)

        botResponseBox.addLayout(botHBox)
        BotReponseLabel = QLabel(self)
        # BotReponseLabel.setStyleSheet("background-color:red;")
        BotReponseLabel.setText(result)
        # hbox.addWidget(pathBox, alignment=QtCore.Qt.AlignCenter)
        botResponseBox.addWidget(BotReponseLabel, alignment=Qt.AlignmentFlag.AlignRight)
        botResponseBox.setAlignment(Qt.AlignmentFlag.AlignLeft)

        botWidget.setLayout(botResponseBox)
        self.text_area.addItem(botItemN)
        botItemN.setSizeHint(botWidget.sizeHint())
        self.text_area.setItemWidget(botItemN, botWidget)

    def select_option(self, x):
        self.handle_input(x)

    def update(self):
        self.handle_input(self.message.text())

    def set_user_message(self, message):
        # self.text_area.addItem("User : " + message)
        widgetLayout = QHBoxLayout()
        # userMessageLabel = QLabel(self)
        # #userMessageLabel.setStyleSheet("background-color:cyan;")
        # userMessageLabel.setAlignment(Qt.AlignmentFlag.AlignRight)
        # userMessageLabel.setText(message+" : User")
        # widgetLayout.addWidget(userMessageLabel)
        # self.UserListWidgetItemN = QListWidgetItem()
        # self.userMessageWidget = QWidget()
        # self.userMessageWidget.setLayout(widgetLayout)
        # #widgetLayout.addStretch(1)
        # self.text_area.addItem(self.UserListWidgetItemN)
        # self.UserListWidgetItemN.setSizeHint(self.userMessageWidget.sizeHint())
        # #self.UserListWidgetItemN.setTextAlignment(Qt.AlignmentFlag.AlignLeft)
        # self.text_area.setItemWidget(self.UserListWidgetItemN, self.userMessageWidget)


        #self.text_area.addItem(self.userMessageWidget,Qt.AlignmentFlag.AlignLeft)
        # self.itemN.setSizeHint(self.widget.sizeHint())

        # self.text_area.setItemWidget(self.itemN, self.widget)

        userVBox = QVBoxLayout()
        userWidget = QWidget()
        userItemN = QListWidgetItem()

        ulabel = QLabel(self)
        ulabel.resize(50, 50)

        w = ulabel.width();
        h = ulabel.height();

        pixmap = QPixmap('images/user.png')

        # label.setPixmap(pixmap)
        ulabel.setPixmap(pixmap.scaled(w, h));

        userVBox.addWidget(ulabel)
        userNameLabel = QLabel(self)
        userNameLabel.setText("You")
        userNameLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        userVBox.addWidget(userNameLabel)


        BotReponseLabel = QLabel(self)
        # BotReponseLabel.setStyleSheet("background-color:red;")
        BotReponseLabel.setText(message)
        #BotReponseLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        #BotReponseLabel.setStyleSheet("padding:0px 0px 0px 50px")
        # hbox.addWidget(pathBox, alignment=QtCore.Qt.AlignCenter)
        widgetLayout.addWidget(BotReponseLabel, alignment=Qt.AlignmentFlag.AlignRight)
        widgetLayout.setAlignment(Qt.AlignmentFlag.AlignRight)
        widgetLayout.addLayout(userVBox)

        userWidget.setLayout(widgetLayout)
        self.text_area.addItem(userItemN)
        userItemN.setSizeHint(userWidget.sizeHint())
        self.text_area.setItemWidget(userItemN, userWidget)


# You need one (and only one) QApplication instance per application.
# Pass in sys.argv to allow command line arguments for your app.
# If you know you won't use command line arguments QApplication([]) works too.
app = QApplication(sys.argv)

# Create a Qt widget, which will be our window.
window = MainWindow()
window.show()  # IMPORTANT!!!!! Windows are hidden by default.

# Start the event loop.
app.exec()