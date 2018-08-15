from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import main

fileName = " "
cvImage = " "

def chooseImage():
    global fileName
    global cvImage
    fileName = filedialog.askopenfilename(filetypes=(("JPEG", "*.jpg"),
                                                     ("PNG", "*.png"),
                                                     ("All Files", "*.*")))
    photo1 = Image.open(fileName)

    cvImage = cv2.imread(fileName)
    height, width, ch = cvImage.shape


    photo = resizeImage(height, width, photo1)
    photo = ImageTk.PhotoImage(photo)

    label = Label(bottomFrame, image=photo)
    label.image = photo
    label.grid(row=0 , column=0)



def detectDigits():
    value = main.main(cvImage, 0)
    print("Value : ", value)
    textValue = Text(bottomFrame, width=100, height=3)
    textValue.insert(END, value)
    textValue.grid(row=1, column=0)

def detectCharacters():
    value = main.main(cvImage, 1)
    print("Value : ", value)
    textValue = Text(bottomFrame, width=100, height=3)
    textValue.insert(END, value)
    textValue.grid(row=1, column=0)

def resizeImage(height, width, img):
    if width>height :
        basewidth = 400
        wpercent = (basewidth / float(img.size[0]))
        hsize = int((float(img.size[1]) * float(wpercent)))
        img = img.resize((basewidth, hsize), Image.ANTIALIAS)
    else:
        baseheight = 400
        hpercent = (baseheight / float(img.size[1]))
        wsize = int((float(img.size[0]) * float(hpercent)))
        img = img.resize((wsize, baseheight), Image.ANTIALIAS)
    return img


root = Tk()
root.title("OCR")

#Top frame and its children (3 buttons)
topFrame = Frame(root, height=100, width=800)
topFrame.pack()
topFrame.pack_propagate(0)

selectButton = Button(topFrame, height=5, width=15, text="Select an image", bg="purple", fg="white", command=chooseImage)
selectButton.grid(row=0, column=0)

startButton = Button(topFrame, height=5, width=15, text="Detect digits",bg="green", fg="white", command=detectDigits)
startButton.grid(row=0, column=1)

startButton = Button(topFrame, height=5, width=15, text="Detect characters",bg="green", fg="white", command=detectCharacters)
startButton.grid(row=0, column=2)

exitButton = Button(topFrame, height=5, width=15, text="EXIT", bg="red", fg="white", command=quit)
exitButton.grid(row=0, column=3)

#Buttom frame and its children (selected image)
bottomFrame = Frame(root, height=400, width=800)
bottomFrame.pack()

#Down frame and its children (output text)
downFrame = Frame(root, height=200, width=800)
downFrame.pack()
downFrame.pack_propagate(0)

root.mainloop()

