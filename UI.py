#%%
from scipy.ndimage import convolve
import tkinter as tk
from PIL import Image
from PIL import ImageDraw, ImageTk
from tkinter import ttk
from PIL import ImageTk
from tkinter import filedialog
from ttkthemes import ThemedStyle
import cv2
import numpy as np
import tkinter.filedialog
import tkinter
import tkinter.messagebox
import customtkinter
from tensorflow import keras
from utils import class_labels


customtkinter.set_appearance_mode("Dark")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

once = True
img_screenshot = None

class App(customtkinter.CTk):
    original_image = None
    changed_image = None
    edited_image = None
    def __init__(self):
        super().__init__()

        # configure window
        self.title("FinalProject.py")
        self.geometry(f"{1280}x{720}") 

        self.logo_image = customtkinter.CTkImage(Image.open("images/iconshow.png"))
        self.iconbitmap("images/iconshow.ico")
        
        self.model = keras.models.load_model("models/pretrained_model.keras")

        # Đọc hình ảnh từ file
        # image = Image.open("image/save1.png")
        # export_img=Image.open("image/export.png")
        # draw_img=Image.open("image/draw.png")
        open_img=Image.open("images/open.png")
        # reset_img=Image.open("image/reset.png")
        # avatar_img=Image.open("image/iconshow1.png")
        
        # Chuyển đổi hình ảnh thành đối tượng PhotoImage
        
        # photo1 = ImageTk.PhotoImage(image)
        # export=ImageTk.PhotoImage(export_img)
        open=ImageTk.PhotoImage(open_img)
        # reset=ImageTk.PhotoImage(reset_img)
        # draw=ImageTk.PhotoImage(draw_img)
        # avatar=ImageTk.PhotoImage(avatar_img)

        self.info_frame=customtkinter.CTkFrame(self,fg_color="transparent")

        self.info_frame.grid(row=3,column=3,padx=20,pady=(10,10))

        # Banner
        self.member = customtkinter.CTkLabel(self.info_frame, text="Member:",font=customtkinter.CTkFont(size=16, weight="bold"), compound="left",anchor="w")
        self.member.grid(row=0, column=0, sticky="nsew")
        self.info = customtkinter.CTkLabel(self.info_frame, text="Le Hoang Lam",font=customtkinter.CTkFont(size=14, weight="normal"),compound="left",anchor="w")
        self.info.grid(row=1, column=1, sticky="nsew")
        self.info1 = customtkinter.CTkLabel(self.info_frame, text="Nguyen Hoang Nhan",font=customtkinter.CTkFont(size=14, weight="normal"),compound="left",anchor="w")
        self.info1.grid(row=2, column=1, sticky="nsew")
        self.info2 = customtkinter.CTkLabel(self.info_frame, text="Nguyen Viet Anh",font=customtkinter.CTkFont(size=14, weight="normal"),compound="left",anchor="w")
        self.info2.grid(row=3, column=1, sticky="nsew")
        self.info3 = customtkinter.CTkLabel(self.info_frame, text="Le Y Thien",font=customtkinter.CTkFont(size=14, weight="normal"),compound="left",anchor="w")
        self.info3.grid(row=4, column=1,  sticky="nsew")
        self.info4 = customtkinter.CTkLabel(self.info_frame, text="Nguyen Hoang Nhat Nam",font=customtkinter.CTkFont(size=14, weight="normal"),compound="left",anchor="w")
        self.info4.grid(row=5, column=1,  sticky="nsew")


        # configure grid layout (4x4)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=0)
        # self.grid_columnconfigure(3, weight=1)
        self.grid_rowconfigure((0, 1, 2), weight=1)
        
         # create sidebar frame with widgets
        self.sidebar_frame = customtkinter.CTkFrame(self, width=140, corner_radius=0)
        # self.grid_columnconfigure(0,minsize=500) #set width of first column
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew") #"nsew" có nghĩa là widget con sẽ giãn ra theo phương ngang và dọc của ô đặt của widget cha.
        self.sidebar_frame.grid_rowconfigure(7, weight=1) #5 is max row in 1 grid column

        self.sidebar_button_open_image = customtkinter.CTkButton(self.sidebar_frame, image=open,text="Open Image",command=self.open_file, text_color	 ="white")
        self.sidebar_button_open_image.grid(row=1, column=0, padx=20, pady=10)

        
        self.sidebar_button_predict = customtkinter.CTkButton(self.sidebar_frame, text="Predict",command=self.handle_predict,text_color	 ="white")
        self.sidebar_button_predict.grid(row=2, column=0, padx=20, pady=10)


        # Image frame
        self.image_frame = customtkinter.CTkFrame(self)
        self.image_frame.grid(row=0, column=1, padx=20, pady=(10,10))

        self.img_initial = cv2.imread("images/open_image.jpg")
        self.img_initial = cv2.cvtColor(self.img_initial, cv2.COLOR_BGR2RGB)

        # Resize the image to a fixed size
        img_height, img_width, _ = self.img_initial.shape
        if img_height > img_width:
            new_height = 640
            new_width = int((img_width / img_height) * new_height)
        else:
            new_width = 480
            new_height = int((img_height / img_width) * new_width)
        # self.original_img_size = (new_width, new_height)
        
        self.img_initial = Image.fromarray(self.img_initial)
        self.img_initial = ImageTk.PhotoImage(self.img_initial)

        self.original_img_lbl = tk.Label(self.image_frame,image=self.img_initial)
        self.original_img_lbl.grid(row=0,column=0,padx=20,pady=(10,10))

        # handle image frame
        self.handle_img_frame=customtkinter.CTkFrame(self,fg_color="transparent")
        
        self.handle_img_frame.grid(row=0,column=3,padx=20,pady=(10,10))
    
    def handle_predict(self):
        self.predict() 

    # Resize Image 
    def resize_image(self,img, width, height):
        if img is None:
            return None

        img_height, img_width = img.shape[:2]

        # Calculate the aspect ratio of the image
        aspect_ratio = img_width / img_height

        if width is not None and height is not None:
            # Resize the image to the specified width and height
            resized_img = cv2.resize(img, (width, height))
        elif width is not None:
            # Calculate the new height based on the aspect ratio and the desired width
            new_height = int(width / aspect_ratio)
            resized_img = cv2.resize(img, (width, new_height))
        elif height is not None:
            # Calculate the new width based on the aspect ratio and the desired height
            new_width = int(height * aspect_ratio)
            resized_img = cv2.resize(img, (new_width, height))
        else:
            # Return None if no width or height is specified
            return None

        return resized_img



    def show_image(self, *args):
        self.changed_image = self.resize_image(self.changed_image,640, 480)
        self.changed_image = Image.fromarray(self.changed_image)
        self.img_initial_copy = self.changed_image.copy()
        self.changed_image = ImageTk.PhotoImage(self.changed_image)
        self.original_img_lbl.configure(image=self.changed_image)
        self.original_img_lbl.image = self.changed_image


    def open_file(self):
        global once
        once = True
        img_file = filedialog.askopenfilename() 
        if img_file  != '':     
            self.img_path = img_file 
            # self.log_Scale.set(self.log_Scale.get() + 1)
            # self.log_Scale.set(self.log_Scale.get() - 1)

            self.original_image = cv2.imread(self.img_path)
            self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            self.changed_image = self.original_image
            self.temp_image = self.changed_image
            self.show_image()
        else:
            return 0

    def predict(self, *args):
        target_size = (28, 28)
        self.original_image = cv2.imread(self.img_path)
        self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        self.temp_image = self.original_image
        self.original_image = cv2.resize(self.original_image, target_size)
        test_images = np.array([self.original_image])  # Wrap image in a list
        y_proba = self.model.predict(test_images).round(2)  # return probabilities (output of output neurons)
        y_pred = np.argmax(y_proba, axis=1)  # return class with highest probability

        predicted_labels = [class_labels[idx] for idx in y_pred]

        # Convert the image from BGR to RGB (for displaying with matplotlib)
        image_with_text = cv2.cvtColor(self.temp_image, cv2.COLOR_RGB2BGR)
        
        # Define the class label text and position
        label_text = f"{predicted_labels[0]}"
        text_position = (5, 100)  # Adjust position as needed
        text_color = (0, 0, 255) 
        background_color = (255, 255, 255)
        font_scale = 1

        # Get the size of the text
        (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)

        # Define the background rectangle
        background_rectangle = ((text_position[0], text_position[1] - text_height - 5),
                                (text_position[0] + text_width + 5, text_position[1] + baseline + 5))
        # Draw the filled rectangle
        cv2.rectangle(image_with_text, background_rectangle[0], background_rectangle[1], background_color, -1)

        # Draw the text on the image
        cv2.putText(image_with_text, label_text, text_position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1, cv2.LINE_AA)                # Convert the image back to RGB (for displaying with tkinter)
        image_with_text = cv2.cvtColor(image_with_text, cv2.COLOR_BGR2RGB)
        self.changed_image = image_with_text
        self.show_image()

if __name__ == "__main__":
    app = App()
    app.mainloop()

# %%
