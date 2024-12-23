import tkinter as tk
import customtkinter as ctk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw
import os
from image_processing_algorithm import start  # Make sure to import your algorithm here


class ImageProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processor")
        self.root.geometry("800x1000")
        self.root.configure(fg_color="#2c3e50")  # Corrected method name

        # Main Frame
        self.main_frame = ctk.CTkFrame(self.root, width=800, height=600, corner_radius=10)
        self.main_frame.pack(padx=20, pady=20, fill="both", expand=True)

        # Title Label
        self.title_label = ctk.CTkLabel(self.main_frame, text="Image Processor", font=("Roboto Monto", 24), text_color="white")
        self.title_label.pack(pady=20)

        # Buttons (using customtkinter)
        self.select_btn = ctk.CTkButton(self.main_frame, text="Select Image", command=self.select_image, width=200, height=40, corner_radius=8)
        self.select_btn.pack(pady=10)

        self.process_btn = ctk.CTkButton(self.main_frame, text="Process Image", state=ctk.DISABLED, command=self.process_image, width=200, height=40, corner_radius=8)
        self.process_btn.pack(pady=10)

        self.save_btn = ctk.CTkButton(self.main_frame, text="Save Processed Image", state=ctk.DISABLED, command=self.save_image, width=200, height=40, corner_radius=8)
        self.save_btn.pack(pady=10)

        # Canvas to display and draw on the image
        self.canvas = tk.Canvas(self.main_frame, width=600, height=400, bg="#34495e")  # Slightly different background
        self.canvas.pack(pady=20)

        # Internal variables
        self.img_path = None
        self.original_image = None
        self.drawing_image = None
        self.processed_image = None
        self.output_dir = "output_images"  # Folder where processed images will be saved
        self.drawing = False

    def select_image(self):
        """ Open file dialog to select an image """
        filetypes = (("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff"), ("All files", "*.*"))
        self.img_path = filedialog.askopenfilename(filetypes=filetypes)
        
        if self.img_path:
            self.original_image = Image.open(self.img_path)
            self.display_image(self.original_image)
            self.initialize_drawing_image()
            self.process_btn.configure(state=ctk.NORMAL)  # Enable processing button after image is selected

    def initialize_drawing_image(self):
        """ Create a blank black image of the same size as the displayed image """
        if self.original_image:
            self.drawing_image = Image.new("RGB", self.original_image.size, "black")

    def display_image(self, img):
        """ Display image on the canvas """
        img.thumbnail((600, 400))  # Resize image to fit the canvas
        self.tk_image = ImageTk.PhotoImage(img)  # Convert to Tkinter-friendly format
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        self.canvas.image = self.tk_image  # Keep a reference to avoid garbage collection

        # Enable drawing on the canvas
        self.enable_drawing_mode()

    def enable_drawing_mode(self):
        """ Enable the user to draw on the canvas """
        self.canvas.bind("<Button-1>", self.start_drawing)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_drawing)

    def start_drawing(self, event):
        """ Start drawing """
        self.drawing = True
        self.last_x, self.last_y = event.x, event.y

    def draw(self, event):
        """ Draw on the canvas using circles for better continuity """
        if self.drawing and self.drawing_image:
            x, y = event.x, event.y

        # Draw a circle on the canvas
        radius = 5  # Adjust for marker size
        self.canvas.create_oval(
            x - radius, y - radius, x + radius, y + radius, 
            fill="white", outline="white"
        )

        # Draw a circle on the PIL image
        draw = ImageDraw.Draw(self.drawing_image)
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius], 
            fill="white", outline="white"
        )

        # Update last position for continuity, though it's less critical with circles
        self.last_x, self.last_y = x, y


    def stop_drawing(self, event):
        """ Stop drawing """
        self.drawing = False

    def process_image(self):
        """ Process the selected image using the algorithm """
        if not self.img_path:
            messagebox.showerror("Error", "No image selected!")
            return

        try:
            # Save the user's drawing
            drawing_path = os.path.join(self.output_dir, "user_drawing.png")
            os.makedirs(self.output_dir, exist_ok=True)
            self.drawing_image.save(drawing_path)

            # Call your image processing algorithm
            img = cv2.imread(self.img_path)
            start(img, drawing_path, self.output_dir)  # Pass original image and user-drawing path

            self.processed_image = os.path.join(self.output_dir, "final_result.jpg")

            # Display the processed image
            self.display_image(Image.open(self.processed_image))
            self.save_btn.configure(state=ctk.NORMAL)  # Enable save button after processing
        except Exception as e:
            print(e)
            messagebox.showerror("Error", f"Processing failed: {e}")

    def save_image(self):
        """ Save the processed image to a file """
        if not self.processed_image:
            messagebox.showerror("Error", "No processed image to save!")
            return

        save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
        if save_path:
            try:
                processed_img = Image.open(self.processed_image)
                processed_img.save(save_path)
                messagebox.showinfo("Success", "Image saved successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Saving failed: {e}")


if __name__ == "__main__":
    # Initialize the main window using CustomTkinter
    ctk.set_appearance_mode("System")  # Use system theme (light/dark based on user settings)
    ctk.set_default_color_theme("blue")  # Set default color theme (blue is a good choice)

    root = ctk.CTk()
    app = ImageProcessorApp(root)
    root.mainloop()
