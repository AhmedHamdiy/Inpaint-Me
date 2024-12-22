import tkinter as tk
import customtkinter as ctk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
from image_processing_algorithm import start  # Make sure to import your algorithm here


class ImageProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processor")
        self.root.geometry("800x600")
        self.root.configure(bg_color="#2c3e50")

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

        # Canvas to display the image
        self.canvas = ctk.CTkCanvas(self.main_frame, width=500, height=400, bg="#34495e")  # Slightly different background
        self.canvas.pack(pady=20)

        # Internal variables
        self.img_path = None
        self.processed_image = None
        self.output_dir = "output_images"  # Folder where processed images will be saved
        self.selected_point = None
    
    def select_image(self):
        """ Open file dialog to select an image """
        filetypes = (("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff"), ("All files", "*.*"))
        self.img_path = filedialog.askopenfilename(filetypes=filetypes)
        
        if self.img_path:
            self.display_image(self.img_path)
            self.process_btn.configure(state=ctk.NORMAL)  # Enable processing button after image is selected
            
    
    def display_image(self, img_path):
        """ Display image on the canvas """
        img = Image.open(img_path)
        img.thumbnail((500, 500))  # Resize image to fit the canvas

        # Convert the image to a Tkinter-friendly format
        img_tk = ImageTk.PhotoImage(img)
        
        # Display the image on the canvas
        self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        self.canvas.image = img_tk  # Keep a reference to avoid garbage collection

    def process_image(self):
        """ Process the selected image using the algorithm """
        if not self.img_path:
            messagebox.showerror("Error", "No image selected!")
            return

        if not self.selected_point:
            messagebox.showerror("Error", "No point selected!")
            return

        # Get the coordinates from the selected point
        x, y = self.selected_point

        # Call your image processing algorithm
        try:
            img = cv2.imread(self.img_path)
            start(img, x, y, self.output_dir)  # Pass the directory to save the output images
            self.processed_image = os.path.join(self.output_dir, "filled_region.png")

            # After processing, display the filled region image (not the watershed result)
            self.display_image(self.processed_image)
            self.save_btn.configure(state=ctk.NORMAL)  # Enable save button after processing
        except Exception as e:
            messagebox.showerror("Error", f"Processing failed: {e}")

    def save_image(self):
        """ Save the processed image to a file """
        if not self.processed_image:
            messagebox.showerror("Error", "No processed image to save!")
            return

        # Ask the user where to save the processed image
        save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
        if save_path:
            try:
                processed_img = Image.open(self.processed_image)
                processed_img.save(save_path)
                messagebox.showinfo("Success", "Image saved successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Saving failed: {e}")
    
    def on_canvas_click(self, event):
        """ Handle mouse click event to select a point on the canvas """
        if self.img_path:
            # Get the canvas coordinates
            canvas_x, canvas_y = event.x, event.y
            self.selected_point = (canvas_x, canvas_y)

            # Draw a small circle on the canvas to mark the point
            self.canvas.create_oval(canvas_x-3, canvas_y-3, canvas_x+3, canvas_y+3, outline="red", width=2)

            messagebox.showinfo("Point Selected", f"Point selected at: ({canvas_x}, {canvas_y})")

    def enable_point_selection(self):
        """ Enable point selection mode """
        self.canvas.bind("<Button-1>", self.on_canvas_click)


if __name__ == "__main__":
    # Initialize the main window using CustomTkinter
    ctk.set_appearance_mode("System")  # Use system theme (light/dark based on user settings)
    ctk.set_default_color_theme("blue")  # Set default color theme (blue is a good choice)

    root = ctk.CTk()
    app = ImageProcessorApp(root)

    # Enable point selection mode after an image is selected
    app.select_btn.configure(command=lambda: [app.select_image(), app.enable_point_selection()])


    root.mainloop()
