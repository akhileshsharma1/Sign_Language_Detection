import os
import cv2
import tkinter as tk
from tkinter import messagebox

class DataCollectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Camera Data Collection")
        self.root.geometry("800x600")
        
        # Camera and data collection settings
        self.DATA_DIR = './data'
        self.number_of_classes = 3
        self.dataset_size = 100
        self.camera_index = 0
        
        # Create directory if not exists
        if not os.path.exists(self.DATA_DIR):
            os.makedirs(self.DATA_DIR)
        
        # Open camera
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            messagebox.showerror("Error", f"Cannot open camera at index {self.camera_index}")
            return
        
        # Create UI elements
        self.create_ui()
        
        # Start video stream
        self.update_frame()
    
    def create_ui(self):
        # Video display frame
        self.video_frame = tk.Label(self.root)
        self.video_frame.pack(padx=10, pady=10)
        
        # Buttons frame
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10)
        
        # Create buttons for each class
        for j in range(self.number_of_classes):
            btn = tk.Button(
                button_frame, 
                text=f'Collect Class {j}', 
                command=lambda x=j: self.collect_data_for_class(x),
                bg='#4CAF50', 
                fg='white'
            )
            btn.pack(side=tk.LEFT, padx=5)
        
        # Status label
        self.status_label = tk.Label(
            self.root, 
            text="Ready to collect data", 
            fg='#333', 
            font=('Arial', 12)
        )
        self.status_label.pack(pady=10)
    
    def update_frame(self):
        # Capture frame-by-frame
        ret, frame = self.cap.read()
        if ret:
            # Convert frame to PhotoImage
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = cv2.resize(frame_rgb, (640, 480))
            img = self.convert_to_tkinter_image(img)
            
            # Update video frame
            self.video_frame.configure(image=img)
            self.video_frame.image = img
        
        # Repeat every 10 ms
        self.root.after(10, self.update_frame)
    
    def convert_to_tkinter_image(self, frame):
        from PIL import Image, ImageTk
        return ImageTk.PhotoImage(image=Image.fromarray(frame))
    
    def collect_data_for_class(self, class_index):
        # Create class directory
        class_dir = os.path.join(self.DATA_DIR, str(class_index))
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
        
        # Collect images
        for i in range(self.dataset_size):
            ret, frame = self.cap.read()
            if ret:
                # Save image
                image_path = os.path.join(class_dir, f'{i}.jpg')
                cv2.imwrite(image_path, frame)
                
                # Update status
                self.status_label.config(
                    text=f'Collecting Class {class_index}: {i+1}/{self.dataset_size}'
                )
                self.root.update()
        
        # Completion message
        self.status_label.config(
            text=f'Completed collecting data for Class {class_index}'
        )

# Run the application
def main():
    root = tk.Tk()
    app = DataCollectionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()