# Exmplar Based Object Removal

## Prerequisites

Make sure you have Python installed on your system. It's recommended to use Python 3.8 or later.

## Step 1: Create a Virtual Environment

It is best practice to create a virtual environment to manage your dependencies:

```bash
python -m venv myenv
source myenv/bin/activate  # On Windows, use `myenv\Scripts\activate`
```

## Step 2: Install Dependencies

Use the following command to install the required dependencies:

```bash
pip install numpy==1.21.5 opencv-python==4.10.0.84 matplotlib==3.5.1
```

## Step 3: Verify the Installation

You can verify that the dependencies are installed correctly by running:

```bash
python -c "import numpy, cv2, matplotlib; print('All dependencies are installed successfully!')"
```

## Step 4: Run the GUI

To run the GUI application, execute the Python script responsible for the GUI. Replace `your_script.py` with the name of your script:

```bash
cd src
python gui.py
```

## Notes

- Ensure your virtual environment is activated while running the GUI.
- If you encounter any issues, ensure all required dependencies are correctly installed by running:

```bash
pip list
```

This will show you all installed packages and their versions.
