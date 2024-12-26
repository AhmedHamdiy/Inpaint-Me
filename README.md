# Exmplar Based Object Removal

This is an Implementation of exempler based object removal.
https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/criminisi_cvpr2003.pdf

We Have A NICE GUI if you want to run the code

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



# Deep Learning Based Object Removal


- Navigate to `DNNs` Folder
- Put your Images and Masks in in_lama Folder
- Run command `bash RunDNN.sh`
- Wait Till it finish and find your results in `out_lama`


# Contributors

<table>
	<tbody>
		<tr>
            <td align="center">
                <a href="https://github.com/Asmaa-204">
                    <img src="https://avatars.githubusercontent.com/u/130288326?v=4" width="100;" alt="Asmaa-204"/>
                    <br />
                    <sub><b>Asmaa Abozaid</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/AhmedHamdiy">
                    <img src="https://avatars.githubusercontent.com/u/111378492?v=4" width="100;" alt="AhmedHamdiy"/>
                    <br />
                    <sub><b>Ahmed Hamdy</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/akramhany">
                    <img src="https://avatars.githubusercontent.com/u/121282837?v=4" width="100;" alt="Abdelrahman Sami"/>
                    <br />
                    <sub><b>Abdelrahman Sami</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/shehab299">
                    <img src="https://avatars.githubusercontent.com/u/89648315?v=4" width="100;" alt="shehab299"/>
                    <br />
                    <sub><b>shehab khaled</b></sub>
                </a>
            </td>
		</tr>
	<tbody>
</table>


