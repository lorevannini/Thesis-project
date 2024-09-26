


import numpy as np
import re
from PIL import Image
import pytesseract
import csv





def image_to_csv(image_path, output_csv_path):
    # Open the image file
    image = Image.open(image_path)

    # Specify the number of rows and columns in the table
    num_rows = 12
    num_columns = 6

    # Set the PSM value based on the table structure
    psm_value = 6  # Use --psm 6 for a uniform block of text

    # Define the whitelist for each column
    char_whitelist = [
        'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghilmnopqrstuvzwyxk'
        'ABCDEFGHIJKLMNOPQRSTUVWXYZ',  # Uppercase letters
        '0123456789eE-.',
        '0123456789eE-.',
        '0123456789eE-.',
        '0123456789eE-.',
        '0123456789eE-.',
        '0123456789eE-.'
    ]

    # Example configuration options
    custom_config = f'--oem 3 --psm {psm_value} -c tessedit_char_whitelist={"+".join(char_whitelist)}'

    # Use pytesseract to extract text from the image with custom configuration
    text = pytesseract.image_to_string(image, config=custom_config)

    # Process the extracted text based on the expected table structure
    table_data = [row.split() for row in text.split('\n') if row.strip()]

    # Post-process the extracted text using regular expressions
    for i in range(1, len(table_data)):
        for j in range(1, len(table_data[i])):
            if re.match(r'^\d+\.\d+$', table_data[i][j]):  # Check if the element is a decimal number
                table_data[i][j] = format(float(table_data[i][j]), '.3f')  # Add leading zeros to the number

    # Write the data to a CSV file
    with open(output_csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(table_data)

if __name__ == "__main__":
    # Replace 'input_image.png' with the path to your screenshot image
    input_image_path = 'yield.png'

    # Replace 'output_table.csv' with the desired CSV output path
    output_csv_path = 'output_table.csv'

    image_to_csv(input_image_path, output_csv_path)







