def split_text_file(input_file, lines_to_split, encoding='utf-8'):
    # Define the names for the output files
    output_file1 = 'arxiv_metadata_raw_pt_1.xml'
    output_file2 = 'arxiv_metadata_raw_pt_2.xml'

    try:
        # Open the input file for reading with the specified encoding
        with open(input_file, 'r', encoding=encoding) as file:
            lines = file.readlines()

        # Split the lines based on the specified number
        first_half = lines[:lines_to_split]  # Include the split line in the first half
        second_half = lines[lines_to_split:]  # Start the second half after the split line

        # Write the first half to the first output file
        with open(output_file1, 'w', encoding=encoding) as file:
            file.writelines(first_half)

        # Write the second half to the second output file
        with open(output_file2, 'w', encoding=encoding) as file:
            file.writelines(second_half)

        print(f"File successfully split into {output_file1} and {output_file2}")

    except FileNotFoundError:
        print(f"The file {input_file} does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
input_file = 'arXiv_metadata_raw.txt'
lines_to_split = 18498938  # Change this to the number of lines you want for the split
split_text_file(input_file, lines_to_split)