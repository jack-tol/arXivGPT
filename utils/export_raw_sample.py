input_file = 'arXiv_metadata_raw.xml'
output_file = 'arXiv_metadata_raw_100mb_subset.xml'
target_size = 100 * 1024 * 1024

read_size = 0

with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
    for line in infile:
        outfile.write(line)
        
        read_size += len(line.encode('utf-8'))
        
        if read_size >= target_size:
            break

print(f"{output_file} created with a subset of {read_size} bytes.")