import re
import pandas as pd
import csv

record_pattern = re.compile(r'<record.*?>(.*?)</record>', re.DOTALL)
id_pattern = re.compile(r'<id>(.*?)</id>')
title_pattern = re.compile(r'<title>(.*?)</title>', re.DOTALL)
author_pattern = re.compile(r'<author>(.*?)</author>', re.DOTALL)
keyname_pattern = re.compile(r'<keyname>(.*?)</keyname>')
forenames_pattern = re.compile(r'<forenames>(.*?)</forenames>')

parsed_data = []
failed_records = []

def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip()

def extract_id_title_authors(record):
    id_match = id_pattern.search(record)
    title_match = title_pattern.search(record)
    record_id = id_match.group(1) if id_match else None
    title = clean_text(title_match.group(1)) if title_match else None

    authors = []
    for author in author_pattern.findall(record):
        keyname = keyname_pattern.search(author)
        forenames = forenames_pattern.search(author)
        if keyname and forenames:
            authors.append(f"{forenames.group(1).strip()} {keyname.group(1).strip()}")
        elif keyname:
            authors.append(keyname.group(1).strip())
        elif forenames:
            authors.append(forenames.group(1).strip())

    authors_str = ", ".join(authors)
    return record_id, title, authors_str

def process_record(record):
    record_id, title, authors = extract_id_title_authors(record)
    if record_id and title and authors:
        parsed_data.append({'document_id': record_id, 'title': title, 'authors': authors})
    else:
        failed_records.append(record)

def process_xml_file(filepath):
    current_chunk = ''
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            current_chunk += line
            if '</record>' in line:
                for record in record_pattern.findall(current_chunk):
                    process_record(record)
                current_chunk = '' 

def create_dataframe():
    df = pd.DataFrame(parsed_data)
    return df

def clean_and_transform_data(df):
    df = df[df['document_id'].str.match(r'^\d')].copy()
    df = df[~df['title'].str.contains(r'\\"', regex=True, na=False)]
    df = df[~df['authors'].str.contains(r'\\"', regex=True, na=False)]
    
    df['authors'] = df['authors'].apply(lambda x: ', '.join(x.split(', ')[:5]))
    df['title_by_authors'] = df['title'] + " by " + df['authors']
    df = df[['document_id', 'title_by_authors']]
    return df

def save_dataframe_to_csv(df, filename='bulk_processed_parsed_data.csv'):
    try:
        with open(filename, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file, quoting=csv.QUOTE_ALL)
            writer.writerow(df.columns)
            for _, row in df.iterrows():
                writer.writerow(row.astype(str))
    except Exception as e:
        print(f"Error saving DataFrame to CSV: {e}")
    return filename

def display_failed_records():
    if failed_records:
        print("\nFailed to parse the following records:")
        for failed in failed_records:
            print(failed)
    else:
        print("All records parsed successfully.")

if __name__ == "__main__":
    process_xml_file('arXiv_metadata_raw.xml')  
    df = create_dataframe()
    df = clean_and_transform_data(df)
    output_filename = save_dataframe_to_csv(df)
    display_failed_records()
    print(f"Data parsed, cleaned, and saved to {output_filename}")