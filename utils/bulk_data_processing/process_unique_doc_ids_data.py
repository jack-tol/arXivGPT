import re
import pandas as pd
import logging
import csv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

record_pattern = re.compile(r'<record.*?>(.*?)</record>', re.DOTALL)
id_pattern = re.compile(r'<id>(.*?)</id>')

parsed_data = []
failed_records = []

def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip()

def extract_record_id(record):
    try:
        id_match = id_pattern.search(record)
        record_id = clean_text(id_match.group(1)) if id_match else None
        logger.debug(f"Extracted ID: {record_id}")
        return record_id
    except Exception as e:
        logger.error(f"Error extracting record ID: {e}")
        return None

def process_record(record):
    record_id = extract_record_id(record)
    if record_id:
        parsed_data.append({'document_id': record_id})
    else:
        failed_records.append(record)
        logger.warning("Failed to process record; added to failed records list.")

def process_xml_file(filepath):
    logger.info(f"Starting to process XML file: {filepath}")
    current_chunk = ''
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            for line in file:
                current_chunk += line
                if '</record>' in line:
                    for record in record_pattern.findall(current_chunk):
                        process_record(record)
                    current_chunk = ''
        logger.info("Finished processing XML file.")
    except Exception as e:
        logger.error(f"Error reading XML file: {e}")

def create_dataframe():
    logger.info("Creating DataFrame from parsed data.")
    df = pd.DataFrame(parsed_data)
    logger.info(f"DataFrame created with {len(df)} records.")
    return df

def clean_and_transform_data(df):
    logger.info("Starting data cleaning and transformation.")
    try:
        original_count = len(df)
        df['document_id'] = df['document_id'].astype(str).str.strip()
        df = df[df['document_id'].str.match(r'^\d$|^\d+')].copy()
        logger.info(f"Filtered invalid IDs; {len(df)} records remain out of {original_count}.")
        df = df.drop_duplicates(subset='document_id').reset_index(drop=True)
        logger.info(f"Removed duplicates; {len(df)} unique records remain.")
    except Exception as e:
        logger.error(f"Error during data cleaning and transformation: {e}")
    return df

def save_dataframe_to_csv(df, filename="unique_document_ids.csv"):
    try:
        logger.info(f"Saving DataFrame to CSV file: {filename}")
        with open(filename, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file, quoting=csv.QUOTE_ALL)
            writer.writerow(df.columns)
            for _, row in df.iterrows():
                writer.writerow(row.astype(str))
        logger.info(f"Data successfully saved to CSV file: {filename}")
        return filename
    except Exception as e:
        logger.error(f"Error saving DataFrame to CSV: {e}")
        return None

def display_failed_records():
    if failed_records:
        logger.warning(f"Failed to parse {len(failed_records)} records.")
        print("\nFailed to parse the following records:")
        for failed in failed_records:
            print(failed)
    else:
        logger.info("All records parsed successfully.")

if __name__ == "__main__":
    logger.info("Script started.")
    try:
        process_xml_file('arXiv_metadata_raw.xml')
        df = create_dataframe()
        df = clean_and_transform_data(df)
        output_filename = save_dataframe_to_csv(df)
        if output_filename:
            logger.info(f"Script completed successfully. Data saved to {output_filename}")
            print(f"Data parsed, cleaned, and saved to {output_filename}")
        display_failed_records()
    except Exception as e:
        logger.critical(f"Critical error encountered: {e}")