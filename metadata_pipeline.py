import re
import logging
import pandas as pd
import csv
import asyncio
from sickle import Sickle
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
import os
from datetime import datetime
import pytz
from dotenv import load_dotenv

load_dotenv(override=True)
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class DataDownloader:
    def __init__(self, base_url):
        self.sickle = Sickle(base_url)

    async def download_records(self, from_date, until_date):
        loop = asyncio.get_event_loop()
        params = {'metadataPrefix': 'arXiv', 'from': from_date, 'until': until_date}

        try:
            records = await loop.run_in_executor(
                None, 
                lambda: list(self.sickle.ListRecords(**params))
            )
            raw_data = "".join(str(record.raw) for record in records)
            logging.info(f"Downloaded {len(records)} records from {from_date} to {until_date}.")
            return raw_data
        except Exception as e:
            logging.error(f"Failed to download records: {e}")
            raise


class RecordParser:
    def __init__(self):
        self.record_pattern = re.compile(r'<record.*?>(.*?)</record>', re.DOTALL)
        self.id_pattern = re.compile(r'<id>(.*?)</id>')
        self.title_pattern = re.compile(r'<title>(.*?)</title>', re.DOTALL)
        self.author_pattern = re.compile(r'<author>(.*?)</author>', re.DOTALL)
        self.keyname_pattern = re.compile(r'<keyname>(.*?)</keyname>')
        self.forenames_pattern = re.compile(r'<forenames>(.*?)</forenames>')
        self.parsed_data = []
        self.failed_records = []

    def extract_id_title_authors(self, record):
        id_match = self.id_pattern.search(record)
        title_match = self.title_pattern.search(record)
        record_id = id_match.group(1) if id_match else None
        title = title_match.group(1).replace('\n', ' ').strip() if title_match else None
        title = self.clean_text(title) if title else None
        authors = self.extract_authors(record)
        return record_id, title, authors

    def extract_authors(self, record):
        authors = []
        for author in self.author_pattern.findall(record):
            keyname = self.keyname_pattern.search(author)
            forenames = self.forenames_pattern.search(author)
            if keyname and forenames:
                authors.append(f"{forenames.group(1).strip()} {keyname.group(1).strip()}")
            elif keyname:
                authors.append(keyname.group(1).strip())
            elif forenames:
                authors.append(forenames.group(1).strip())
        return ", ".join(filter(None, authors))

    @staticmethod
    def clean_text(text):
        return re.sub(r'\s+', ' ', text).strip()

    def parse_records(self, raw_data):
        for record in self.record_pattern.findall(raw_data):
            record_id, title, authors = self.extract_id_title_authors(record)
            if record_id and title and authors:
                self.parsed_data.append({'id': record_id, 'title': title, 'authors': authors})
            else:
                self.failed_records.append(record)

    def display_failed_records(self):
        if self.failed_records:
            logging.warning("Some records failed to parse.")
        else:
            logging.info("All records parsed successfully.")


class DataProcessor:
    @staticmethod
    def clean_and_transform(parsed_data):
        df = pd.DataFrame(parsed_data)
        df = df[df['id'].str.match(r'^\d')].copy()
        df = df[~df['title'].str.contains(r'\\"', regex=True, na=False)]
        df = df[~df['authors'].str.contains(r'\\"', regex=True, na=False)]
        df['authors'] = df['authors'].apply(lambda x: ', '.join(x.split(', ')[:5]))
        df.rename(columns={'id': 'document_id'}, inplace=True)
        df.drop_duplicates(subset='document_id', inplace=True)
        df['title_by_authors'] = df['title'] + " by " + df['authors']
        return df[['document_id', 'title_by_authors']]

    @staticmethod
    def filter_existing(df, filename='unique_document_ids.csv'):
        try:
            existing_records = pd.read_csv(filename, dtype={'document_id': str})
            existing_records['document_id'] = existing_records['document_id'].str.strip()
            df['document_id'] = df['document_id'].astype(str).str.strip()
            df_filtered = df[~df['document_id'].isin(existing_records['document_id'])]
            updated_records = pd.concat([existing_records, df_filtered[['document_id']]]).drop_duplicates(subset='document_id')
            updated_records.sort_values(by='document_id', ascending=False).to_csv(filename, index=False, quoting=csv.QUOTE_ALL)
            logging.info(f"Updated unique document IDs saved to {filename}.")
        except FileNotFoundError:
            logging.warning(f"{filename} not found. Creating new file.")
            df[['document_id']].to_csv(filename, index=False, quoting=csv.QUOTE_ALL)
        return df_filtered


async def upload_to_pinecone(filtered_unique_records, batch_size=100):
    pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    index_name = "arxiv-rag-metadata"

    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

    index = pc.Index(index_name)

    for i in range(0, len(filtered_unique_records), batch_size):
        batch_df = filtered_unique_records.iloc[i:i + batch_size]
        texts = batch_df['title_by_authors'].tolist()

        response = await asyncio.to_thread(
            client.embeddings.create,
            input=texts,
            model="text-embedding-3-small"
        )
        embeddings = [item.embedding for item in response.data]

        vectors = [
            {
                "id": str(row['document_id']),
                "values": embedding,
                "metadata": {"document_id": str(row['document_id']), "text": row['title_by_authors']}
            }
            for row, embedding in zip(batch_df.to_dict(orient="records"), embeddings)
        ]
        
        await asyncio.to_thread(index.upsert, vectors=vectors)


def get_current_est_date():
    return datetime.now(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d')


async def main_pipeline():
    try:
        downloader = DataDownloader('http://export.arxiv.org/oai2')
        raw_data = await downloader.download_records(get_current_est_date(), get_current_est_date())

        parser = RecordParser()
        await asyncio.to_thread(parser.parse_records, raw_data)

        processor = DataProcessor()
        df = await asyncio.to_thread(processor.clean_and_transform, parser.parsed_data)
        filtered_unique_records = await asyncio.to_thread(processor.filter_existing, df)

        parser.display_failed_records()

        if not filtered_unique_records.empty:
            await upload_to_pinecone(filtered_unique_records)

        logging.info("Main pipeline completed successfully.")

    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        raise
