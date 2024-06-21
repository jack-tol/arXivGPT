# metadata_pipeline.py

import os
import aiofiles
import aiofiles.os
import asyncio
import logging
import pandas as pd
from sickle import Sickle
from sickle.oaiexceptions import NoRecordsMatch
from requests.exceptions import HTTPError, RequestException
from datetime import datetime, timedelta
import pytz
import xml.etree.ElementTree as ET
import ast
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

logging.basicConfig(filename='combined_log.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def run_metadata_pipeline():
    """Run the daily metadata pipeline."""
    logger.info('Starting daily task.')
    current_date = get_current_est_date()
    logger.info(f'Current EST date: {current_date}')
    from_date = current_date
    until_date = current_date
    try:
        await download_metadata(from_date, until_date)
        await remove_line_breaks_and_wrap('arxiv_metadata.xml', 'arxiv_metadata_cleaned.xml')
        df = await parse_xml_to_dataframe('arxiv_metadata_cleaned.xml')
        await aiofiles.os.remove('arxiv_metadata_cleaned.xml')
        await aiofiles.os.remove('arxiv_metadata.xml')
        df_processed = process_arxiv_metadata(df)

        if not df_processed.empty:
            logger.info('DataFrame is Not Empty. Proceeding with Pinecone Upload.')
            embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
            index_name = "arxiv-rag-metadata"
            vector_store = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings_model)
            await upload_to_pinecone(df_processed, vector_store)
        else:
            logger.error("DataFrame is empty. Skipping upload.")
    except NoRecordsMatch:
        logger.warning("Metadata is not available for today, trying tomorrow instead.")
    except Exception as e:
        logger.error(f"An error occurred during the daily task: {e}")
    logger.info('Daily task completed.')

def get_current_est_date():
    """Get the current date in EST."""
    est = pytz.timezone('US/Eastern')
    return datetime.now(est).strftime('%Y-%m-%d')

async def download_metadata(from_date, until_date):
    """Download metadata from arXiv for the specified date range."""
    connection = Sickle('http://export.arxiv.org/oai2')
    logger.info('Getting papers...')
    params = {'metadataPrefix': 'arXiv', 'from': from_date, 'until': until_date, 'ignore_deleted': True}
    data = connection.ListRecords(**params)
    logger.info('Papers retrieved.')

    iters = 0
    errors = 0

    async with aiofiles.open('arxiv_metadata.xml', 'a+', encoding="utf-8") as f:
        while True:
            try:
                record = await asyncio.to_thread(lambda: next(data, None))
                if record is None:
                    logger.info(f'Metadata For The Specified Period, {from_date} - {until_date} Downloaded.')
                    return
                await f.write(record.raw)
                await f.write('\n')
                errors = 0
                iters += 1
                if iters % 1000 == 0:
                    logger.info(f'{iters} Processing Attempts Made Successfully.')

            except HTTPError as e:
                await handle_http_error(e)

            except RequestException as e:
                logger.error(f'RequestException: {e}')
                raise

            except Exception as e:
                errors += 1
                logger.error(f'Unexpected error: {e}')
                if errors > 5:
                    logger.critical('Too many consecutive errors, stopping the harvester.')
                    raise

async def handle_http_error(e):
    """Handle HTTP errors during metadata download."""
    if e.response.status_code == 503:
        retry_after = e.response.headers.get('Retry-After', 30)
        logger.warning(f"HTTPError 503: Server Busy. Retrying After {retry_after} Seconds.")
        await asyncio.sleep(int(retry_after))
    else:
        logger.error(f'HTTPError: Status code {e.response.status_code}')
        raise e

async def remove_line_breaks_and_wrap(input_file: str, output_file: str):
    """Remove line breaks and wrap the content in the XML file."""
    logger.info(f'Removing line breaks and wrapping content in {input_file}.')
    async with aiofiles.open(input_file, 'r', encoding='utf-8') as infile, aiofiles.open(output_file, 'w', encoding='utf-8') as outfile:
        await outfile.write("<records>")
        async for line in infile:
            cleaned_line = line.replace('\n', '').replace('\r', '')
            await outfile.write(cleaned_line)
        await outfile.write("</records>")
    logger.info(f'Content wrapped and saved to {output_file}.')

async def parse_xml_to_dataframe(input_file: str):
    """Parse the XML file to a pandas DataFrame."""
    logger.info(f'Parsing XML file {input_file} to DataFrame.')

    def extract_records(file_path):
        context = ET.iterparse(file_path, events=('end',))
        for event, elem in context:
            if elem.tag == '{http://www.openarchives.org/OAI/2.0/}record':
                header = elem.find('oai:header', namespaces)
                metadata = elem.find('oai:metadata', namespaces)
                arxiv = metadata.find('arxiv:arXiv', namespaces) if metadata is not None else None
                
                record_data = {
                    'datestamp': header.find('oai:datestamp', namespaces).text if header.find('oai:datestamp', namespaces) is not None else '',
                    'created': arxiv.find('arxiv:created', namespaces).text if arxiv is not None and arxiv.find('arxiv:created', namespaces) is not None else '',
                    'id': arxiv.find('arxiv:id', namespaces).text if arxiv is not None and arxiv.find('arxiv:id', namespaces) is not None else '',
                    'authors': [{"keyname": author.find('arxiv:keyname', namespaces).text if author.find('arxiv:keyname', namespaces) is not None else '', "forenames": author.find('arxiv:forenames', namespaces).text if author.find('arxiv:forenames', namespaces) is not None else ''} for author in arxiv.findall('arxiv:authors/arxiv:author', namespaces)] if arxiv is not None else [],
                    'title': arxiv.find('arxiv:title', namespaces).text if arxiv is not None and arxiv.find('arxiv:title', namespaces) is not None else ''
                }
                yield record_data
                elem.clear()
    
    namespaces = {
        'oai': 'http://www.openarchives.org/OAI/2.0/',
        'arxiv': 'http://arxiv.org/OAI/arXiv/'
    }
    
    records = await asyncio.to_thread(lambda: list(extract_records(input_file)))
    
    df = pd.DataFrame(records)
    logger.info(f'Parsed XML to DataFrame with {len(df)} records.')
    return df

def process_arxiv_metadata(df: pd.DataFrame):
    """Process arXiv metadata DataFrame."""
    logging.info('Processing DataFrame Metadata.')
    
    df.rename(columns={
        'datestamp': 'last_edited',
        'id': 'document_id',
        'created': 'date_created'
    }, inplace=True)

    df.drop_duplicates(subset='document_id', inplace=True)
    
    df.replace(to_replace=r'\s\s+', value=' ', regex=True, inplace=True)
    
    df['document_id'] = df['document_id'].astype(str)
    df = df[df['document_id'].str.match(r'^\d')]
    
    df.loc[:, 'last_edited'] = pd.to_datetime(df['last_edited'])
    df.loc[:, 'date_created'] = pd.to_datetime(df['date_created'])
    df.loc[:, 'authors'] = df['authors'].astype(str)
    df.loc[:, 'title'] = df['title'].astype(str)
    
    def parse_authors(authors_str):
        authors_list = ast.literal_eval(authors_str)
        authors_list = authors_list[:5]
        formatted_authors = [f"{author['forenames']} {author['keyname']}" for author in authors_list]
        return ', '.join(formatted_authors)
    
    df.loc[:, 'authors'] = df['authors'].apply(parse_authors)
    
    df = df[df['last_edited'] == (df['date_created'] + pd.Timedelta(days=1))]
    
    df['title_by_authors'] = df.apply(lambda row: f"{row['title']} by {row['authors']}", axis=1)
    
    df.drop(columns=['last_edited', 'date_created', 'authors', 'title'], inplace=True)
    
    df.sort_values(by='document_id', ascending=False, inplace=True)
    
    logging.info('DataFrame Processing Complete.')
    return df

async def upload_to_pinecone(df, vector_store):
    """Upload processed data to Pinecone vector store."""
    num_papers = len(df)
    logger.info(f'Preparing to Upload {num_papers} Research Papers to Pinecone Vector Store.')
    
    texts = df['title_by_authors'].tolist()
    metadatas = df[['document_id']].to_dict(orient='records')
    await asyncio.to_thread(vector_store.add_texts, texts=texts, metadatas=metadatas)
    
    logger.info(f'Successfully Uploaded {num_papers} Research Papers to Pinecone Vector Store.')
    logger.info('Upload to Pinecone Complete.')

async def daily_metadata_task():
    """Run the daily metadata pipeline at 11 PM EST."""
    est = pytz.timezone('US/Eastern')
    now = datetime.now(est)
    target_time = datetime.now(est).replace(hour=23, minute=0, second=0, microsecond=0)
    
    if now > target_time:
        target_time += timedelta(days=1)
    
    wait_time = (target_time - now).total_seconds()
    await asyncio.sleep(wait_time)
    
    while True:
        await run_metadata_pipeline()
        
        target_time += timedelta(days=1)
        wait_time = (target_time - datetime.now(est)).total_seconds()
        await asyncio.sleep(wait_time)