import time
import pandas as pd
import lxml.etree as ET
from sickle import Sickle
from requests.exceptions import HTTPError, RequestException
import logging
import os

from langchain_pinecone import PineconeVectorStore
from langchain_cohere import CohereEmbeddings

# Configure logging
def setup_logging():
    logging.basicConfig(filename='arxiv_download.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_metadata(from_date, until_date):
    connection = Sickle('http://export.arxiv.org/oai2')
    logging.info('Getting papers...')
    params = {'metadataPrefix': 'arXiv', 'from': from_date, 'until': until_date, 'ignore_deleted': True}
    data = connection.ListRecords(**params)
    logging.info('Papers retrieved.')

    iters = 0
    errors = 0

    with open('arXiv_metadata_raw.xml', 'a+', encoding="utf-8") as f:
        while True:
            try:
                record = next(data).raw
                f.write(record)
                f.write('\n')
                errors = 0
                iters += 1
                if iters % 1000 == 0:
                    logging.info(f'{iters} Processing Attempts Made Successfully.')

            except HTTPError as e:
                handle_http_error(e)

            except RequestException as e:
                logging.error(f'RequestException: {e}')
                raise

            except StopIteration:
                logging.info(f'Metadata For The Specified Period, {from_date} - {until_date} Downloaded.')
                break

            except Exception as e:
                errors += 1
                logging.error(f'Unexpected error: {e}')
                if errors > 5:
                    logging.critical('Too many consecutive errors, stopping the harvester.')
                    raise

def handle_http_error(e):
    if e.response.status_code == 503:
        retry_after = e.response.headers.get('Retry-After', 30)
        logging.warning(f"HTTPError 503: Server busy. Retrying after {retry_after} seconds.")
        time.sleep(int(retry_after))
    else:
        logging.error(f'HTTPError: Status code {e.response.status_code}')
        raise e

def parse_xml_to_df(xml_file, batch_size=100000):
    ns = {
        'oai': 'http://www.openarchives.org/OAI/2.0/',
        'arxiv': 'http://arxiv.org/OAI/arXiv/'
    }
    records = []
    count = 0
    all_records = []
    buffer = ""

    with open(xml_file, 'r', encoding='utf-8') as f:
        for line in f:
            buffer += line.strip()
            if buffer.endswith('</record>'):
                try:
                    elem = ET.fromstring(buffer)
                    data = {}
                    header = elem.find('oai:header', ns)
                    data['identifier'] = header.find('oai:identifier', ns).text
                    data['datestamp'] = header.find('oai:datestamp', ns).text
                    data['setSpec'] = [elem.text for elem in header.findall('oai:setSpec', ns)]
                    
                    metadata = elem.find('oai:metadata/arxiv:arXiv', ns)
                    data['id'] = metadata.find('arxiv:id', ns).text
                    data['created'] = metadata.find('arxiv:created', ns).text
                    data['updated'] = metadata.find('arxiv:updated', ns).text if metadata.find('arxiv:updated', ns) is not None else None
                    data['authors'] = [
                        (author.find('arxiv:keyname', ns).text if author.find('arxiv:keyname', ns) is not None else None,
                         author.find('arxiv:forenames', ns).text if author.find('arxiv:forenames', ns) is not None else None)
                        for author in metadata.findall('arxiv:authors/arxiv:author', ns)
                    ]
                    data['title'] = metadata.find('arxiv:title', ns).text
                    data['categories'] = metadata.find('arxiv:categories', ns).text
                    data['comments'] = metadata.find('arxiv:comments', ns).text if metadata.find('arxiv:comments', ns) is not None else None
                    data['report_no'] = metadata.find('arxiv:report-no', ns).text if metadata.find('arxiv:report-no', ns) is not None else None
                    data['journal_ref'] = metadata.find('arxiv:journal-ref', ns).text if metadata.find('arxiv:journal-ref', ns) is not None else None
                    data['doi'] = metadata.find('arxiv:doi', ns).text if metadata.find('arxiv:doi', ns) is not None else None
                    data['license'] = metadata.find('arxiv:license', ns).text if metadata.find('arxiv:license', ns) is not None else None
                    data['abstract'] = metadata.find('arxiv:abstract', ns).text.strip() if metadata.find('arxiv:abstract', ns) is not None else None
                    
                    records.append(data)
                    count += 1

                    if count >= batch_size:
                        all_records.extend(records)
                        records = []
                        count = 0

                except ET.XMLSyntaxError as e:
                    print(f"Error parsing record: {e}")

                buffer = ""

    if records:
        all_records.extend(records)

    df = pd.DataFrame(all_records)
    return df

def preprocess_dataframe(df):
    # drops all columns which are not relevant for this application
    df = df[['datestamp', 'id', 'created', 'authors', 'title']].copy()

    # renames fields to more accurately align with what they represent 
    df.rename(columns={
        'datestamp': 'last_edited',
        'id': 'document_id',
        'created': 'date_created'
    }, inplace=True)
    
    # converts the 'title' and 'authors' fields to the string datatype
    df.loc[:, 'title'] = df['title'].astype(str)
    df.loc[:, 'authors'] = df['authors'].astype(str)

    # replace any double spaces with single spaces in the 'title' and 'authors' fields
    df.loc[:, 'title'] = df['title'].str.replace('  ', ' ', regex=True)
    df.loc[:, 'authors'] = df['authors'].str.replace('  ', ' ', regex=True)

    # checks to make sure that each record within the 'document_id' field starts with an integer
    df = df[df['document_id'].str.match('^\d')]

    # removes any line breaks from the 'title' field
    df.loc[:, 'title'] = df['title'].str.replace('\n', '', regex=True)
    
    # removes any square brackets, parentheses, single quotes, and double quotes from the 'authors' field
    df.loc[:, 'authors'] = df['authors'].str.replace('[\[\]\'"()]', '', regex=True)

    # defines a flip_names function which changes the 'authors' format from 'Lastname, Firstname' to 'Firstname, Lastname' and limits the amount of authors to 10
    def flip_names(authors):
        author_list = authors.split(', ')
        flipped_authors = []
        for i in range(0, len(author_list), 2):
            if i + 1 < len(author_list):
                flipped_authors.append(f"{author_list[i + 1]} {author_list[i]}")
        return ', '.join(flipped_authors[:10])

    # applies the flip_names function to each record in the 'authors' field
    df.loc[:, 'authors'] = df['authors'].apply(flip_names)
    
    # converts the 'last_edited' and 'date_created' fields to the datatime datatype
    df.loc[:, 'last_edited'] = pd.to_datetime(df['last_edited'])
    df.loc[:, 'date_created'] = pd.to_datetime(df['date_created'])
    
    # filter for rows where 'last_edited' is the same as 'date_created' + 1 day
    df = df[df['last_edited'] == df['date_created'] + pd.Timedelta(days=1)]
    
    # concatenate 'title' and 'authors' into a new field 'title_by_authors'
    df.loc[:, 'title_by_authors'] = df['title'] + ' by ' + df['authors']
    
    # drops the original 'title', 'authors', 'date_created', and 'last_edited' fields
    df.drop(['title', 'authors', 'date_created', 'last_edited'], axis=1, inplace=True)
    
    # exports the processed dataframe to a csv file
    df.to_csv('metadata_processed.csv', index=False)
    
    return df

def upload_to_pinecone(df, vector_store):
    texts = df['title_by_authors'].tolist()
    metadatas = df[['document_id']].to_dict(orient='records')
    vector_store.add_texts(texts=texts, metadatas=metadatas)

def main():
    setup_logging()
    xml_file = 'arXiv_metadata_raw.xml'
    df = parse_xml_to_df(xml_file, batch_size=100000)
    df = preprocess_dataframe(df)
    if not df.empty:
        PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
        embeddings_model = CohereEmbeddings(model="embed-english-v3.0")
        index_name = "arxiv-rag-metadata"
        vector_store = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings_model)
        upload_to_pinecone(df, vector_store)
    else:
        logging.error("DataFrame is empty. Skipping upload.")

if __name__ == '__main__':
    main()