import os
import arxiv
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
import re
import ast
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
import chainlit as cl
from openai import AsyncOpenAI
from chainlit.context import context
from chainlit.user_session import user_session
from langchain_text_splitters import RecursiveCharacterTextSplitter
from aiohttp import ClientSession

# Configure logging
logging.basicConfig(filename='combined_log.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Function to initialize embedding model
def initialize_embeddings():
    """Initialize the OpenAI embedding model."""
    logger.info("Initializing OpenAI embeddings...")
    return OpenAIEmbeddings(model="text-embedding-3-small")

# Function to initialize vector stores
def initialize_vector_stores(embedding_model):
    """Initialize Pinecone vector stores for metadata and chunks."""
    logger.info("Initializing Pinecone vector stores...")
    metadata_vector_store = PineconeVectorStore.from_existing_index(embedding=embedding_model, index_name="arxiv-rag-metadata")
    chunks_vector_store = PineconeVectorStore.from_existing_index(embedding=embedding_model, index_name="arxiv-rag-chunks")
    return metadata_vector_store, chunks_vector_store

# Function to initialize text splitter
def initialize_text_splitter():
    """Initialize the recursive character text splitter."""
    logger.info("Initializing text splitter...")
    return RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False
    )

async def send_actions():
    """Send action options to the user."""
    actions = [
        cl.Action(name="ask_followup_question", value="followup_question", description="Uses The Previously Retrieved Context", label="Ask a Follow-Up Question"),
        cl.Action(name="ask_new_question", value="new_question", description="Retrieves New Context", label="Ask a New Question About the Same Paper"),
        cl.Action(name="ask_about_new_paper", value="new_paper", description="Ask About A Different Paper", label="Ask About a Different Paper")
    ]
    await cl.Message(content="### Please Select One of the Following Options:", actions=actions).send()

@cl.on_stop
async def on_stop():
    """Handle session stop event to clean up tasks."""
    streaming_task = user_session.get('streaming_task')
    if streaming_task:
        streaming_task.cancel()
        await send_actions()
    user_session.set('streaming_task', None)
    logger.info("Session stopped and streaming task cleaned up.")

@cl.on_chat_start
async def main():
    """Main function to start the chat session."""
    # Schedule the daily metadata task
    asyncio.create_task(daily_metadata_task())
    
    embedding_model = initialize_embeddings()
    metadata_vector_store, chunks_vector_store = initialize_vector_stores(embedding_model)
    text_splitter = initialize_text_splitter()

    user_session.set('embedding_model', embedding_model)
    user_session.set('metadata_vector_store', metadata_vector_store)
    user_session.set('chunks_vector_store', chunks_vector_store)
    user_session.set('text_splitter', text_splitter)
    user_session.set('current_document_id', None)

    text_content = """## Welcome to the arXiv Research Paper Learning Supplement

This system is connected to the live stream of papers being uploaded to arXiv daily.

### Instructions

1. **Enter the Title**: Start by entering the title of the research paper you wish to learn more about.
2. **Select a Paper**: Choose a paper from the list of retrieved papers.
3. **Database Check**: The system will check if the research paper exists in the research paper content database.
   - If it exists, you will be prompted to enter your question.
   - If it does not exist, the program will download the paper to the database and then ask you to enter your question.
4. **Read the Answer**: After reading the answer, you will have the following options:
   - Ask a follow-up question.
   - Ask a new question about the same paper.
   - Ask a new question about a different paper.

### Get Started
When You're Ready, Follow the First Step Below.
"""
    await cl.Message(content=text_content).send()

    await ask_initial_query()

async def ask_initial_query():
    """Prompt the user to enter the title of the research paper."""
    res = await cl.AskUserMessage(content="### Please Enter the Title of the Research Paper You Wish to Learn More About:", timeout=3600).send()
    if res:
        initial_query = res['output']
        metadata_vector_store = user_session.get('metadata_vector_store')
        logger.info(f"Searching for metadata with query: {initial_query}")
        search_results = metadata_vector_store.similarity_search(query=initial_query, k=5)
        logger.info(f"Metadata search results: {search_results}")
        selected_doc_id = await select_document_from_results(search_results)
        if selected_doc_id:
            logger.info(f"Document selected with ID: {selected_doc_id}")
            user_session.set('current_document_id', selected_doc_id)
            chunks_exist = await do_chunks_exist_already(selected_doc_id)
            if not chunks_exist:
                await process_and_upload_chunks(selected_doc_id)
            else:
                await ask_user_question(selected_doc_id)

async def ask_user_question(document_id):
    """Prompt the user to enter a question about the selected document."""
    logger.info(f"Asking user question for document_id: {document_id}")
    context, user_query = await process_user_query(document_id)
    if context and user_query:
        task = asyncio.create_task(query_openai_with_context(context, user_query))
        user_session.set('streaming_task', task)
        await task

async def select_document_from_results(search_results):
    if not search_results:
        await cl.Message(content="No Search Results Found").send()
        return None

    message_content = "### Please Enter the Number Corresponding to Your Desired Paper:\n"
    message_content += "| No. | Paper Title | Doc. ID |\n"
    message_content += "|-----|-------------|---------|\n"

    for i, doc in enumerate(search_results, start=1):
        page_content = doc.page_content
        document_id = doc.metadata['document_id']
        message_content += f"| {i} | {page_content} | {document_id} |\n"

    await cl.Message(content=message_content).send()

    while True:
        res = await cl.AskUserMessage(content="", timeout=3600).send()
        if res:
            try:
                user_choice = int(res['output']) - 1
                if 0 <= user_choice < len(search_results):
                    selected_doc_id = search_results[user_choice].metadata['document_id']
                    selected_paper_title = search_results[user_choice].page_content
                    await cl.Message(content=f"\n**You selected:** {selected_paper_title}").send()
                    return selected_doc_id
                else:
                    await cl.Message(content="\nInvalid Selection. Please enter a valid number from the list.").send()
            except ValueError:
                await cl.Message(content="\nInvalid input. Please enter a number.").send()
        else:
            await cl.Message(content="\nNo selection made. Please enter a valid number from the list.").send()

async def do_chunks_exist_already(document_id):
    """Check if chunks for the document already exist."""
    chunks_vector_store = user_session.get('chunks_vector_store')
    filter = {"document_id": {"$eq": document_id}}
    test_query = chunks_vector_store.similarity_search(query="Chunks Existence Check", k=1, filter=filter)
    logger.info(f"Chunks existence check result for document_id {document_id}: {test_query}")
    return bool(test_query)

async def download_pdf(session, document_id, url, filename):
    """Download the PDF file asynchronously."""
    logger.info(f"Downloading PDF for document_id: {document_id} from URL: {url}")
    async with session.get(url) as response:
        if response.status == 200:
            async with aiofiles.open(filename, mode='wb') as f:
                await f.write(await response.read())
            logger.info(f"Successfully downloaded PDF for document_id: {document_id}")
        else:
            logger.error(f"Failed to download PDF for document_id: {document_id}, status code: {response.status}")
            raise Exception(f"Failed to download PDF: {response.status}")

async def process_and_upload_chunks(document_id):
    """Download, process, and upload chunks of the document."""
    await cl.Message(content="#### It seems that paper isn't currently in our database. Don't worry, we are currently downloading, processing, and uploading it. This will only take a few moments.").send()
    await asyncio.sleep(2)

    try:
        # Create an async session for downloading
        async with ClientSession() as session:
            paper = await asyncio.to_thread(next, arxiv.Client().results(arxiv.Search(id_list=[str(document_id)])))
            url = paper.pdf_url
            filename = f"{document_id}.pdf"
            await download_pdf(session, document_id, url, filename)

        # Load and split the PDF into pages
        loader = PyPDFLoader(filename)
        pages = await asyncio.to_thread(loader.load)

        # Process and split pages into chunks
        text_splitter = user_session.get('text_splitter')
        content = []
        found_references = False

        for page in pages:
            if found_references:
                break
            page_text = page.page_content
            if "references" in page_text.lower():
                content.append(page_text.split("References")[0])
                found_references = True
            else:
                content.append(page_text)

        full_content = ''.join(content)
        chunks = text_splitter.split_text(full_content)

        # Ensure embedding model is initialized
        embedding_model = user_session.get('embedding_model')
        if not embedding_model:
            raise ValueError("Embedding model not initialized")

        # Upload chunks to Pinecone asynchronously
        chunks_vector_store = user_session.get('chunks_vector_store')
        await asyncio.to_thread(
            chunks_vector_store.from_texts,
            texts=chunks,
            embedding=embedding_model,
            metadatas=[{"document_id": document_id} for _ in chunks],
            index_name="arxiv-rag-chunks"
        )

        # Clean up the downloaded PDF file asynchronously
        await aiofiles.os.remove(filename)
        logger.info(f"Successfully processed and uploaded chunks for document_id: {document_id}")

        # Ensure the transition to asking a question happens
        await ask_user_question(document_id)

    except Exception as e:
        logger.error(f"Error processing and uploading chunks for document_id {document_id}: {e}")
        await cl.Message(content="#### An error occurred during processing. Please try again.").send()
        return

async def process_user_query(document_id):
    """Process the user's query about the document."""
    res = await cl.AskUserMessage(content="### Please Enter Your Question:", timeout=3600).send()
    if res:
        user_query = res['output']
        context = []
        chunks_vector_store = user_session.get('chunks_vector_store')
        filter = {"document_id": {"$eq": document_id}}
        attempts = 5  # Number of attempts to check for context
        for attempt in range(attempts):
            search_results = chunks_vector_store.similarity_search(query=user_query, k=15, filter=filter)
            logger.info(f"Context retrieval attempt {attempt + 1}: Found {len(search_results)} results")
            context = [doc.page_content for doc in search_results]
            if context:
                break
            logger.info(f"No context found, retrying... (attempt {attempt + 1}/{attempts})")
            await asyncio.sleep(2)  # Wait before retrying

        logger.info(f"User query processed. Context length: {len(context)}, User Query: {user_query}")
        return context, user_query
    return None, None

async def query_openai_with_context(context, user_query):
    """Query OpenAI with the context and user query."""
    if not context:
        await cl.Message(content="No context available to answer the question.").send()
        return

    client = AsyncOpenAI()

    settings = {
        "model": "gpt-4o",
        "temperature": 0.3,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
    }

    message_history = [
        {"role": "system", "content": """
         Your job is to answer the user's query using only the provided context.
         Be detailed and long-winded. Format your responses in markdown formatting, making good use of headings,
         subheadings, ordered and unordered lists, and regular text formatting such as bolding of text and italics.
         Sometimes the equations retrieved from the context will be formatted improperly and in an incompatible format
         for correct LaTeX rendering. Therefore, if you ever need to provide equations, make sure they are
         formatted properly using LaTeX, wrapping the equation in single dollar signs ($) for inline equations
         or double dollar signs ($$) for bigger, more visual equations. Keep your answer grounded in the facts
         of the provided context. If the context does not contain the facts needed to answer the user's query, return:
         "I do not have enough information available to accurately answer the question."
         """},
        {"role": "user", "content": f"Context: {context}"},
        {"role": "user", "content": f"Question: {user_query}"}
    ]

    msg = cl.Message(content="")
    await msg.send()

    async def stream_response():
        stream = await client.chat.completions.create(messages=message_history, stream=True, **settings)
        async for part in stream:
            if token := part.choices[0].delta.content:
                await msg.stream_token(token)

    streaming_task = asyncio.create_task(stream_response())
    user_session.set('streaming_task', streaming_task)

    try:
        await streaming_task
    except asyncio.CancelledError:
        streaming_task.cancel()
        return

    await msg.update()

    await send_actions()

# Action callbacks
@cl.action_callback("ask_followup_question")
async def handle_followup_question(action):
    """Handle follow-up question action."""
    logger.info("Follow-up question button clicked.")
    current_document_id = user_session.get('current_document_id')
    if current_document_id:
        context, user_query = await process_user_query(current_document_id)
        if context and user_query:
            logger.info(f"Processing follow-up question for document_id: {current_document_id}")
            task = asyncio.create_task(query_openai_with_context(context, user_query))
            user_session.set('streaming_task', task)
            await task
        else:
            logger.warning("Context or user query not found for follow-up question.")
    else:
        logger.warning("No current document ID found for follow-up question.")

@cl.action_callback("ask_new_question")
async def handle_new_question(action):
    """Handle new question action."""
    logger.info("New question about the same paper button clicked.")
    current_document_id = user_session.get('current_document_id')
    if current_document_id:
        logger.info(f"Asking new question for document_id: {current_document_id}")
        await ask_user_question(current_document_id)
    else:
        logger.warning("No current document ID found for new question.")

@cl.action_callback("ask_about_new_paper")
async def handle_new_paper(action):
    """Handle new paper action."""
    logger.info("New paper button clicked.")
    await ask_initial_query()

async def run_metadata_pipeline():
    logger.info('Starting daily task.')
    current_date = get_current_est_date()
    logger.info(f'Current EST date: {current_date}')
    from_date = current_date
    until_date = current_date
    try:
        await download_metadata(from_date, until_date)
        await remove_line_breaks_and_wrap('arxiv_metadata.xml', 'arxiv_metadata_cleaned.xml')
        df = await parse_xml_to_dataframe('arxiv_metadata_cleaned.xml')
        await aiofiles.os.remove('arxiv_metadata_cleaned.xml')  # Clean up intermediate file
        await aiofiles.os.remove('arxiv_metadata.xml')  # Clean up the original downloaded XML file
        df_processed = await process_arxiv_metadata(df)

        if not df_processed.empty:
            logger.info('DataFrame is not empty. Proceeding with Pinecone upload.')
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
    est = pytz.timezone('US/Eastern')
    return datetime.now(est).strftime('%Y-%m-%d')

async def download_metadata(from_date, until_date):
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
                record = await asyncio.to_thread(next, data).raw
                await f.write(record)
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

            except StopIteration:
                logger.info(f'Metadata For The Specified Period, {from_date} - {until_date} Downloaded.')
                break

            except Exception as e:
                errors += 1
                logger.error(f'Unexpected error: {e}')
                if errors > 5:
                    logger.critical('Too many consecutive errors, stopping the harvester.')
                    raise

async def handle_http_error(e):
    if e.response.status_code == 503:
        retry_after = e.response.headers.get('Retry-After', 30)
        logger.warning(f"HTTPError 503: Server busy. Retrying after {retry_after} seconds.")
        await asyncio.sleep(int(retry_after))
    else:
        logger.error(f'HTTPError: Status code {e.response.status_code}')
        raise e

async def remove_line_breaks_and_wrap(input_file: str, output_file: str):
    logger.info(f'Removing line breaks and wrapping content in {input_file}.')
    async with aiofiles.open(input_file, 'r', encoding='utf-8') as infile, aiofiles.open(output_file, 'w', encoding='utf-8') as outfile:
        await outfile.write("<records>")
        
        async for line in infile:
            cleaned_line = line.replace('\n', '').replace('\r', '')
            await outfile.write(cleaned_line)
        
        await outfile.write("</records>")
    logger.info(f'Content wrapped and saved to {output_file}.')

async def parse_xml_to_dataframe(input_file: str):
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
    logging.info('Processing DataFrame metadata.')
    
    df.rename(columns={
        'datestamp': 'last_edited',
        'id': 'document_id',
        'created': 'date_created'
    }, inplace=True)
    
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
    
    logging.info('DataFrame processing complete.')
    return df

async def upload_to_pinecone(df, vector_store):
    logger.info('Uploading data to Pinecone vector store.')
    texts = df['title_by_authors'].tolist()
    metadatas = df[['document_id']].to_dict(orient='records')
    await asyncio.to_thread(vector_store.add_texts, texts=texts, metadatas=metadatas)
    logger.info('Upload to Pinecone complete.')

async def daily_metadata_task():
    """Run the daily metadata pipeline at 11 PM EST."""
    est = pytz.timezone('US/Eastern')
    now = datetime.now(est)
    target_time = datetime.now(est).replace(hour=23, minute=00, second=0, microsecond=0)
    
    if now > target_time:
        target_time += timedelta(days=1)
    
    wait_time = (target_time - now).total_seconds()
    await asyncio.sleep(wait_time)
    
    while True:
        await run_metadata_pipeline()
        
        # Schedule next run for 11 PM EST the next day
        target_time += timedelta(days=1)
        wait_time = (target_time - datetime.now(est)).total_seconds()
        await asyncio.sleep(wait_time)

if __name__ == "__main__":
    asyncio.run(main())
