import os
import arxiv
import aiofiles
import aiofiles.os
import asyncio
import logging
import pandas as pd
from requests.exceptions import HTTPError, RequestException
from datetime import datetime, timedelta
import pytz
import re
import ast
import chainlit as cl
from openai import AsyncOpenAI
from chainlit.context import context
from chainlit.user_session import user_session
from aiohttp import ClientSession
from metadata_pipeline import daily_metadata_task
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.basicConfig(filename='combined_log.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

daily_task_scheduled = False

def initialize_embeddings():
    """Initialize the OpenAI embedding model."""
    logger.info("Initializing OpenAI embeddings...")
    return OpenAIEmbeddings(model="text-embedding-3-small")

def initialize_vector_stores(embedding_model):
    """Initialize Pinecone vector stores for metadata and chunks."""
    logger.info("Initializing Pinecone vector stores...")
    metadata_vector_store = PineconeVectorStore.from_existing_index(embedding=embedding_model, index_name="arxiv-rag-metadata")
    chunks_vector_store = PineconeVectorStore.from_existing_index(embedding=embedding_model, index_name="arxiv-rag-chunks")
    return metadata_vector_store, chunks_vector_store

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
    global daily_task_scheduled
    
    if not daily_task_scheduled:
        asyncio.create_task(daily_metadata_task())
        daily_task_scheduled = True
    
    embedding_model = initialize_embeddings()
    metadata_vector_store, chunks_vector_store = initialize_vector_stores(embedding_model)
    text_splitter = initialize_text_splitter()

    user_session.set('embedding_model', embedding_model)
    user_session.set('metadata_vector_store', metadata_vector_store)
    user_session.set('chunks_vector_store', chunks_vector_store)
    user_session.set('text_splitter', text_splitter)
    user_session.set('current_document_id', None)

    text_content = """## Welcome to the arXiv Research Assistant

This system is designed to support students, researchers, and enthusiasts by providing real-time access to, and understanding of, the extensive research continually uploaded to arXiv.

With daily updates, it seamlessly integrates new papers, ensuring users always have the latest information at their fingertips.

### Instructions
1. **Enter the Title**: Start by entering the title of the research paper you wish to learn more about.
2. **Select a Paper**: Select a paper from the retrieved list by entering its corresponding number.
3. **Database Check**: The system will verify if the paper is already in the database.
   - If it exists, you'll be prompted to enter your question.
   - If it does not exist, the system will download the paper to the database and then prompt you to enter your question.
4. **Read the Answer**: After receiving the answer, you can:
   - Ask a follow-up question.
   - Ask a new question about the same paper.
   - Ask a new question about a different paper.

### Get Started
When you're ready, follow the first step below.
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
    """Allow user to select a document from the search results."""
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
        async with ClientSession() as session:
            paper = await asyncio.to_thread(next, arxiv.Client().results(arxiv.Search(id_list=[str(document_id)])))
            url = paper.pdf_url
            filename = f"{document_id}.pdf"
            await download_pdf(session, document_id, url, filename)

        loader = PyPDFLoader(filename)
        pages = await asyncio.to_thread(loader.load)

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

        embedding_model = user_session.get('embedding_model')
        if not embedding_model:
            raise ValueError("Embedding model not initialized")

        chunks_vector_store = user_session.get('chunks_vector_store')
        await asyncio.to_thread(
            chunks_vector_store.from_texts,
            texts=chunks,
            embedding=embedding_model,
            metadatas=[{"document_id": document_id} for _ in chunks],
            index_name="arxiv-rag-chunks"
        )

        await aiofiles.os.remove(filename)
        logger.info(f"Successfully processed and uploaded chunks for document_id: {document_id}")
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
        attempts = 5
        for attempt in range(attempts):
            search_results = chunks_vector_store.similarity_search(query=user_query, k=15, filter=filter)
            logger.info(f"Context retrieval attempt {attempt + 1}: Found {len(search_results)} results")
            context = [doc.page_content for doc in search_results]
            if context:
                break
            logger.info(f"No context found, retrying... (attempt {attempt + 1}/{attempts})")
            await asyncio.sleep(2)

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

if __name__ == "__main__":
    asyncio.run(main())