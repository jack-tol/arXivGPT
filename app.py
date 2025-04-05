import os
import logging
import asyncio
from datetime import datetime, timedelta

import aiofiles
import aiohttp
import pytz
import arxiv
import chainlit as cl
from chainlit.user_session import user_session
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from openai import AsyncOpenAI

from metadata_pipeline import main_pipeline

load_dotenv(override=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
metadata_vector_store = PineconeVectorStore.from_existing_index(
    embedding=embedding_model, index_name="arxiv-rag-metadata"
)
chunks_vector_store = PineconeVectorStore.from_existing_index(
    embedding=embedding_model, index_name="arxiv-rag-chunks"
)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=0, separators=["\n\n", "\n", " "])

task_scheduled = False

pipeline_lock = asyncio.Lock()

async def schedule_daily_metadata_task():
    async def run_main_pipeline_daily():
        est = pytz.timezone("US/Eastern")
        while True:
            try:
                async with pipeline_lock:
                    now = datetime.now(est)
                    
                    target_time = now.replace(hour=23, minute=0, second=0, microsecond=0)
                    if now >= target_time:
                        target_time += timedelta(days=1)
                    
                    delay_seconds = (target_time - now).total_seconds()
                    logging.info(f"Next metadata update at {target_time.strftime('%Y-%m-%d %H:%M:%S %Z')} "
                                 f"(in {delay_seconds/3600:.1f} hours)")
                    
                    await asyncio.sleep(delay_seconds)
                    logging.info("Starting daily metadata update...")
                    
                    try:
                        await main_pipeline()
                    except Exception as e:
                        logging.error(f"Pipeline failed: {e}")
                        await asyncio.sleep(3600)

            except Exception as e:
                logging.error(f"Scheduler error: {e}")
                await asyncio.sleep(3600)

    global task_scheduled
    if not task_scheduled:
        task_scheduled = True
        asyncio.create_task(run_main_pipeline_daily())
        logging.info("Daily metadata scheduler initialized")

async def does_paper_exist(document_id):
    try:
        filter = {"document_id": {"$eq": document_id}}
        exists = len(chunks_vector_store.similarity_search(query="Chunks Existence Check", k=1, filter=filter)) > 0
        return exists
    except Exception as e:
        logging.error(f"Error checking if paper exists: {e}")
        return False

async def process_paper(document_id):
    try:
        async with aiohttp.ClientSession() as session:
            paper = await asyncio.to_thread(
                next, arxiv.Client().results(arxiv.Search(id_list=[str(document_id)]))
            )
            if not paper.pdf_url:
                raise Exception(f"No PDF URL found for document ID: {document_id}")

            filename = f"{document_id}.pdf"
            async with session.get(paper.pdf_url) as response:
                if response.status == 200:
                    async with aiofiles.open(filename, "wb") as f:
                        await f.write(await response.read())
                else:
                    raise Exception(f"Failed to download PDF. Status: {response.status}")

            loader = PyPDFLoader(filename)
            pages = await asyncio.to_thread(loader.load)

            content, found_references = [], False
            for page in pages:
                if found_references:
                    break
                page_text = page.page_content
                if "references" in page_text.lower():
                    content.append(page_text.split("References")[0])
                    found_references = True
                else:
                    content.append(page_text)

            chunks = text_splitter.split_text("".join(content))
            if not chunks:
                raise Exception("No valid chunks generated from the text.")

            await asyncio.to_thread(
                chunks_vector_store.from_texts,
                texts=chunks,
                embedding=embedding_model,
                metadatas=[{"document_id": document_id} for _ in chunks],
                index_name="arxiv-rag-chunks",
            )

            if os.path.exists(filename):
                os.remove(filename)

            return True
    except Exception as e:
        logging.error(f"Error processing paper ID {document_id}: {e}")
        return False

async def retrieve_context(document_id):
    for attempt in range(10):
        try:
            filter = {"document_id": {"$eq": document_id}}
            retrieved_chunks = chunks_vector_store.similarity_search(
                query="Retrieve Context", k=100, filter=filter
            )
            if retrieved_chunks:
                return "\n".join(chunk.page_content for chunk in retrieved_chunks)
            await asyncio.sleep(5)
        except Exception as e:
            logging.error(f"#### Error retrieving context for document ID {document_id} on attempt {attempt + 1}: {e}")
    raise Exception(f"#### Failed to retrieve chunks for document ID {document_id} after 5 attempts.")

async def select_paper(search_results):
    if not search_results:
        return None

    paper_list_message = (
        "### Select a Paper by Entering Its Number\n\n"
        "| No. | Paper Title | Doc. ID |\n"
        "|-----|-------------|---------|\n"
        + "".join(
            f"| {i + 1} | {doc.page_content} | {doc.metadata['document_id']} |\n"
            for i, doc in enumerate(search_results)
        )
    )
    res = await cl.AskUserMessage(content=paper_list_message, timeout=3600).send()

    while True:
        if not res or not res.get("output"):
            res = await cl.AskUserMessage(content="❌ Invalid selection. Please enter a valid number.", timeout=3600).send()
            continue

        try:
            choice = int(res["output"]) - 1
            if 0 <= choice < len(search_results):
                return search_results[choice].metadata["document_id"]
            else:
                res = await cl.AskUserMessage(content="❌ Number out of range. Please enter a valid number from the list.", timeout=3600).send()
        except ValueError:
            res = await cl.AskUserMessage(content="❌ Invalid input. Please enter a valid number.", timeout=3600).send()

@cl.on_message
async def on_message(message: cl.Message):
    if not user_session.get("in_chat_mode", False):
        await cl.Message(content="#### No context available. Please select a paper first.").send()
        return

    message_history = user_session.get("message_history", [])
    message_history.append({"role": "user", "content": message.content})

    try:
        msg = cl.Message(content="")
        await msg.send()

        stream = await client.chat.completions.create(
            messages=message_history, stream=True, model="gpt-4o", temperature=0.7
        )

        async for part in stream:
            if token := part.choices[0].delta.content or "":
                await msg.stream_token(token)

        await msg.update()
        message_history.append({"role": "assistant", "content": msg.content})
        user_session.set("message_history", message_history)

    except Exception as e:
        await cl.Message(content=f"#### An error occurred while processing your query: {e}").send()

@cl.on_chat_start
async def main():
    system_message = {
        "role": "system",
        "content": """Your job is to answer the users' queries using only the provided context.
Be detailed and long-winded. Format your responses in markdown formatting, making good use of headings,
subheadings, ordered and unordered lists, and regular text formatting such as **bold** and *italic* text.
Provide equations properly formatted in LaTeX for correct rendering.

Use Markdown-compatible LaTeX formatting:
- For inline equations, wrap them in single dollar signs `$` (e.g., `$(x^2 + y^2 = z^2)$`).
- For larger, display-style equations, wrap them in double dollar signs `$$` (e.g., `$$x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$$`).
This style is commonly used in Markdown-based tools like Obsidian, Jupyter Notebooks, and Notion.

For example, `$(\kappa)$` will render correctly, whereas `( \kappa )` will not. Similarly, `$(\ell_{\text{margin}})$` renders properly, while `( \ell_{\text{margin}} )` does not. Ensure all LaTeX provided in your response is properly formatted for Markdown compatibility."""
    }

    user_session.set("message_history", [system_message])
    user_session.set("in_chat_mode", False)
    user_session.set("processing_paper", False)

    await schedule_daily_metadata_task()

    while True:
        if user_session.get("processing_paper", False):
            await cl.Message(content="### Processing in progress. Please wait...").send()
            continue

        raw_input = await cl.AskUserMessage(
            content="""## Welcome to arXivGPT

arXivGPT assists students and researchers by providing real-time access to the latest research uploaded to arXiv.

Updated daily, it ensures that users always have the most up-to-date information at their fingertips.

### Instructions
1. **Enter the Title**: Start by entering the title of the research paper you're interested in.
2. **Select a Paper**: Choose a paper from the list by entering its number.
3. **Database Check**: The system will check if the paper is in the database.
   - If it's already in the database, you can immediately start asking questions.
   - If it's not, the paper will be downloaded, and then you can begin asking your questions.

4. **Enjoy the Conversation**: Feel free to ask about any aspect of the paper. If you wish to explore a different paper, simply click the "New Chat" button in the top left corner to start again.

### Get Started
Enter the title of the research paper you want to learn more about.
""",
            timeout=3600,
        ).send()

        if not raw_input or not raw_input.get("output"):
            await cl.Message(content="Invalid input. Please try again.").send()
            continue

        paper_input = str(raw_input.get("output"))

        user_session.set("processing_paper", True)

        try:
            search_results = metadata_vector_store.similarity_search(query=paper_input, k=5)

            if not search_results:
                await cl.Message(content="#### No results found. Please try a different title.").send()
                user_session.set("processing_paper", False)
                continue

            selected_doc_id = await select_paper(search_results)

            if not selected_doc_id:
                user_session.set("processing_paper", False)
                continue

            status_msg = cl.Message(content="#### Processing your paper. Please wait...")
            await status_msg.send()

            if not await does_paper_exist(selected_doc_id):
                if not await process_paper(selected_doc_id):
                    status_msg.content = "### An error occurred while processing the paper."
                    await status_msg.update()
                    user_session.set("processing_paper", False)
                    continue

            combined_content = await retrieve_context(selected_doc_id)
            user_session.set("message_history", [
                system_message,
                {"role": "system", "content": f"Context: {combined_content}"}
            ])
            user_session.set("in_chat_mode", True)

            status_msg.content = "#### Paper details retrieved. You can now begin asking questions!"
            await status_msg.update()

            break
        except Exception as e:
            await cl.Message(content=f"An error occurred: {e}").send()
        finally:
            user_session.set("processing_paper", False)