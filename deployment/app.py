import os
import asyncio
from openai import OpenAI
import arxiv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_pinecone import PineconeVectorStore
import chainlit as cl

# Initialize components
os.environ["TOKENIZERS_PARALLELISM"] = "false"
embedding_model = HuggingFaceEmbeddings()
metadata_vector_store = PineconeVectorStore.from_existing_index(embedding=embedding_model, index_name="arxiv-metadata")
chunks_vector_store = PineconeVectorStore.from_existing_index(embedding=embedding_model, index_name="arxiv-project-chunks")
semantic_chunker = SemanticChunker(embeddings=embedding_model, buffer_size=1, add_start_index=False)

session_active = True
selected_doc_id = None
context = []

async def process_and_upload_chunks(document_id):
    try:
        await cl.Message(content="Paper Not Found. Downloading, Processing, and Uploading. Please Wait.").send()
        await asyncio.sleep(1)  # Short delay to ensure the message is displayed
        paper = next(arxiv.Client().results(arxiv.Search(id_list=[str(document_id)])))
        paper.download_pdf(filename=f"{document_id}.pdf")
        loader = PyPDFLoader(f"{document_id}.pdf")
        pages = loader.load_and_split()

        chunks = []
        for page in pages:
            text = page.page_content
            chunks.extend(semantic_chunker.split_text(text))
        chunks_vector_store.from_texts(
            texts=chunks,
            embedding=embedding_model,
            metadatas=[{"document_id": document_id} for _ in chunks],
            index_name="arxiv-project-chunks"
        )
        os.remove(f"{document_id}.pdf")
    except Exception as e:
        await cl.Message(content=f"Error Processing & Uploading Chunks: {str(e)}").send()

async def check_chunks_existence(document_id):
    filter = {"document_id": {"$eq": document_id}}
    test_query = chunks_vector_store.similarity_search(query="Chunks Existence Check", k=1, filter=filter)
    return bool(test_query)

async def process_user_query(document_id):
    global context
    context = []
    res = await cl.AskUserMessage(content="### Please Enter Your Question:").send()
    if res is None:
        return None, None
    user_query = res['output']
    filter = {"document_id": {"$eq": document_id}}
    search_results = chunks_vector_store.similarity_search(query=user_query, k=15, filter=filter)
    context = [doc.page_content for doc in search_results]
    return context, user_query

async def query_openai_with_context(context, user_query):
    if not context:
        return "No context available to answer the question."
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Use the provided context to answer the provided question."},
            {"role": "user", "content": f"Context: {context}"},
            {"role": "user", "content": f"Question: {user_query}"}
        ]
    )
    return response.choices[0].message.content.strip()

async def select_document_from_results(search_results):
    if not search_results:
        await cl.Message(content="No search results found.").send()
        return None

    response = "### List of Retrieved Papers | Please Select the Number Corresponding to Your Desired Paper:\n"
    for i, doc in enumerate(search_results, start=1):
        page_content = doc.page_content
        document_id = doc.metadata['document_id']
        response += f"{i}: {page_content}\n"

    await cl.Message(content=response).send()
    res = await cl.AskUserMessage(content="").send()
    if res is None:
        return None
    try:
        user_choice = int(res['output']) - 1
    except ValueError:
        await cl.Message(content="\nInvalid selection. Please run the process again and select a valid number.").send()
        return None

    if 0 <= user_choice < len(search_results):
        selected_doc_id = search_results[user_choice].metadata['document_id']
        selected_paper_title = search_results[user_choice].page_content
        await cl.Message(content=f"\nYou selected: {selected_paper_title}").send()
        return selected_doc_id
    else:
        await cl.Message(content="\nInvalid selection. Please run the process again and select a valid number.").send()
        return None

async def ask_for_paper():
    res = await cl.AskUserMessage(content="### Please Enter the Title of the Research Paper You Wish to Learn More About:", timeout=3600).send()
    if res is None:
        return None
    initial_query = res['output']
    search_results = metadata_vector_store.similarity_search(query=initial_query, k=5)
    return await select_document_from_results(search_results)

async def ask_for_query(document_id):
    chunks_exist = await check_chunks_existence(document_id)
    if not chunks_exist:
        await process_and_upload_chunks(document_id)
        # Add a longer delay to ensure chunks are indexed
        await asyncio.sleep(5)  # Increase the duration as needed

        # Verify chunks upload
        retries = 5
        for attempt in range(retries):
            chunks_exist = await check_chunks_existence(document_id)
            if chunks_exist:
                break
            await asyncio.sleep(2)  # Wait before retrying
            await cl.Message(content=f"Verifying chunks upload: Attempt {attempt + 1}/{retries}").send()

        if not chunks_exist:
            await cl.Message(content="Chunks were not successfully uploaded. Please try again.").send()
            return

    retries = 3
    for attempt in range(retries):
        context, user_query = await process_user_query(document_id)
        if context:
            response = await query_openai_with_context(context, user_query)
            await cl.Message(content=response).send()
            await send_query_actions()
            return
        await asyncio.sleep(2)  # Wait before retrying
        await cl.Message(content=f"Retrying context retrieval: Attempt {attempt + 1}/{retries}").send()

    await cl.Message(content="Unable to retrieve context for the document. Please try again.").send()

async def send_query_actions():
    if session_active:
        actions = [
            cl.Action(name="ask_followup_question", value="followup_question", description="Ask a follow-up question (Uses the previously retrieved context)", label="Ask a Follow-Up Question"),
            cl.Action(name="ask_new_question", value="new_question", description="Ask a new question about the same paper (Retrieves new context)", label="Ask a New Question About the Same Paper"),
            cl.Action(name="ask_about_new_paper", value="new_paper", description="Ask a new question about a new paper", label="Ask About a Different Paper"),
            cl.Action(name="exit_session", value="exit", description="Quit the Program", label="Exit Session")
        ]
        await cl.Message(content="Choose an action:", actions=actions).send()

@cl.action_callback("ask_followup_question")
async def on_ask_followup_question(action):
    await action.remove()
    global context
    if session_active and context:
        followup_res = await cl.AskUserMessage(content="### Please Enter Your Follow-Up Question:", timeout=3600).send()
        if followup_res is None:
            return
        followup_query = followup_res['output']
        response = await query_openai_with_context(context, followup_query)
        await cl.Message(content=response).send()
        await send_query_actions()

@cl.action_callback("ask_new_question")
async def on_ask_new_question(action):
    global selected_doc_id
    await action.remove()
    if session_active and selected_doc_id:
        await ask_for_query(selected_doc_id)
    else:
        await cl.Message(content="No Document Selected or Session is Inactive.").send()

@cl.action_callback("ask_about_new_paper")
async def on_ask_about_new_paper(action):
    global selected_doc_id
    selected_doc_id = None
    await action.remove()
    if session_active:
        selected_doc_id = await ask_for_paper()
        if selected_doc_id:
            await ask_for_query(selected_doc_id)

@cl.action_callback("exit_session")
async def on_exit_session(action):
    global session_active
    session_active = False
    await cl.Message(content="Session Exited. Take Care.").send()
    await action.remove()

@cl.on_chat_start
async def main():
    global selected_doc_id  # Ensure global usage
    text_content = """# Welcome to the arXiv Research Paper Learning Supplement

This system is connected to the live stream of papers being uploaded to arXiv daily.

## Instructions

1. **Enter the Title**: Start by entering the title of the research paper you wish to learn more about.
2. **Select a Paper**: Choose a paper from the list of retrieved papers.
3. **Database Check**: The system will check if the research paper exists in the research paper content database.
   - If it exists, you will be prompted to enter your question.
   - If it does not exist, the program will download the paper to the database and then ask you to enter your question.
4. **Read the Answer**: After reading the answer, you will have the following options:
   - Ask a follow-up question.
   - Ask a new question about the same paper.
   - Ask a new question about a different paper.
   - Exit the application.

"""
    elements = [
        cl.Text(content=text_content, display="inline")
    ]
    await cl.Message(
        content="",
        elements=elements,
    ).send()
    
    selected_doc_id = await ask_for_paper()
    if selected_doc_id:
        await ask_for_query(selected_doc_id)