import os

import arxiv
import asyncio
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_pinecone import PineconeVectorStore
import chainlit as cl
from openai import AsyncOpenAI

os.environ["TOKENIZERS_PARALLELISM"] = "false"
embedding_model = HuggingFaceEmbeddings()
metadata_vector_store = PineconeVectorStore.from_existing_index(embedding=embedding_model, index_name="arxiv-metadata")
chunks_vector_store = PineconeVectorStore.from_existing_index(embedding=embedding_model, index_name="arxiv-project-chunks")
semantic_chunker = SemanticChunker(embeddings=embedding_model, buffer_size=1, add_start_index=False)
current_task = None

async def send_query_actions():
    if cl.user_session.get("session_active"):
        actions = [
            cl.Action(name="ask_followup_question", value="followup_question", description="Ask a follow-up question (Uses the previously retrieved context)", label="Ask a Follow-Up Question"),
            cl.Action(name="ask_new_question", value="new_question", description="Ask a new question about the same paper (Retrieves new context)", label="Ask a New Question About the Same Paper"),
            cl.Action(name="ask_about_new_paper", value="new_paper", description="Ask a new question about a new paper", label="Ask About a Different Paper"),
            cl.Action(name="exit_session", value="exit", description="Quit the Program", label="Exit Session")
        ]
        await cl.Message(content="### Please Select One of the Following Options:", actions=actions).send()

@cl.on_chat_start
async def main():
    cl.user_session.set("session_active", True)
    cl.user_session.set("current_context", [])
    cl.user_session.set("current_document_id", None)
    
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
   - Exit the application.

### Get Started
When You're Ready, Follow the First Step Below.
"""
    await cl.Message(content=text_content).send()
    await ask_initial_query()

async def ask_initial_query():
    res = await cl.AskUserMessage(content="### Please Enter the Title of the Research Paper You Wish to Learn More About:", timeout=3600).send()
    if res:
        initial_query = res['output']
        search_results = metadata_vector_store.similarity_search(query=initial_query, k=5)
        selected_doc_id = await select_document_from_results(search_results)
        if selected_doc_id:
            cl.user_session.set("current_document_id", selected_doc_id)
            if not await do_chunks_exist_already(selected_doc_id):
                await process_and_upload_chunks(selected_doc_id)
            await ask_user_question(selected_doc_id)

async def ask_user_question(document_id):
    context, user_query = await process_user_query(document_id)
    if context and user_query:
        cl.user_session.set("current_context", context)
        cl.user_session.set("current_task", asyncio.create_task(query_openai_with_context(context, user_query)))
        await cl.user_session.get("current_task")

@cl.action_callback("ask_followup_question")
async def handle_followup_question(action):
    if not cl.user_session.get("session_active"):
        await cl.Message(content="No Active Session").send()
        return
    res = await cl.AskUserMessage(content="### Please Enter Your Follow-Up Question:", timeout=3600).send()
    if res:
        user_query = res['output']
        await query_openai_with_context(cl.user_session.get("current_context"), user_query)

@cl.action_callback("ask_new_question")
async def handle_new_question(action):
    if not cl.user_session.get("session_active"):
        await cl.Message(content="No Active Session").send()
        return
    if cl.user_session.get("current_document_id"):
        await ask_user_question(cl.user_session.get("current_document_id"))

@cl.action_callback("ask_about_new_paper")
async def handle_new_paper(action):
    if not cl.user_session.get("session_active"):
        await cl.Message(content="No Active Session").send()
        return
    await ask_initial_query()

@cl.action_callback("exit_session")
async def handle_exit(action):
    if not cl.user_session.get("session_active"):
        await cl.Message(content="No Active Session").send()
        return
    cl.user_session.set("session_active", False)
    await cl.Message(content="Session Ended. Please Take Care.").send()

async def select_document_from_results(search_results):
    if not search_results:
        await cl.Message(content="No Search Results Found").send()
        return None

    message_content = "### List of Retrieved Papers | Please Select the Number Corresponding to Your Desired Paper:\n"
    for i, doc in enumerate(search_results, start=1):
        page_content = doc.page_content[:100]
        document_id = doc.metadata['document_id']
        message_content += f"{i}: {page_content}\n"

    await cl.Message(content=message_content).send()

    while True:
        res = await cl.AskUserMessage(content="", timeout=3600).send()
        if res:
            try:
                user_choice = int(res['output']) - 1
                if 0 <= user_choice < len(search_results):
                    selected_doc_id = search_results[user_choice].metadata['document_id']
                    selected_paper_title = search_results[user_choice].page_content
                    await cl.Message(content=f"\nYou selected: {selected_paper_title}").send()
                    return selected_doc_id
                else:
                    await cl.Message(content="\nInvalid Selection. Please enter a valid number from the list.").send()
            except ValueError:
                await cl.Message(content="\nInvalid input. Please enter a number.").send()
        else:
            await cl.Message(content="\nNo selection made. Please enter a valid number from the list.").send()

async def do_chunks_exist_already(document_id):
    filter = {"document_id": {"$eq": document_id}}
    test_query = chunks_vector_store.similarity_search(query="Chunks Existence Check", k=1, filter=filter)
    return bool(test_query)

async def process_and_upload_chunks(document_id):
    await cl.Message(content="### Paper Not Found. Downloading, Processing, and Uploading in Progress. Please Wait; This Won't Take Long.").send()
    await asyncio.sleep(2)
    paper = next(arxiv.Client().results(arxiv.Search(id_list=[str(document_id)])))
    paper.download_pdf(filename=f"{document_id}.pdf")
    loader = PyPDFLoader(f"{document_id}.pdf")

    pages = loader.load_and_split()

    chunks = []
    for page in pages:
        text = page.page_content
        chunks.extend(semantic_chunker.split_text(text))
    chunks_vector_store.from_texts(texts=chunks, embedding=embedding_model, metadatas=[{"document_id": document_id} for _ in chunks], index_name="arxiv-project-chunks")
    os.remove(f"{document_id}.pdf")

async def process_user_query(document_id):
    res = await cl.AskUserMessage(content="### Please Enter Your Question:", timeout=3600).send()
    if res:
        user_query = res['output']
        context = []
        filter = {"document_id": {"$eq": document_id}}
        search_results = chunks_vector_store.similarity_search(query=user_query, k=15, filter=filter)
        for doc in search_results:
            context.append(doc.page_content)
        print(context)
        print(user_query)
        return context, user_query
    return None, None

@cl.on_stop
async def on_stop():
    if cl.user_session.get("current_task"):
        cl.user_session.get("current_task").cancel()
        cl.user_session.set("current_task", None)
    await send_query_actions()

async def query_openai_with_context(context, user_query):
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
         Sometimes the equations retrieved from the context will be formatted improperly in an incompatible format
         for correct markdown rendering. Therefore, if you ever need to provide equations, make sure they are
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

    stream_task = asyncio.create_task(stream_response())
    cl.user_session.set("current_task", stream_task)

    try:
        await stream_task
    except asyncio.CancelledError:
        stream_task.cancel()
        await send_query_actions()
        return

    final_response = msg.content
    await msg.update()

    cl.user_session.set("current_task", None)

    await send_query_actions()
    return final_response