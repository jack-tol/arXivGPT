import chainlit as cl
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings, OpenAI
import arxiv

# Initialize components
embedding = HuggingFaceEmbeddings()
metadata_vector_store = PineconeVectorStore.from_existing_index(embedding=embedding, index_name="arxiv-metadata")
chunks_vector_store = PineconeVectorStore.from_existing_index(embedding=embedding, index_name="arxiv-project-chunks")
semantic_chunker = SemanticChunker(embeddings=embedding, buffer_size=1, add_start_index=False)

@cl.on_chat_start
async def main():
    selected_doc_id = None
    while True:
        if selected_doc_id is None:
            res = await cl.AskUserMessage(content="Enter the title or topic of the paper you're interested in:", timeout=300).send()
            initial_query = res['output']
            search_results = await perform_similarity_search(metadata_vector_store, initial_query, 5)
            selected_doc_id = await select_document_from_results(search_results)

        if selected_doc_id:
            chunks_exist = await do_chunks_exist_already(selected_doc_id)
            if not chunks_exist:
                await process_and_index_document(selected_doc_id)

            context, user_query = await process_user_query(selected_doc_id)
            if context:  # Ensure there's context before querying
                response = await query_openai_with_context(context, user_query)
                await cl.Message(content=response).send()

                # Ask the user for their next action
                next_action = await cl.AskUserMessage(content="Choose an action: 1) Ask a new question on the same paper, 2) Ask about a new paper, 3) Exit: ", timeout=300).send()
                if next_action['output'] == '1':
                    continue
                elif next_action['output'] == '2':
                    selected_doc_id = None
                    continue
                elif next_action['output'] == '3':
                    break
                else:
                    await cl.Message(content="Invalid choice. Exiting.").send()
                    break
            else:
                await cl.Message(content="Unable to retrieve context for the document.").send()
                selected_doc_id = None  # Optionally reset if no context could be fetched


async def perform_similarity_search(vector_store, query, k=5, filter=None):
    try:
        results = vector_store.similarity_search(query, k, filter=filter)
        return results
    except Exception as e:
        await cl.Message(content=f"Error during similarity search: {str(e)}").send()
        return []

async def select_document_from_results(search_results):
    if not search_results:
        await cl.Message(content="No search results found.").send()
        return None

    response = "Top search results based on content and metadata:\n"
    for i, doc in enumerate(search_results, start=1):
        page_content = doc.page_content
        document_id = doc.metadata['document_id']
        response += f"{i}: Research Paper Title & Author: {page_content}\n   Document ID: {document_id}\n"

    await cl.Message(content=response).send()
    res = await cl.AskUserMessage(content="Select a paper by entering its number:", timeout=60).send()
    user_choice = int(res['output']) - 1

    if 0 <= user_choice < len(search_results):
        selected_doc_id = search_results[user_choice].metadata['document_id']
        await cl.Message(content=f"\nYou selected document ID: {selected_doc_id}").send()
        return selected_doc_id
    else:
        await cl.Message(content="\nInvalid selection. Please run the process again and select a valid number.").send()
        return None

async def do_chunks_exist_already(document_id):
    filter = {"document_id": {"$eq": document_id}}
    try:
        test_query = await perform_similarity_search(chunks_vector_store, "Initial existence check", 1, filter)
        exist = len(test_query) > 0
        if not exist:
            await cl.Message(content="No existing chunks found, need to process and index the document.").send()
        return exist
    except Exception as e:
        await cl.Message(content=f"Error checking chunk existence: {str(e)}").send()
        return False


async def process_and_index_document(document_id):
    await cl.Message(content="Selected paper not found. Attempting to locate and download.").send()
    paper = next(arxiv.Client().results(arxiv.Search(id_list=[str(document_id)])))  # Ensure document_id is a string
    paper.download_pdf(filename=f"{document_id}.pdf")
    await cl.Message(content="Download Successful. Preparing document for processing.").send()

    loader = PyPDFLoader(f"{document_id}.pdf")
    pages = loader.load_and_split()
    await cl.Message(content="Document processed. Extracting and organizing content for indexing.").send()

    chunks = []
    for page in pages:
        text = page.page_content
        chunks.extend(semantic_chunker.split_text(text))
    await cl.Message(content="Content organized. Embedding and initiating upload to search index.").send()

    docsearch = PineconeVectorStore.from_texts(chunks, embedding, index_name="arxiv-project-chunks", metadatas=[{"document_id": document_id} for _ in chunks])
    await cl.Message(content="Upload finished. The document is now indexed and searchable.").send()
    return docsearch

async def process_user_query(document_id):
    # Step 1: Create an empty context list
    context = []

    # Step 3: Define the filter
    filter = {"document_id": {"$eq": document_id}}

    # Step 4: Get a user query via ChainLit's chat interface
    res = await cl.AskUserMessage(content="Please enter your question:", timeout=120).send()
    user_query = res['output']

    # Step 5: Perform the similarity search on user query and chunks vector store
    search_results = await perform_similarity_search(chunks_vector_store, user_query, 10, filter)

    # Step 6: Append relevant chunks to the context list
    for doc in search_results:
        context.append(doc.page_content)

    return context, user_query


async def query_openai_with_context(context, user_query):
    # Initialize the OpenAI client
    template = """Using the following context:
    {context}
    To answer this question:
    {user_query}
    """
    prompt = ChatPromptTemplate.from_template(template)
    model = OpenAI()
    parser = StrOutputParser()

    chain = prompt | model | parser

    # Join context into a single string
    formatted_context = '\n'.join(context)

    # Debugging: Log the context being sent
    print(f"Context being sent to OpenAI model: {formatted_context}")

    # Invoke the chain with the user's query and the context
    response = chain.invoke({"context": formatted_context, "user_query": user_query})

    # Return the output for further processing or display
    return response