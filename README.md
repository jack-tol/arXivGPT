# arXivGPT

Welcome to the GitHub repository for **arXivGPT**, a project leveraging Retrieval-Augmented Generation (RAG) to help students, enthusiasts, and researchers understand and consume the latest research published on arXiv.

arXivGPT is a ConversationAI Product. For more information. visit https://conversationai.io.

## About

arXivGPT provides detailed explanations of research papers, making complex concepts and methodologies more accessible. This tool utilizes language models and RAG to enhance the learning experience.

A three-part series covers the development and rationale behind the functions and code of this project. If you're interested in a deeper dive, I recommend reading it.

## Important Links

- [Hosted Web Application](https://arxivgpt.net)
- [ConversationAI](https://conversationai.io)
- [Blog and Additional Resources](https://jacktol.net)

## How does it work?

This system connects to the live stream of papers uploaded to arXiv daily through a custom-built metadata pipeline that facilitates the downloading, processing, and uploading of arXiv metadata. Essentially, it creates an arXiv 'Search Engine' where you can search for and select a research paper to learn more about.

Once you select a paper, the system checks if chunks of that paper exist in the chunks vector store. If they do, you can enter your query about the paper immediately. If no chunks exist, the system remotely downloads, loads, chunks, embeds, and uploads the paper within seconds before asking you to enter your question.

### Instructions

1. **Enter the Title**: Start by entering the title of the research paper you wish to learn more about.
2. **Select a Paper**: Choose a paper from the list of retrieved papers.
3. **Database Check**: The system will check if the research paper exists in the database.
   - If it exists, you will be prompted to enter your question.
   - If it does not exist, the program will download the paper to the database and then ask you to enter your question.
4. **Read the Answer**: After reading the answer, you will have the following options:
   - Ask a follow-up question.
   - Ask a new question about the same paper.
   - Ask a new question about a different paper.

### YouTube Videos

- [arXivGPT - Prototype Demonstration — Jack Tol](https://youtu.be/uJbo8HF8ZaM)
- [arXivGPT - Deployment + Demonstration — Jack Tol](https://youtu.be/4lSm1JisKeY)

For any inquiries, further information, or to provide feedback and suggestions, please reach out via email at contact@jacktol.net.

