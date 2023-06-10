import pinecone
from tqdm.auto import tqdm
from uuid import uuid4
from decouple import config
from langchain.embeddings.openai import OpenAIEmbeddings
from constants import EMBEDDING_MODEL
from langchain.document_loaders import PyPDFLoader
from dotenv import find_dotenv, load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import logging, coloredlogs
load_dotenv(find_dotenv())

# ------------constants
BATCH_SIZE = 100

# --------------setup
logger = logging.getLogger(__name__)
coloredlogs.install(level=config('LOG_LEVEL', default='INFO'))

logger.info('Setup')
embed = OpenAIEmbeddings(model=EMBEDDING_MODEL)

pinecone.init(
    api_key=config('PINECONE_API_KEY'),  # find api key in console at app.pinecone.io
    environment=config('PINECONE_ENV')  # find next to api key in console
)

# # delete index if it exists
# pinecone.delete_index(config('PINECONE_INDEX_NAME'))
# create a new index
# pinecone.create_index(
#     name=config('PINECONE_INDEX_NAME'),
#     metric='dotproduct', # dotproduct bc the embeddings are normalized = 1 (see here: https://platform.openai.com/docs/guides/embeddings/which-distance-function-should-i-use)
#     dimension=1536 # 1536 dim of text-embedding-ada-002
# )

index = pinecone.Index(config('PINECONE_INDEX_NAME'))


def create_index(index, folder_path):
    logger.info(f'index stats before we start: \b{index.describe_index_stats()}')

    pdf_files = [file for file in os.listdir(folder_path) if file.endswith(".pdf")]
    for filename in tqdm(pdf_files):
        logger.info(f'Loading and Splitting Book: {filename}')
        file_path = os.path.join(folder_path, filename)
        loader = PyPDFLoader(file_path=file_path)
        pages = loader.load_and_split() # this allows us to track the page number
        n_pages = len(pages)
        logger.info('Running Batches to Embed and Send into Index:')
        for i in tqdm(range((n_pages // BATCH_SIZE) + 1)):
            batch = pages[i*BATCH_SIZE: (i+1)*BATCH_SIZE]
            batch_text = [page.page_content for page in batch]
            metadatas = [{
                "page": page.metadata['page'], 
                "text": page.page_content,
                "book": filename.strip('.pdf')
            } for page in batch]

            ids = [str(uuid4()) for _ in range(len(batch))]

            logger.debug('Embedding...')
            embeds = embed.embed_documents(batch_text)
            logger.debug('Inserting into Index...')
            index.upsert(vectors=zip(ids, embeds, metadatas))


    logger.info(f'Index stats after: \n{index.describe_index_stats()}')


if __name__ == "__main__":
    create_index(index, './data/books-pdfs')