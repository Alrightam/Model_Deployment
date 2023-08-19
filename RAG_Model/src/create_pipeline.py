# Import the EmbeddingRetriever and FARMReader classes from the haystack.nodes module.
from haystack.nodes import EmbeddingRetriever, FARMReader

# Import the ExtractiveQAPipeline and Pipeline classes from the haystack.pipelines module.
from haystack.pipelines import ExtractiveQAPipeline
from haystack.pipelines import Pipeline

# Define a function called make_document_qa_pipeline that takes a document_store as input.
def make_document_qa_pipeline(document_store):
    # Create an instance of the EmbeddingRetriever class with the specified document_store and embedding_model.
    retriever = EmbeddingRetriever(
        document_store=document_store,
        embedding_model="sentence-transformers/all-mpnet-base-v2"
    )
    # Update the embeddings of the documents in the document_store using the retriever.
    document_store.update_embeddings(retriever)
    
    # Create an instance of the FARMReader class with the specified model_name_or_path.
    reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2")
    
    # Create an instance of the ExtractiveQAPipeline class with the specified reader and retriever.
    document_qa = ExtractiveQAPipeline(reader=reader, retriever=retriever)
    
    # Return the document_qa pipeline.
    return document_qa
