# First Party
from granite_common.base.types import Document


class ElasticsearchRetriever:
    """Retriever for documents hosted on an ElasticSearch server."""

    def __init__(
        self,
        corpus_name: str,
        host: str,
        **kwargs: dict[str, int],
    ):
        """
        :param hosts: Full url:port to the Elasticsearch server.
        :param kwargs: Additional kwargs to pass to the Elasticsearch class.
        """

        # Third Party
        from elasticsearch import Elasticsearch

        self.corpus_name = corpus_name

        # Hosts is the minimum required param to init a connection to the
        # Elasticsearch server, so make it explicit here.
        self.hosts = host
        self.kwargs = kwargs

        self.es = Elasticsearch(hosts=host, **kwargs)

    def create_es_body(self, limit, query):
        """
        :param limit: Max number of documents to retrieve.
        :param query: Query string for retrieving documents.
        """

        body = {
            "size": limit,
            "query": {
                "bool": {
                    "must": {
                        "text_expansion": {
                            "ml.tokens": {
                                "model_id": ".elser_model_1",
                                "model_text": query,
                            }
                        }
                    }
                }
            },
        }
        return body

    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        body = self.create_es_body(top_k, query)

        retriever_results = self.es.search(
            index=self.corpus_name,
            body=body,
        )
        hits = retriever_results["hits"]["hits"]

        # Format for the processor.
        documents = []
        for hit in hits:
            document = Document(
                doc_id=hit["_id"],
                text=hit["_source"]["text"],
                score=hit["_score"],
            )
            documents.append(document)

        return documents
