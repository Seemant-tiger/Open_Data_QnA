from google.cloud import aiplatform


class VSConnector:
    def __init__(self, endpoint_name):
        """Initializes the Firestore connection and authentication."""
        self.index_endpoint = aiplatform.MatchingEngineIndexEndpoint(
            index_endpoint_name=endpoint_name
        )

    def get_top_ids(self, query_emb, top_n=3):
        try:
            # run query
            response = self.index_endpoint.find_neighbors(
                deployed_index_id="endpoint",
                queries=[query_emb],
                num_neighbors=top_n,
            )

            ids = [res.id for res in response[0]]
            invalid_response = False
        except:
            ids = []
            invalid_response = True

        return ids, invalid_response
