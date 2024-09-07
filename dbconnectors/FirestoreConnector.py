from google.cloud import firestore 
from google.cloud.exceptions import NotFound
import time
from dbconnectors import DBConnector
from abc import ABC
import uuid

def create_unique_id():
  """Creates a unique ID using the UUID4 algorithm.

  Returns:
    A string representing a unique ID.
  """

  return str(uuid.uuid1())


class FirestoreConnector(DBConnector, ABC):
    def __init__(self, 
                project_id:str, 
                firestore_database:str):
        """Initializes the Firestore connection and authentication."""
        self.db = firestore.Client(project=project_id,database=firestore_database)

    def log_chat(
        self,
        session_id,
        user_question,
        bot_response,
        user_id="TEST",
    ):
        """Logs a chat message to Firestore.
        Args:
            session_id (str): The ID of the chat session.
            user_id (str): The ID of the user who sent the message.
            user_question (str): The question the user asked.
            bot_response (str): The response from the bot.
        """

        log_chat = {
            "session_id": session_id,
            "user_id": user_id,
            "user_question": user_question,
            "bot_response": bot_response,
            "timestamp": firestore.SERVER_TIMESTAMP,
        }

        doc_ref = self.db.collection("session_logs").document()
        doc_ref.set(log_chat)

        return doc_ref.id

    def update_chat(self, doc_id, nl_response=None):
        """Updates an existing chat message in Firestore.

        Args:
            doc_id (str): The ID of the document to update.
            user_question (str, optional): The updated question the user asked.
            bot_response (str, optional): The updated response from the bot.
        """
        doc_ref = self.db.collection("session_logs").document(doc_id)
        update_data = {}

        if nl_response is not None:
            update_data["nl_response"] = nl_response

        if update_data:  # Only update if there's data to update
            doc_ref.update(update_data)

        print(f"Document {doc_id} updated successfully.")  
        
    def get_chat_logs_for_session(self,session_id):
        """Gets all chat logs for a given session.
        Args:
            session_id (str): The ID of the chat session.
        """

        sessions_log_ref = self.db.collection("session_logs")

        # sessions_log_ref=sessions_log_ref.order_by("timestamp")
        query= sessions_log_ref.where(filter=firestore.FieldFilter("session_id","==",session_id))
        
        # query = sessions_log_ref.where("session_id", "==", session_id).order_by("timestamp")  

        # Note: Use of CollectionRef stream() is prefered to get()
        docs = query.stream()

        session_history=[]
        for doc in docs:
            session_history.append(doc.to_dict())  # Add values to the list
        sorted_session_history=sorted(session_history,key=lambda x: x["timestamp"])

        return [{'user_question': item['user_question'], 'bot_response': item['bot_response'], 'nl_response': item.get('nl_response', ''),'timestamp':item['timestamp']} for item in sorted_session_history]
  
