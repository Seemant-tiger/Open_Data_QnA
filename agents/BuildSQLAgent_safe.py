import json
from abc import ABC
from datetime import datetime
import time

import pandas as pd
import vertexai
from dbconnectors import bqconnector, firestoreconnector, pgconnector
from google.cloud.aiplatform import telemetry
from utilities import PG_REGION, PROJECT_ID, PROMPTS, format_prompt
from vertexai.generative_models import Content, GenerationConfig, GenerativeModel, Part
from vertexai.language_models import CodeChatModel

from .core import Agent

vertexai.init(project=PROJECT_ID, location=PG_REGION)


class BuildSQLAgent(Agent, ABC):

    agentType: str = "BuildSQLAgent"

    def __init__(self, model_id="gemini-1.5-pro"):
        super().__init__(model_id=model_id)

    def build_sql(
        self,
        source_type,
        user_grouping,
        user_question,
        session_history,
        # tables_schema,
        # columns_schema,
        similar_sql,
        max_output_tokens=2048,
        temperature=0,
        top_p=1,
        top_k=32,
        context=None,
        logs_dict = {}
    ):
        connections_start_time = time.time()
        not_related_msg = (
            f"""select 'Question is not related to the dataset' as unrelated_answer;"""
        )

        if source_type == "bigquery":

            from dbconnectors import bq_specific_data_types

            specific_data_types = bq_specific_data_types()

        else:

            from dbconnectors import pg_specific_data_types

            specific_data_types = pg_specific_data_types()

        if f"usecase_{source_type}_{user_grouping}" in PROMPTS:
            usecase_context = PROMPTS[f"usecase_{source_type}_{user_grouping}"]
        else:
            usecase_context = "No extra context for the usecase is provided"

        if context:
            context_prompt = context
        else:
            context_prompt = PROMPTS[f"buildsql_{source_type}"]


        # Get the current date
        current_date = datetime.now().strftime("%Y-%m-%d")

        context_prompt = format_prompt(
            context_prompt,
            current_date=current_date,
            specific_data_types=specific_data_types,
            not_related_msg=not_related_msg,
            usecase_context=usecase_context,
            similar_sql=similar_sql,

            # tables_schema=tables_schema,
            # columns_schema=columns_schema,
        )

        # print(f"Prompt to Build SQL: \n{context_prompt}")

        # Chat history Retrieval

        batch_size = 10
        num_questions = len(session_history)
        if num_questions > 10:
            start_idx = num_questions - batch_size
            end_idx = num_questions
        else:
            start_idx = 0
            end_idx = num_questions

        # Slice session history to get the last 11 entries or fewer if not available
        relevant_history = session_history[start_idx:end_idx]

        # # Debugging statements
        # print(f"Total number of session history entries: {num_questions}")
        # print(f"Using session history from index {start_idx} to {end_idx}")
        # print("Session History Entries Used:")
        # for i, entry in enumerate(relevant_history, start=0):
        #     print(f"Entry {i}: {entry}")

        chat_history = []
        for entry in relevant_history:

            timestamp = entry["timestamp"]
            timestamp_str = timestamp.isoformat(timespec="auto")

            user_message = Content(
                parts=[Part.from_text(entry["user_question"])], role="user"
            )

            bot_message = Content(
                parts=[Part.from_text(entry["bot_response"])], role="assistant"
            )
            chat_history.extend([user_message, bot_message])  # Add both to the history
        logs_dict['SQL Building Time']['connections_history_loading'] = time.time() -connections_start_time
        # # Print chat history details
        # print("Chat History Built:")
        # for i, message in enumerate(chat_history, start=1):
        #     print(f"Message {i}: {message.parts[0].text}")

        # print("Chat History Retrieved")
        if self.model_id == "codechat-bison-32k":
            with telemetry.tool_context_manager("opendataqna-buildsql-v2"):

                chat_session = self.model.start_chat(context=context_prompt)
        elif "gemini" in self.model_id:
            set_context_start_time = time.time()
            with telemetry.tool_context_manager("opendataqna-buildsql-v2"):

                # print("SQL Builder Agent : " + str(self.model_id))
                config = GenerationConfig(
                    max_output_tokens=max_output_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                )
                chat_session = self.model.start_chat(
                    history=chat_history, response_validation=False
                )
                chat_session.send_message(context_prompt)
            set_context_end_time = time.time()
            logs_dict['SQL Building Time']['context_setting_time_build_sql'] = set_context_end_time-set_context_start_time

        else:
            raise ValueError("Invalid Model Specified")
        start_time = time.time()
        # if session_history is None or not session_history:
        #     # concated_questions = None
        #     # re_written_qe = None
        #     previous_question = None
        #     previous_sql = None

        # else:
        #     # concated_questions, re_written_qe = self.rewrite_question(
        #     #     user_question, session_history
        #     # )
        #     previous_question, previous_sql = self.get_last_sql(session_history)

        # build_context_prompt = f"""

        # Below is the previous user question from this conversation and its generated sql.

        # Previous Question:  {previous_question}

        # Previous Generated SQL : {previous_sql}

        # Respond with

        # Generate SQL for User Question : {user_question}

        # """
        build_context_prompt = f"""
        Generate SQL for User Question : {user_question}
        """


        # print("BUILD CONTEXT ::: "+str(build_context_prompt))

        with telemetry.tool_context_manager("opendataqna-buildsql-v2"):

            response = chat_session.send_message(build_context_prompt, stream=False)
            generated_sql = (
                (str(response.text)).replace("```sql", "").replace("```", "")
            )

        generated_sql = (str(response.text)).replace("```sql", "").replace("```", "")
        end_time = time.time()
        logs_dict['SQL Building Time']['initial_sql_generation_build_sql'] = end_time-start_time

        return generated_sql, context_prompt, logs_dict

    def rewrite_question(self, question, session_history):

        batch_size = 10

        total_questions = len(session_history)

        if total_questions > 10:
            start_index = total_questions - batch_size
            end_index = total_questions
        else:
            start_index = 0
            end_index = total_questions

        current_batch_history = session_history[start_index:end_index]

        # # Print the number of questions in history and their content
        # print(f"Total number of questions provided: {total_questions}")
        # print(f"Start index for history: {start_index}")
        # print(f"End index for history: {end_index}")
        # print(f"Number of questions used for history: {len(current_batch_history)}")

        # print("Questions used for history:")
        # for i, _row in enumerate(current_batch_history, start=1):
        #     print(f"Q{i}: {_row['user_question']}")

        formatted_history = ""
        concat_questions = ""
        for i, _row in enumerate(current_batch_history, start=1):
            user_question = _row["user_question"]
            sql_query = _row["bot_response"]
            # print(user_question)
            formatted_history += f"User Question - Turn :: {i} : {user_question}\n"
            formatted_history += f"Generated SQL - Turn :: {i}: {sql_query}\n\n"
            concat_questions += f"{user_question} "

        #print('-------------------------formatted_history-----------------------', formatted_history)

        # print(formatted_history)

        # context_prompt = f"""
        #     Your main objective is to rewrite and refine the question passed based on the session history of question and sql generated.

        #     Refine the given question using the provided session history to produce a queryable statement. The refined question should be self-contained, requiring no additional context for accurate SQL generation.

        #     Make sure all the information is included in the re-written question

        #     Below is the previous session history:

        #     {formatted_history}

        #     Question to rewrite:

        #     {question}
        # """

        # context_prompt = f"""
        #     Your task is to rewrite and refine the provided question based on the session history of questions and SQL responses.

        #     The rewritten question should be self-contained and standalone. This means that the question should not rely on any context from previous questions or information that is not included in the new question. Ensure that all relevant details are included to make the question clear and queryable on its own.

        #     If the current question is based on previous rewritten questions or session history (e.g., it references a concept or detail from a previous question), include those missing details directly in the rewritten question. This ensures that the new question does not need to rely on the history for clarity.

        #     Do not use any table or column names in the rewritten question. Instead, include relevant descriptive attributes directly.

        #     Provide only the rewritten question as the output.

        #     Below is the previous session history:

        #     {formatted_history}

        #     The question to rewrite is:

        #     {question}
        #     """
        context_prompt = f"""
            Your task is to rewrite and refine the given question using the session history of questions and SQL responses.

            1. The rewritten question should be self-contained and clear, requiring no additional context from previous questions or external sources.
            2. Ensure the rewritten question includes all relevant details from the original question to maintain thoroughness and precision.
            3. Avoid incorporating specific values, column names, or timelines not mentioned in the original question. Use session history only to fill in necessary details, but avoid adding extraneous information.
            4. Use session history selectively. Include only relevant details needed to accurately reflect and complete the original question.
            5. Maintain accuracy in reflecting the original question. Do not assume prior knowledge or context.
            6. Ensure that the rewritten question contains all pertinent information from the original question without omission.
            7. The rewritten question should be standalone and coherent, accurately representing the intent and details of the original question.

            Adhere closely to these guidelines to ensure the rewritten question is accurate and complete.

            Provide only the rewritten question as the output.

            Here is the session history:

            {formatted_history}

            The question to rewrite is:

            {question}
            """

        re_written_qe = str(self.generate_llm_response(context_prompt))

        print(
            "*" * 25
            + "Re-written question for the follow up:: "
            + "*" * 25
            + "\n"
            + str(re_written_qe)
        )

        return str(concat_questions), str(re_written_qe)

    def get_last_sql(self, session_history):

        for entry in reversed(session_history):
            if entry.get("bot_response"):
                return entry["user_question"], entry["bot_response"]

        return None
