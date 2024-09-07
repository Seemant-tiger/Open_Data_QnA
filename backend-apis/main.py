# -*- coding: utf-8 -*-


# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import datetime
import json
import logging as log
import os
import re
import sys
import textwrap
import time
import urllib

import pandas as pd
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from opendataqna import (
    embed_sql,
    generate_sql,
    get_all_databases,
    get_kgq,
    get_response,
    get_response_and_chart_type,
    get_results,
    round_and_convert,
    visualize,
    get_top_similar_examples
)

module_path = os.path.abspath(os.path.join('.'))
sys.path.append(module_path)


RUN_DEBUGGER = True
DEBUGGING_ROUNDS = 2
LLM_VALIDATION = False
EXECUTE_FINAL_SQL = True
Embedder_model = 'vertex'
SQLBuilder_model = 'gemini-1.5-flash'
SQLChecker_model = 'gemini-1.5-flash'
SQLDebugger_model = 'gemini-1.5-flash'
KGQMODEL = 'gemini-1.5-flash'
Responder_model="gemini-1.5-flash"
visualise_model = "gemini-1.5-flash"
num_table_matches = 5
num_column_matches = 10
table_similarity_threshold = 0.3
column_similarity_threshold = 0.3
example_similarity_threshold = 0.3
num_sql_matches = 3

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})



@app.route("/available_databases", methods=["GET"])
def getBDList():

    result,invalid_response=get_all_databases()

    if not invalid_response:
        responseDict = {
                "ResponseCode" : 200,
                "KnownDB" : result,
                "Error":""
                }

    else:
        responseDict = {
                "ResponseCode" : 500,
                "KnownDB" : "",
                "Error":result
                }
    return jsonify(responseDict)




# @app.route("/top_examples", methods=["POST"])
# async def topExamples():

#     envelope = str(request.data.decode('utf-8'))
#     envelope=json.loads(envelope)
#     user_question=envelope.get('user_question')


#     examples,logs= get_top_similar_examples(user_question, Embedder_model)

#     responseDict = {
#                "ResponseCode" : 200,
#                "examples" : examples,
#                "logs": logs

#                }
#     return jsonify(responseDict)




@app.route("/embed_sql", methods=["POST"])
async def embedSql():

    envelope = str(request.data.decode('utf-8'))
    envelope=json.loads(envelope)
    user_grouping=envelope.get('user_grouping')
    generated_sql = envelope.get('generated_sql')
    re_written_qe = envelope.get('re_written_qe')
    session_id = envelope.get('session_id')

    embedded, invalid_response=await embed_sql(session_id,user_grouping,re_written_qe,generated_sql, table = "example_prompt_sql_embeddings_thumbs_up")

    if not invalid_response:
        responseDict = {
                        "ResponseCode" : 201,
                        "Message" : "Example SQL has been accepted for embedding",
                        "SessionID" : session_id,
                        "Error":""
                        }
        return jsonify(responseDict)
    else:
        responseDict = {
                   "ResponseCode" : 500,
                   "KnownDB" : "",
                   "SessionID" : session_id,
                   "Error":embedded
                   }
        return jsonify(responseDict)




@app.route("/run_query", methods=["POST"])
def getSQLResult():

    envelope = str(request.data.decode('utf-8'))
    envelope=json.loads(envelope)

    user_question = envelope.get('user_question')
    user_grouping = envelope.get('user_grouping')
    generated_sql = envelope.get('generated_sql')
    session_id = envelope.get('session_id')
    re_written_qe = envelope.get('re_written_qe')
    doc_id = envelope.get('doc_id', None)

    # result_df,invalid_response=get_results(user_grouping,generated_sql)

    result_df, invalid_response,logs_dict = get_results(
            user_grouping,
            generated_sql,
            invalid_response=False,
            EXECUTE_FINAL_SQL=EXECUTE_FINAL_SQL,
            logs_dict={},
        )


    if not invalid_response:
        result_df = result_df.applymap(round_and_convert)

        # Convert the DataFrame to a list of dictionaries
        records = result_df.to_dict(orient="records")
        res_json = '\n'.join([str(i) for i in records])
        if res_json == '':
            res_json = 'No results returned'

        # _resp,invalid_response=get_response(session_id,user_question,result_df.to_json(orient='records'))
        print(visualise_model)
        _resp, invalid_response,logs_dict, charts_list = get_response_and_chart_type(session_id, generated_sql, user_question, re_written_qe,res_json, user_grouping,visualise_model= visualise_model, Responder_model=Responder_model, nl_prompt = None, logs_dict = {}, vis_prompt_1 = None, doc_id=doc_id)

        if not invalid_response:
            responseDict = {
                    "ResponseCode" : 200,
                    "KnownDB" : res_json,
                    "NaturalResponse" : _resp,
                    "SessionID" : session_id,
                    "ChartsList" : charts_list,
                    "Error":""
                    }
        else:
            responseDict = {
                    "ResponseCode" : 500,
                    "KnownDB" : res_json,
                    "NaturalResponse" : _resp,
                    "SessionID" : session_id,
                    "ChartsList" : charts_list,
                    "Error":""
                    }

    else:
        _resp=result_df
        responseDict = {
                "ResponseCode" : 500,
                "KnownDB" : "",
                "NaturalResponse" : _resp,
                "SessionID" : session_id,
                "ChartsList" : [],
                "Error":result_df,
                }
    return jsonify(responseDict)




@app.route("/get_known_sql", methods=["POST"])
def getKnownSQL():
    envelope = str(request.data.decode('utf-8'))
    envelope=json.loads(envelope)

    user_grouping = envelope.get('user_grouping')
    re_written_qe = envelope.get('re_written_qe')

    result, similar_sql_result, invalid_response, logs_dict=get_kgq(user_grouping, re_written_qe, Embedder_model, KGQMODEL, num_sql_matches=20)

    if not invalid_response:
        responseDict = {
                "ResponseCode" : 200,
                "KnownSQLFollowUps" : result,
                "similar_sql_result": similar_sql_result,
                "logs": logs_dict,
                "Error":""
                }

    else:
        responseDict = {
                "ResponseCode" : 500,
                "KnownSQLFollowUps" : "",
                "Error":result
                }
    return jsonify(responseDict)



@app.route("/generate_sql", methods=["POST"])
async def generateSQL():

    envelope = str(request.data.decode('utf-8'))
    #    print("Here is the request payload " + envelope)
    envelope=json.loads(envelope)

    user_question = envelope.get('user_question')
    user_grouping = envelope.get('user_grouping')
    session_id = envelope.get('session_id')
    user_id = envelope.get('user_id')
    (
        final_sql,
        session_id,
        invalid_response,
        similar_sql,
        table_matches,
        column_matches,
        context_prompt,
        logs_dict,
        audit_text,
        re_written_qe,
        doc_id
    ) = await generate_sql(
        session_id,
        user_question,
        user_grouping,
        RUN_DEBUGGER,
        DEBUGGING_ROUNDS,
        LLM_VALIDATION,
        Embedder_model,
        SQLBuilder_model,
        SQLChecker_model,
        SQLDebugger_model,
        # num_table_matches,
        # num_column_matches,
        # table_similarity_threshold,
        # column_similarity_threshold,
        example_similarity_threshold,
        num_sql_matches,
        context = None,
        logs_dict = {}
    )

    if not invalid_response:
        responseDict = {
                        "ResponseCode" : 200,
                        "GeneratedSQL" : final_sql,
                        "SessionID" : session_id,
                        "RewrittenQuestion": re_written_qe,
                        "Error":"",
                        "logs": logs_dict ,#delete later
                        "similar_sql": similar_sql, #delete_later
                        "doc_id": doc_id
                        }
    else:
        responseDict = {
                        "ResponseCode" : 500,
                        "GeneratedSQL" : "",
                        "SessionID" : session_id,
                        "RewrittenQuestion": re_written_qe,
                        "Error":final_sql
                        }

    return jsonify(responseDict)


@app.route("/generate_viz", methods=["POST"])
async def generateViz():
    envelope = str(request.data.decode('utf-8'))
    # print("Here is the request payload " + envelope)
    envelope=json.loads(envelope)

    user_question = envelope.get('user_question')
    generated_sql = envelope.get('generated_sql')
    sql_results = envelope.get('sql_results')
    session_id = envelope.get('session_id')
    re_written_qe = envelope.get('re_written_qe')
    charts_list = envelope.get('charts_list')
    chart_js=''

    try:
        # chart_js, invalid_response = visualize(session_id,user_question,generated_sql,sql_results)
        vis_prompt_1 = None
        vis_prompt_2 = None
        latency = {}
        start_time = time.time()
        chart_js, invalid_response = visualize(session_id, user_question, generated_sql, sql_results, vis_prompt_1 , vis_prompt_2, re_written_qe, charts_list, visualise_model = visualise_model)
        end_time = time.time()
        latency['visualize'] = end_time-start_time
        if not invalid_response:
            responseDict = {
            "ResponseCode" : 200,
            "GeneratedChartjs" : chart_js,
            "Error":"",
            "latency": latency,
            "SessionID":session_id
            }
        else:
            responseDict = {
                "ResponseCode" : 500,
                "GeneratedSQL" : "",
                "SessionID":session_id,
                "latency": latency,
                "Error": chart_js
                }


        return jsonify(responseDict)

    except Exception as e:
        # util.write_log_entry("Cannot generate the Visualization!!!, please check the logs!" + str(e))
        responseDict = {
                "ResponseCode" : 500,
                "GeneratedSQL" : "",
                "SessionID":session_id,
                "Error":"Issue was encountered while generating the Google Chart, please check the logs!"  + str(e)
                }
        return jsonify(responseDict)

@app.route("/summarize_results", methods=["POST"])
async def getSummary():
    envelope = str(request.data.decode('utf-8'))
    envelope=json.loads(envelope)

    user_question = envelope.get('user_question')
    sql_results = envelope.get('sql_results')

    result,invalid_response=get_response(user_question,sql_results)

    if not invalid_response:
        responseDict = {
                    "ResponseCode" : 200,
                    "summary_response" : result,
                    "Error":""
                    }

    else:
        responseDict = {
                    "ResponseCode" : 500,
                    "summary_response" : "",
                    "Error":result
                    }
    return jsonify(responseDict)




@app.route("/natural_response", methods=["POST"])
async def getNaturalResponse():
   start_time = time.time()
   envelope = str(request.data.decode('utf-8'))
   #print("Here is the request payload " + envelope)
   envelope=json.loads(envelope)

   user_question = envelope.get('user_question')
   user_grouping = envelope.get('user_grouping')
   session_id = envelope.get('session_id')
   user_id = envelope.get('user_id')
   sql_start_time = time.time()
   (
        generated_sql,
        session_id,
        invalid_response,
        similar_sql,
        table_matches,
        column_matches,
        context_prompt,
        logs_dict,
        audit_text,
        re_written_qe,
        doc_id
    ) = await generate_sql(
        session_id,
        user_question,
        user_grouping,
        RUN_DEBUGGER,
        DEBUGGING_ROUNDS,
        LLM_VALIDATION,
        Embedder_model,
        SQLBuilder_model,
        SQLChecker_model,
        SQLDebugger_model,
        # num_table_matches,
        # num_column_matches,
        # table_similarity_threshold,
        # column_similarity_threshold,
        example_similarity_threshold,
        num_sql_matches,
        context = None,
        logs_dict = {}
    )
   sql_end_time = time.time()
   if not invalid_response:

        # result_df,invalid_response=get_results(user_grouping,generated_sql)
        get_result_start_time = time.time()
        result_df, invalid_response,logs_dict = get_results(
                user_grouping,
                generated_sql,
                invalid_response=False,
                EXECUTE_FINAL_SQL=EXECUTE_FINAL_SQL,
                logs_dict=logs_dict,
            )

        get_result_end_time = time.time()
        if not invalid_response:
            get_resp_start_time = time.time()
            result_df = result_df.applymap(round_and_convert)

            # Convert the DataFrame to a list of dictionaries
            records = result_df.to_dict(orient="records")
            res_json = '\n'.join([str(i) for i in records])
            if res_json == '':
                res_json = 'No results returned'
            # _resp,invalid_response=get_response(session_id,user_question,result_df.to_json(orient='records'))
            _resp, invalid_response,logs_dict, charts_list = get_response_and_chart_type(session_id, generated_sql, user_question, re_written_qe,res_json, user_grouping, Embedder_model= Embedder_model, visualise_model= visualise_model, Responder_model=Responder_model, nl_prompt = None, logs_dict = logs_dict, vis_prompt_1 = None, doc_id=doc_id)

            get_resp_end_time = time.time()
            latency = {}
            latency['sql time'] = sql_end_time- sql_start_time
            latency['get result time'] = get_result_end_time - get_result_start_time
            latency['get response time'] = get_resp_end_time - get_resp_start_time
            latency['whole_time'] = time.time() -start_time
            latency['start_time'] = start_time
            if not invalid_response:
                responseDict = {
                            "ResponseCode" : 200,
                            "summary_response" : _resp,
                            "charts_list": charts_list,
                            "sql": generated_sql,
                            "logs": logs_dict,
                            "similar_sql": similar_sql,
                            'results_json': res_json,
                            "Error":"",
                            "latency": latency,
                            "re_written_qe": re_written_qe
                            }
            else:
                responseDict = {
                            "ResponseCode" : 500,
                            "summary_response" : "",
                            "Error":_resp,
                            "logs": logs_dict,
                            "latency": latency,
                            "re_written_qe": re_written_qe
                            }


        else:
            responseDict = {
                    "ResponseCode" : 500,
                    "KnownDB" : "",
                    "Error":result_df,
                    "logs": logs_dict,
                    "re_written_qe": re_written_qe

                    }

   else:
        responseDict = {
                        "ResponseCode" : 500,
                        "GeneratedSQL" : "",
                        "Error":generated_sql,
                        "logs": logs_dict,
                        "re_written_qe": re_written_qe
                        }

   return jsonify(responseDict)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))