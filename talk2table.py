# =========================================================
# IMPORT LIBRARIES
# =========================================================

import os
import sqlite3
import pandas as pd

from dotenv import load_dotenv

from typing import TypedDict

#from IPython.display import Image, display

#Matplotlib
import matplotlib.pyplot as plt


# LangChain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# LangGraph
from langgraph.graph import (
    StateGraph,
    END
)

# Rich UI
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box


# =========================================================
# ENVIRONMENT VARIABLES
# =========================================================

load_dotenv(".env")

OPENAI_API_KEY = os.getenv(
    "OPENAI_API_KEY"
)

# =========================================================
# LLM
# =========================================================

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# =========================================================
# CONSOLE
# =========================================================

console = Console()

console.print(
    Panel(
        "Text-to-SQL Chatbot Initialized",
        title="Chatbot Startup"
    )
)


# =========================================================
# LOAD USER CSV FILE FUNCTION
# =========================================================

def load_user_csv(
    csv_source,
    file_name
):

    TABLE_NAME = os.path.splitext(
        file_name
    )[0]

    TABLE_NAME = TABLE_NAME.replace(
        " ",
        "_"
    )

    TABLE_TITLE = " ".join(
        word.capitalize()
        for word in TABLE_NAME.replace(
            "_",
            " "
        ).split()
    )

    df = pd.read_csv(
        csv_source
    )

    return (
        df,
        TABLE_NAME,
        TABLE_TITLE
    )


# =========================================================
# CHATBOT STARTUP MESSAGE
# =========================================================

def show_welcome_message():

    console.print(

        Panel(

            f"Welcome to {TABLE_TITLE} Analyser Chatbot",

            title="Analyser Chatbot",

            border_style="green"

        )

    )


# =========================================================
# STATE
# =========================================================

class SQLState(TypedDict):

    # User Input
    question: str

    # Memory
    enhanced_question: str
    chat_history: list

    df: pd.DataFrame
    table_name: str
    table_title: str
    conn: object

    # Routing
    query_type: str

    # Schema Understanding
    schema: str
    column_context: str

    # SQL Generation
    sql_query: str

    # Visualization Planning
    is_visualization: bool
    chart_type: str
    x_axis: str
    y_axis: str
    group_by: str

    # SQL Execution Output
    result_df: pd.DataFrame

    # Chart Output
    chart_file: str

    # Final Response
    final_answer: str


# =========================================================
# CONVERSATION MEMORY AGENT
# =========================================================

def conversation_memory_agent(
    state: SQLState
):

    console.print(
        "\n[bold yellow]Conversation Memory Agent Running[/bold yellow]"
    )

    question = state[
        "question"
    ].strip()

    chat_history = state[
        "chat_history"
    ]

    question_lower = question.lower()

    # =====================================================
    # SMART MEMORY ACTIVATION
    # =====================================================

    strong_reference_words = [

        "he",
        "she",
        "it",
        "they",
        "them",
        "his",
        "her",
        "their",
        "this",
        "that",
        "these",
        "those",
        "former",
        "latter",
        "same"

    ]

    weak_reference_phrases = [

        "who",
        "what",
        "where",
        "when",
        "which",
        "how much",
        "how many"

    ]

    visualization_followup_words = [

        "pie chart",
        "bar chart",
        "line chart",
        "scatter chart",
        "chart",
        "graph",
        "plot",
        "visual",
        "visualization"

    ]

    has_strong_reference = any(

        word in question_lower.split()

        for word in strong_reference_words

    )

    has_weak_reference = any(

        phrase in question_lower

        for phrase in weak_reference_phrases

    )

    has_visualization_followup = any(

        phrase in question_lower

        for phrase in visualization_followup_words

    )

    is_short_question = (

        len(
            question_lower.split()
        ) <= 7

    )

    memory_required = (

        has_strong_reference

        or

        (

            len(chat_history) > 0

            and has_weak_reference

            and is_short_question

        )

        or

        (

            len(chat_history) > 0

            and has_visualization_followup

            and is_short_question

        )

    )

    if not memory_required:

        state[
            "enhanced_question"
        ] = question

        console.print(
            "[green]Memory Not Required[/green]"
        )

        console.print(

            Panel(

                question,

                title="Enhanced Question"

            )

        )

        return state

    # =====================================================
    # NO PREVIOUS HISTORY AVAILABLE
    # =====================================================

    if len(chat_history) == 0:

        state[
            "enhanced_question"
        ] = question

        console.print(
            "[green]No Previous History Found[/green]"
        )

        return state

    # =====================================================
    # DIRECT VISUALIZATION FOLLOW-UP FIX
    # =====================================================
    
    chart_followup_map = {
        "pie": "pie chart",
        "bar": "bar chart",
        "line": "line chart",
        "scatter": "scatter chart"
    }
    
    if (
        "it" in question_lower
        or "this" in question_lower
        or "same" in question_lower
    ):
    
        for chart_word, chart_phrase in chart_followup_map.items():
    
            if chart_word in question_lower:
    
                last_chat = chat_history[-1]
    
                previous_question = last_chat.get(
                    "enhanced_question",
                    last_chat.get(
                        "question",
                        ""
                    )
                )
    
                previous_question_lower = previous_question.lower()
    
                for old_chart in [
                    "bar chart",
                    "pie chart",
                    "line chart",
                    "scatter chart"
                ]:
    
                    if old_chart in previous_question_lower:
    
                        enhanced_question = previous_question_lower.replace(
                            old_chart,
                            chart_phrase
                        )
    
                        state["enhanced_question"] = enhanced_question
    
                        console.print(
                            Panel(
                                enhanced_question,
                                title="Enhanced Question"
                            )
                        )
    
                        return state
    
                enhanced_question = (
                    previous_question
                    + " as "
                    + chart_phrase
                )
    
                state["enhanced_question"] = enhanced_question
    
                console.print(
                    Panel(
                        enhanced_question,
                        title="Enhanced Question"
                    )
                )
    
                return state
    

    # =====================================================
    # USE ONLY RECENT CONVERSATIONS
    # =====================================================

    recent_history = chat_history[-3:]

    history_text = ""

    for i, chat in enumerate(

        recent_history,

        start=1

    ):

        history_text += f"""

Question {i}:
{chat.get("question", "")}

Enhanced Question {i}:
{chat.get("enhanced_question", chat.get("question", ""))}

Answer {i}:
{chat.get("answer", "")}

Query Type {i}:
{chat.get("query_type", "")}

SQL Query {i}:
{chat.get("sql_query", "")}

Visualization Required {i}:
{chat.get("is_visualization", False)}

Chart Type {i}:
{chat.get("chart_type", "")}

X Axis {i}:
{chat.get("x_axis", "")}

Y Axis {i}:
{chat.get("y_axis", "")}

Group By {i}:
{chat.get("group_by", "")}

"""

    prompt = PromptTemplate.from_template(
        """
You are a Conversation Memory Agent.

Your task is to rewrite the current
question using previous conversation
history ONLY when necessary.

Conversation History:

{history}

Current Question:

{question}

Rules:

1. Use conversation history only when
   the current question depends on
   previous context.

2. Explicit references include:

   he
   she
   it
   they
   them
   his
   her
   their
   this
   that
   these
   those
   former
   latter
   same

3. Short follow-up questions may also
   depend on conversation history,
   for example:

   who sold them

   what about the second one

   where was it sold

   how much was sold

   how many were sold

4. Visualization follow-up questions
   must reuse the previous analytical
   intent and change only the chart type
   if the user asks for a different chart.

5. Examples of visualization follow-ups:

   show it as pie chart

   convert it to pie chart

   make it a bar chart

   show the same as line chart

   change this to scatter chart

6. If the previous question was:

   show revenue by manager as a bar chart

   and the current question is:

   show it as pie chart

   rewrite as:

   show revenue by manager as a pie chart

7. If the previous question was:

   show sales by region as a bar chart

   and the current question is:

   convert it to pie chart

   rewrite as:

   show sales by region as a pie chart

8. If the current question is already
   complete and self-contained,
   return it unchanged.

9. Do NOT answer the question.

10. Do NOT introduce new topics.

11. Do NOT combine unrelated previous
    conversations.

12. Preserve the user's original intent.

13. Only rewrite the question.

14. If no rewrite is required,
    return the original question.

Return ONLY the rewritten question.
"""
    )

    chain = (

        prompt
        | llm
        | StrOutputParser()

    )

    enhanced_question = chain.invoke({

        "history": history_text,

        "question": question

    }).strip()

    if not enhanced_question:

        enhanced_question = question
    
    console.print(
    
        f"[bold cyan]Enhanced Question = {enhanced_question}[/bold cyan]"
    
    )
    
    state[
        "enhanced_question"
    ] = enhanced_question
    
    console.print(

        Panel(

            enhanced_question,

            title="Enhanced Question"

        )

    )

    return state

# =========================================================
# ROUTER AGENT
# =========================================================

def router_agent(state: SQLState):

    console.print(
        "\n[bold cyan]Router Agent Running[/bold cyan]"
    )

    df = state["df"]

    columns = list(
        df.columns
    )

    enhanced_question = state[
        "enhanced_question"
    ]

    question_lower = enhanced_question.lower()

    # =====================================================
    # DIRECT ROUTING SAFETY RULES
    # =====================================================

    visualization_words = [

        "bar chart",
        "pie chart",
        "line chart",
        "scatter chart",
        "chart",
        "graph",
        "plot",
        "visual",
        "visualization",
        "pictorial",
        "diagram",
        "representation"

    ]

    metadata_words = [

        "table name",
        "dataset name",
        "column names",
        "columns",
        "schema",
        "data types",
        "datatype",
        "structure",
        "number of rows",
        "number of columns",
        "how many rows",
        "how many columns",
        "table all about",
        "dataset all about",
        "overview"

    ]

    if any(

        word in question_lower

        for word in visualization_words

    ):

        state[
            "query_type"
        ] = "sql"

        console.print(
            "[green]Route → sql[/green]"
        )

        return state

    if any(

        word in question_lower

        for word in metadata_words

    ):

        state[
            "query_type"
        ] = "metadata"

        console.print(
            "[green]Route → metadata[/green]"
        )

        return state

    # =====================================================
    # LLM ROUTER
    # =====================================================

    prompt = PromptTemplate.from_template(
        """
You are a query routing agent.

Available Columns:

{columns}

User Question:

{enhanced_question}

Classify the question into ONLY ONE category:

metadata
sql
out_of_scope

Definitions:

metadata:

- asks table name
- asks dataset name
- asks column names
- asks available fields
- asks schema
- asks data types
- asks table structure
- asks number of rows
- asks number of columns
- asks dataset overview
- asks general information about the structure of the table

sql:

- requires reading data from the table
- requires retrieving values stored in rows
- requires retrieving information from one or more columns
- requires displaying records
- requires showing entries
- requires retrieving actual data values
- requires listing values present in the dataset
- requires listing unique values
- requires data retrieval
- requires filtering
- requires aggregation
- requires comparison
- requires ranking
- requires calculations
- requires trend analysis
- requires performance analysis
- requires identifying top or bottom performers
- requires summaries or insights from table contents
- requires answering questions using the data stored in the table
- requires analyzing actual table contents
- refers directly or indirectly to table columns
- requires finding relationships between columns
- requires sorting or ordering results
- requires grouping data
- requires statistical analysis
- asks for charts, graphs, plots, visualization,
  pictorial representation, bar chart, pie chart,
  line chart, or scatter chart based on table data

out_of_scope:

- unrelated to the uploaded table
- unrelated to the dataset contents
- cannot be answered using the available table data
- general knowledge questions
- current affairs
- news and recent events
- science and technology topics not present in the dataset
- politics and political discussions
- sports topics not present in the dataset
- entertainment topics not present in the dataset
- history questions not answerable from the dataset
- geography questions not answerable from the dataset
- mathematical problems unrelated to the dataset
- programming or coding questions unrelated to the dataset
- health and medical questions unrelated to the dataset
- legal or financial advice unrelated to the dataset
- philosophical or opinion-based questions
- personal advice requests
- creative writing requests
- story generation requests
- translation requests
- email, letter, or document writing requests
- questions about people, places, organizations, or events not contained in the dataset
- hypothetical questions not based on the dataset
- requests requiring external knowledge beyond the uploaded table
- questions whose answers cannot be derived from the table structure or table contents

Important routing rules:

1. If the question asks for any chart,
   graph, plot, visualization, pictorial
   representation, or diagram from the table,
   classify it as sql.

2. If the question mentions or indirectly
   refers to data columns, values, categories,
   regions, products, managers, ratings, sales,
   revenue, quantity, price, dates, months, or
   any table content, classify it as sql.

3. If the question is a rewritten follow-up
   like "show revenue by manager as a pie chart",
   classify it as sql.

Return ONLY ONE WORD:

metadata

OR

sql

OR

out_of_scope
"""
    )

    chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    query_type = chain.invoke({

        "columns": columns,

        "enhanced_question": enhanced_question

    }).strip().lower()

    if query_type not in [

        "metadata",

        "sql",

        "out_of_scope"

    ]:

        query_type = (
            "out_of_scope"
        )

    state[
        "query_type"
    ] = query_type

    console.print(
        f"[green]Route → {query_type}[/green]"
    )

    return state


# =========================================================
# METADATA AGENT
# =========================================================

def metadata_agent(state: SQLState):

    console.print(
        "\n[bold green]Metadata Agent Running[/bold green]"
    )

    df = state["df"]

    TABLE_NAME = state["table_name"]
    
    question = (
        state["enhanced_question"]
        .strip()
        .lower()
    )

    columns = list(df.columns)

    if (
        "table name" in question
        or "dataset name" in question
    ):

        answer = (
            f"Table Name : {TABLE_NAME}"
        )

    elif (
        "column" in question
        or "field" in question
        or "schema" in question
    ):

        answer = (
            "Available Columns:\n\n"
            + "\n".join(
                f"• {col}"
                for col in columns
            )
        )

    elif (
        "how many columns" in question
    ):

        answer = (
            f"Number of Columns : "
            f"{len(columns)}"
        )

    elif (
        "how many rows" in question
    ):

        answer = (
            f"Number of Rows : "
            f"{len(df)}"
        )

    else:

        answer = (
            f"The uploaded dataset '{TABLE_NAME}' contains "
            f"{df.shape[0]} rows and {df.shape[1]} columns.\n\n"
            f"The available columns are:\n\n"
            + "\n".join(
                f"• {col}"
                for col in df.columns
            )
        )
    state["final_answer"] = (
        answer
    )

    return state


# =========================================================
# OUT OF SCOPE AGENT
# =========================================================

def out_of_scope_agent(state: SQLState):

    console.print(
        "\n[bold red]Out Of Scope Agent Running[/bold red]"
    )

    df = state["df"]

    TABLE_NAME = state["table_name"]

    columns = list(
        df.columns
    )

    state["final_answer"] = f"""
I can answer questions only from the uploaded table.

Table Name:
{TABLE_NAME}

Available Columns:

{', '.join(columns)}

Please ask questions related to these columns.
"""

    return state


# =========================================================
# SCHEMA AGENT
# =========================================================

def schema_agent(state: SQLState):

    console.print(
        "\n[bold yellow]Schema Agent Running[/bold yellow]"
    )

    conn = state["conn"]

    TABLE_NAME = state["table_name"]
    
    cursor = conn.cursor()

    cursor.execute(
        f"PRAGMA table_info({TABLE_NAME})"
    )

    columns_info = cursor.fetchall()

    schema = []

    for column in columns_info:

        col_name = column[1]

        col_type = column[2]

        schema.append(
            f"{col_name} ({col_type})"
        )

    schema_text = (
        f"Table Name : {TABLE_NAME}\n\n"
        + "\n".join(schema)
    )

    state["schema"] = schema_text

    console.print(
        Panel(
            schema_text,
            title="Table Schema"
        )
    )

    return state


# =========================================================
# COLUMN UNDERSTANDING AGENT
# =========================================================

def column_understanding_agent(
    state: SQLState
):

    console.print(
        "\n[bold magenta]Column Understanding Agent Running[/bold magenta]"
    )

    df = state["df"]

    enhanced_question = state[
        "enhanced_question"
    ]

    columns = list(
        df.columns
    )

    # =====================================================
    # DYNAMIC SAMPLE VALUES
    # =====================================================

    sample_values_text = ""

    for col in df.columns:

        samples = (

            df[col]
            .dropna()
            .astype(str)
            .unique()[:5]

        )

        sample_values_text += f"""

{col}:

{', '.join(samples)}

"""

    prompt = PromptTemplate.from_template(
        """
You are an expert database analyst and business analyst.

Table Schema:

{schema}

User Question:

{enhanced_question}

Available Columns:

{columns}

Sample Values From Dataset:

{sample_values}

Your job:

1. Study all available columns.

2. Understand the user's intent.

3. Identify which columns are relevant.

4. Infer semantic meaning.

5. User may not use exact column names.

6. Match user terminology to the closest
   matching column names or sample values.

7. Use semantic understanding instead of
   exact keyword matching.

8. Consider both column names and sample values.

9. Identify possible aggregations.

10. Identify possible ranking/filtering logic.

11. Infer business meaning from the data.

Examples:

Question:
Which product generated highest revenue?

Output:

Relevant Columns:
Product_Name
Revenue

Intent:
Find highest revenue product

Business Mapping:
None


Question:
Who performs best?

Output:

Relevant Columns:
Manager
Revenue

Intent:
Find best-performing manager

Business Mapping:
performs best → highest revenue


Question:
What gadgets were sold in north?

Dataset Values:

Category:
Electronics
Furniture
Appliances

Region:
North
South

Output:

Relevant Columns:
Category
Region
Product_Name

Intent:
Find products sold in North region
belonging to Electronics category

Business Mapping:
gadgets → Electronics


Question:
Show premium items.

Dataset Values:

Category:
Premium
Standard
Economy

Output:

Relevant Columns:
Category

Intent:
Show Premium category records

Business Mapping:
premium items → Premium

Return ONLY in the following format:

Relevant Columns:

Column1
Column2
Column3

Intent:

<clear interpretation of user request>

Business Mapping:

<mappings identified>

If no mappings exist:

Business Mapping:

None
"""
    )

    chain = (

        prompt
        | llm
        | StrOutputParser()

    )

    column_context = chain.invoke({

        "schema": state["schema"],

        "enhanced_question": enhanced_question,

        "columns": "\n".join(
            columns
        ),

        "sample_values": sample_values_text

    })

    state[
        "column_context"
    ] = column_context

    console.print(

        Panel(

            column_context,

            title="Column Understanding"

        )

    )

    return state


# =========================================================
# SQL GENERATOR AGENT
# =========================================================

def sql_generator_agent(state: SQLState):

    console.print(
        "\n[bold cyan]SQL Generator Agent Running[/bold cyan]"
    )

    TABLE_NAME = state["table_name"]

    prompt = PromptTemplate.from_template(
        """
You are an expert senior SQLite SQL Generator.

Table Schema:

{schema}

Column Understanding:

{column_context}

User Question:

{enhanced_question}

Rules:

1. Use ONLY the table:
   {table_name}

2. Use ONLY columns present in the schema.

3. Generate ONLY SQLite-compatible SQL.

4. Return ONLY SQL query.

5. Do NOT return:
   - explanations
   - markdown
   - comments
   - code fences
   - notes
   - reasoning

6. Do NOT use:
   - regular expressions
   - REGEXP
   - Python code
   - pseudocode

7. Always generate executable SQL.

8. If aggregation is requested:
   Use COUNT, SUM, AVG, MIN, MAX.

9. If sorting is required:
   Use ORDER BY.

10. If top records are requested:
    Use LIMIT.

11. If filtering is required:
    Use WHERE.

12. If grouping is required:
    Use GROUP BY.

13. If the question refers to a column indirectly,
    infer the correct column from schema.

14. If multiple columns have similar meaning,
    choose the most relevant one.

15. Column names must exactly match schema.

16. Never invent:
    - tables
    - columns
    - values

17. If question cannot be answered from schema,
    return exactly:

    INVALID_QUERY

18. If user asks:
    - all records
    - show dataset
    - display table
    - show data

    generate:

    SELECT * FROM {table_name};

19. If user asks row count:

    SELECT COUNT(*) FROM {table_name};

20. SQL must end with semicolon.

Return ONLY SQL.
"""
    )

    chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    sql_query = chain.invoke({

        "schema": state["schema"],
    
        "column_context": state[
            "column_context"
        ],
    
        "enhanced_question": state["enhanced_question"],
    
        "table_name": TABLE_NAME
    
    })
    sql_query = sql_query.strip()

    state["sql_query"] = sql_query

    console.print(
        Panel(
            sql_query,
            title="Generated SQL"
        )
    )

    return state


# =========================================================
# SQL VALIDATOR AGENT
# =========================================================

def sql_validator_agent(
    state: SQLState
):

    console.print(
        "\n[bold green]SQL Validator Agent Running[/bold green]"
    )

    df = state["df"]

    TABLE_NAME = state["table_name"]

    conn = state["conn"]

    sql_query = (
        state["sql_query"]
        .replace("```sql", "")
        .replace("```", "")
        .strip()
    )

    if not sql_query:

        state["final_answer"] = (
            "No SQL query generated."
        )

        return state

    if sql_query == "INVALID_QUERY":

        state["final_answer"] = f"""
Unable to generate a valid SQL query.

Please ask questions related to:

{', '.join(df.columns)}
"""

        return state

    dangerous_keywords = [

        "DROP",
        "DELETE",
        "TRUNCATE",
        "ALTER",
        "UPDATE",
        "INSERT",
        "CREATE",
        "REPLACE",
        "ATTACH",
        "DETACH",
        "PRAGMA"

    ]

    upper_sql = sql_query.upper()

    for keyword in dangerous_keywords:

        if keyword in upper_sql:

            state["final_answer"] = (
                "Unsafe SQL detected."
            )

            state["sql_query"] = ""

            return state

    if TABLE_NAME.lower() not in (
        upper_sql.lower()
    ):

        state["final_answer"] = (
            "Generated SQL does not use the uploaded table."
        )

        state["sql_query"] = ""

        return state

    try:

        validation_query = f"""

        SELECT *
        FROM (
            {sql_query.rstrip(';')}
        )
        LIMIT 1

        """

        pd.read_sql_query(
            validation_query,
            conn
        )

    except Exception as e:

        state["final_answer"] = (
            f"SQL Validation Failed: {str(e)}"
        )

        state["sql_query"] = ""

        return state

    state["sql_query"] = sql_query

    console.print(
        "[green]SQL Validation Passed[/green]"
    )

    console.print(
        f"\nValidated SQL:\n{sql_query}"
    )

    return state


# =========================================================
# SQL EXECUTOR AGENT
# =========================================================

def sql_executor_agent(state: SQLState):

    console.print(
        "\n[bold blue]SQL Executor Agent Running[/bold blue]"
    )

    console.print(
        f"\nSQL Received By Executor:\n{state['sql_query']}"
    )

    df = state["df"]

    conn = state["conn"]

    sql_query = (
        state["sql_query"]
        .strip()
    )

    if not sql_query:

        console.print(
            "[red]No SQL Query Found[/red]"
        )

        return state

    try:

        result_df = pd.read_sql_query(
            sql_query,
            conn
        )

        state["result_df"] = (
            result_df
        )

        console.print(

            Panel(

                f"Rows Returned : {len(result_df)}\n"
                f"Columns Returned : {len(result_df.columns)}",

                title="SQL Execution Summary"

            )

        )

        if not result_df.empty:

            preview_table = Table(
                title="Query Result Preview",
                box=box.DOUBLE_EDGE
            )

            for col in result_df.columns:

                preview_table.add_column(
                    str(col)
                )

            preview_rows = min(
                len(result_df),
                10
            )

            for _, row in (
                result_df
                .head(preview_rows)
                .iterrows()
            ):

                preview_table.add_row(

                    *[
                        str(value)
                        for value in row
                    ]

                )

            console.print(
                preview_table
            )

        else:

            console.print(

                "[yellow]Query executed successfully but returned no records.[/yellow]"

            )

    except Exception as e:

        state["final_answer"] = (
            f"SQL Execution Failed : {str(e)}"
        )

        state["result_df"] = (
            pd.DataFrame()
        )

        console.print(

            Panel(

                str(e),

                title="SQL Execution Error",

                border_style="red"

            )

        )
    
    return state


# =========================================================
# VISUALIZATION AGENT
# =========================================================

def visualization_agent(
    state: SQLState
):

    console.print(
        "\n[bold magenta]Visualization Agent Running[/bold magenta]"
    )

    df = state["df"]

    question = state[
        "enhanced_question"
    ]

    result_df = state[
        "result_df"
    ]

    if result_df.empty:

        state[
            "is_visualization"
        ] = False

        return state

    prompt = PromptTemplate.from_template(
    """
You are an expert Data Visualization Planner.

User Question:

{enhanced_question}

Available Columns:

{columns}

Your task is to determine:

1. Whether visualization is required.
2. Which chart type is most suitable.
3. Which column should be X-axis.
4. Which column should be Y-axis.
5. Whether a grouping column is required.

Visualization Detection Rules:

Questions containing words such as:

graph
chart
plot
visualize
visualization
dashboard
pictorial
trend

should be treated as visualization requests.

Chart Selection Rules:

1. Time or Date based analysis
   -> line chart

2. Category vs Numeric analysis
   -> bar chart

3. Share, percentage, composition,
   distribution
   -> pie chart

4. Relationship between two numeric columns
   -> scatter chart

Axis Selection Rules:

1. Choose ONE categorical column
   as X-axis.

2. Choose ONE numeric column
   as Y-axis.

3. If another categorical column
   helps comparison,
   use it as Group By.

4. For Date/Time analysis:

   X-axis = Date/Time column

   Y-axis = Relevant numeric column

   Chart = line

5. For two numeric columns:

   Chart = scatter

6. If multiple numeric columns exist:

   Select the numeric column
   most relevant to the user's question.

7. If multiple categorical columns exist:

   Select the primary category
   mentioned in the user's question
   as X-axis.

   Select the secondary category
   as Group By.

8. If no grouping is needed:

   group_by = None

Examples:

Example 1

Columns:
Category, Numeric

Result:

X-axis = Category

Y-axis = Numeric

Group By = None

Chart = bar


Example 2

Columns:
Category, Numeric, Category

Result:

X-axis = Primary Category

Y-axis = Numeric

Group By = Secondary Category

Chart = bar


Example 3

Columns:
Date, Numeric

Result:

X-axis = Date

Y-axis = Numeric

Group By = None

Chart = line


Example 4

Columns:
Numeric, Numeric

Result:

X-axis = First Numeric

Y-axis = Second Numeric

Group By = None

Chart = scatter

Important:

- Use ONLY column names from Available Columns.
- Never invent column names.
- Choose the columns that best answer the user's question.
- Return exactly in the format shown below.
- Do not provide explanations.

Return ONLY:

visualization=yes
chart_type=bar
x_axis=<column_name>
y_axis=<column_name>
group_by=<column_name_or_None>

OR

visualization=no
chart_type=None
x_axis=None
y_axis=None
group_by=None
"""
    )
    
    chain = (

        prompt
        | llm
        | StrOutputParser()

    )

    response = chain.invoke({

        "enhanced_question": question,

        "columns": list(
            result_df.columns
        )

    })

    visualization = False
    chart_type = ""
    x_axis = ""
    y_axis = ""
    group_by = ""

    for line in response.splitlines():

        line = line.strip()

        if "=" not in line:

            continue

        key, value = line.split(
            "=",
            1
        )

        key = key.strip().lower()

        value = value.strip()

        if key == "visualization":

            visualization = (
                value.lower() == "yes"
            )

        elif key == "chart_type":

            chart_type = value

        elif key == "x_axis":

            x_axis = value

        elif key == "y_axis":

            y_axis = value

        elif key == "group_by":

            if value.lower() != "none":

                group_by = value

    valid_columns = {

        col.lower(): col

        for col in result_df.columns

    }

    if x_axis.lower() in valid_columns:

        x_axis = valid_columns[
            x_axis.lower()
        ]

    else:

        x_axis = result_df.columns[0]

    if y_axis.lower() in valid_columns:

        y_axis = valid_columns[
            y_axis.lower()
        ]

    else:

        numeric_cols = result_df.select_dtypes(
            include="number"
        ).columns

        if len(numeric_cols):

            y_axis = numeric_cols[0]

        else:

            y_axis = result_df.columns[-1]

    if (

        group_by
        and group_by.lower()
        in valid_columns

    ):

        group_by = valid_columns[
            group_by.lower()
        ]

    else:

        group_by = ""

    if chart_type not in [

        "bar",
        "line",
        "pie",
        "scatter"

    ]:

        chart_type = "bar"

    state[
        "is_visualization"
    ] = visualization

    state[
        "chart_type"
    ] = chart_type

    state[
        "x_axis"
    ] = x_axis

    state[
        "y_axis"
    ] = y_axis

    state[
        "group_by"
    ] = group_by

    console.print(

        Panel(

            f"""
Chart Type : {chart_type}

X Axis : {x_axis}

Y Axis : {y_axis}

Group By : {group_by if group_by else 'None'}
""",

            title="Visualization Plan"

        )

    )

    return state


# =========================================================
# CHART GENERATOR AGENT
# =========================================================

def chart_generator_agent(
    state: SQLState
):

    console.print(
        "\n[bold blue]Chart Generator Agent Running[/bold blue]"
    )

    if not state[
        "is_visualization"
    ]:

        return state

    df = state["df"]

    TABLE_NAME = state["table_name"]

    TABLE_TITLE = state["table_title"]

    result_df = state[
        "result_df"
    ]

    if result_df.empty:

        state[
            "final_answer"
        ] = (
            "No data available for visualization."
        )

        return state

    chart_type = state[
        "chart_type"
    ]

    x_col = state[
        "x_axis"
    ]

    y_col = state[
        "y_axis"
    ]

    group_col = state[
        "group_by"
    ]

    if (
        x_col not in result_df.columns
        or
        y_col not in result_df.columns
    ):

        state[
            "final_answer"
        ] = (
            "Unable to identify suitable "
            "columns for visualization."
        )

        return state

    plt.figure(
        figsize=(12, 6)
    )

    # =====================================================
    # BAR CHART
    # =====================================================

    if chart_type == "bar":

        colors = plt.cm.tab20.colors

        if group_col and group_col in result_df.columns:

            pivot_df = result_df.pivot_table(
                index=x_col,
                columns=group_col,
                values=y_col,
                aggfunc="sum"
            )

            pivot_df.plot(
                kind="bar",
                figsize=(12, 6),
                colormap="tab20"
            )

        else:

            plt.bar(
                result_df[x_col],
                result_df[y_col],
                color=colors[:len(result_df)]
            )

    # =====================================================
    # LINE CHART
    # =====================================================

    elif chart_type == "line":

        colors = plt.cm.tab20.colors

        if (
            group_col
            and
            group_col in result_df.columns
        ):

            for i, (
                group,
                group_df
            ) in enumerate(

                result_df.groupby(
                    group_col
                )

            ):

                plt.plot(
                    group_df[x_col],
                    group_df[y_col],
                    label=str(group),
                    color=colors[
                        i % len(colors)
                    ],
                    marker="o",
                    linewidth=2
                )

            plt.legend()

        else:

            plt.plot(
                result_df[x_col],
                result_df[y_col],
                color=plt.cm.Dark2(0),
                marker="o",
                linewidth=2
            )

    # =====================================================
    # PIE CHART
    # =====================================================

    elif chart_type == "pie":

        colors = plt.cm.Set3.colors

        plt.pie(
            result_df[y_col],
            labels=result_df[x_col],
            colors=colors[
                :len(result_df)
            ],
            autopct="%1.1f%%"
        )

    # =====================================================
    # SCATTER CHART
    # =====================================================

    elif chart_type == "scatter":

        plt.scatter(
            result_df[x_col],
            result_df[y_col],
            c=range(
                len(result_df)
            ),
            cmap="tab10",
            s=100
        )

    # =====================================================
    # COMMON FORMATTING
    # =====================================================

    plt.title(
        state["question"]
    )

    plt.xlabel(
        x_col
    )

    plt.ylabel(
        y_col
    )

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.tight_layout()

    chart_file = (
        f"{TABLE_TITLE}_chart.png"
    )

    plt.savefig(
        chart_file,
        bbox_inches="tight"
    )

    plt.show()

    plt.close()

    state[
        "chart_file"
    ] = chart_file

    console.print(

        f"[green]Chart Saved : "
        f"{chart_file}[/green]"

    )

    return state


# =========================================================
# RESPONSE AGENT
# =========================================================

def response_agent(state: SQLState):

    console.print(
        "\n[bold magenta]Response Agent Running[/bold magenta]"
    )

    result_df = state["result_df"]

    question = state["enhanced_question"]

    if result_df is None:

        state["final_answer"] = (
            "No result available."
        )

        return state

    if result_df.empty:

        state["final_answer"] = (
            "No records found."
        )

        return state

    result_text = result_df.head(10).to_string(
        index=False
    )

    prompt = PromptTemplate.from_template(
        """
You are a data response assistant.

User Question:
{question}

Query Result:
{result_text}

Your task:
Convert the query result into a clear natural language answer.

Rules:
- Do not return raw dataframe format.
- Do not mention SQL.
- Use short, simple sentences.
- If there are multiple rows, summarize them clearly.
- Use bullet points when helpful.
- If the result contains names, products, regions, categories or values, include them.
- If the result is a single value, answer directly.
- Do not invent information.

Return only the final answer.
"""
    )

    chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    final_answer = chain.invoke({

        "question": question,

        "result_text": result_text

    })

    state["final_answer"] = final_answer

    return state

# =========================================================
# BUILD LANGGRAPH WORKFLOW
# =========================================================

builder = StateGraph(SQLState)

# ------------------------------------------
# Register Nodes
# ------------------------------------------

builder.add_node(
    "Memory",
    conversation_memory_agent
)

builder.add_node(
    "Router",
    router_agent
)

builder.add_node(
    "Metadata",
    metadata_agent
)

builder.add_node(
    "OutOfScope",
    out_of_scope_agent
)

builder.add_node(
    "Schema",
    schema_agent
)

builder.add_node(
    "ColumnUnderstanding",
    column_understanding_agent
)

builder.add_node(
    "SQLGenerator",
    sql_generator_agent
)

builder.add_node(
    "SQLValidator",
    sql_validator_agent
)

builder.add_node(
    "SQLExecutor",
    sql_executor_agent
)

builder.add_node(
    "Visualization",
    visualization_agent
)

builder.add_node(
    "ChartGenerator",
    chart_generator_agent
)

builder.add_node(
    "Response",
    response_agent
)

# ------------------------------------------
# Entry Point
# ------------------------------------------

builder.set_entry_point(
    "Memory"
)


# =========================================================
# ROUTING FUNCTION
# =========================================================

# Route Question

def route_question(state: SQLState):

    """
    Reads the route assigned by Router Agent
    and returns the next node name for LangGraph.
    """

    query_type = (
        state["query_type"]
        .strip()
        .lower()
    )

    valid_routes = {

        "metadata",
        "sql",
        "out_of_scope"

    }

    if query_type not in valid_routes:

        console.print(

            "[red]Invalid Route Detected[/red]"

        )

        return "out_of_scope"

    return query_type

    
#Route Visualization

def route_visualization(
    state: SQLState
):

    question = state[
        "question"
    ].lower()

    visualization_keywords = [

        "graph",
        "chart",
        "plot",
        "visualize",
        "visualization",
        "dashboard",
        "trend",
        "pictorial"

    ]

    for keyword in visualization_keywords:

        if keyword in question:

            return "visualization"

    return "response"


# ------------------------------------------
# Adding Edges
# ------------------------------------------

builder.add_edge(
    "Memory",
    "Router"
)

# ------------------------------------------
# Conditional Routing
# ------------------------------------------

builder.add_conditional_edges(

    "Router",

    route_question,

    {

        "metadata": "Metadata",

        "sql": "Schema",

        "out_of_scope": "OutOfScope"

    }

)

# ------------------------------------------
# Metadata Path
# ------------------------------------------

builder.add_edge(
    "Metadata",
    END
)

# ------------------------------------------
# Out Of Scope Path
# ------------------------------------------

builder.add_edge(
    "OutOfScope",
    END
)

# ------------------------------------------
# SQL Workflow Path
# ------------------------------------------

builder.add_edge(
    "Schema",
    "ColumnUnderstanding"
)

builder.add_edge(
    "ColumnUnderstanding",
    "SQLGenerator"
)
builder.add_edge(
    "SQLGenerator",
    "SQLValidator"
)

builder.add_edge(
    "SQLValidator",
    "SQLExecutor"
)

builder.add_conditional_edges(

    "SQLExecutor",

    route_visualization,

    {

        "visualization":
            "Visualization",

        "response":
            "Response"

    }

)

builder.add_edge(
    "Visualization",
    "ChartGenerator"
)


builder.add_edge(
    "ChartGenerator",
    "Response"
)

builder.add_edge(
    "Response",
    END
)

# ------------------------------------------
# Compile Graph
# ------------------------------------------

graph = builder.compile()


# =========================================================
# STREAMLIT QUERY RUNNER FUNCTION
# =========================================================

def run_talk2table_query(
    user_question,
    chat_history,
    df,
    table_name,
    table_title,
    conn
):

        question_lower = user_question.lower()

    chart_words = {
        "pie": "pie chart",
        "bar": "bar chart",
        "line": "line chart",
        "scatter": "scatter chart"
    }

    if (
        len(chat_history) > 0
        and (
            "it" in question_lower
            or "this" in question_lower
            or "same" in question_lower
        )
    ):

        for chart_word, chart_phrase in chart_words.items():

            if chart_word in question_lower:

                last_question = chat_history[-1].get(
                    "enhanced_question",
                    chat_history[-1].get(
                        "question",
                        ""
                    )
                )

                last_question_lower = last_question.lower()

                for old_chart in [
                    "bar chart",
                    "pie chart",
                    "line chart",
                    "scatter chart"
                ]:

                    if old_chart in last_question_lower:

                        user_question = last_question_lower.replace(
                            old_chart,
                            chart_phrase
                        )

                        break

                else:

                    user_question = (
                        last_question
                        + " as "
                        + chart_phrase
                    )

                break
    
    result = graph.invoke({

        "question": user_question,

        "enhanced_question": "",

        "chat_history": chat_history,

        "df": df,

        "table_name": table_name,

        "table_title": table_title,

        "conn": conn,

        "query_type": "",

        "schema": "",

        "column_context": "",

        "sql_query": "",

        "is_visualization": False,

        "chart_type": "",

        "x_axis": "",

        "y_axis": "",

        "group_by": "",

        "chart_file": "",

        "result_df": pd.DataFrame(),

        "final_answer": ""

    })

    chat_history.append({

        "question": user_question,

        "answer": result["final_answer"],

        "enhanced_question": result.get(
            "enhanced_question",
            user_question
        ),

        "query_type": result.get(
            "query_type",
            ""
        ),

        "sql_query": result.get(
            "sql_query",
            ""
        ),

        "is_visualization": result.get(
            "is_visualization",
            False
        ),

        "chart_type": result.get(
            "chart_type",
            ""
        ),

        "x_axis": result.get(
            "x_axis",
            ""
        ),

        "y_axis": result.get(
            "y_axis",
            ""
        ),

        "group_by": result.get(
            "group_by",
            ""
        ),

        "chart_file": result.get(
            "chart_file",
            ""
        )

    })

    return result

# =========================================================
# CHAT LOOP
# =========================================================

if __name__ == "__main__":

    CSV_FILE = input(
        "\nEnter CSV file path: "
    ).strip()
    
    CSV_FILE = CSV_FILE.replace(
        "\\ ",
        " "
    )
    
    df, TABLE_NAME, TABLE_TITLE = load_user_csv(
        CSV_FILE,
        os.path.basename(CSV_FILE)
    )

    console.print(
        f"\nTable '{TABLE_NAME}' created successfully."
    )

    # =========================================================
    # CREATE SQLITE DATABASE
    # =========================================================

    DB_NAME = (
        f"{TABLE_TITLE}.db"
    )
    
    conn = sqlite3.connect(
        DB_NAME
    )
    
    df.to_sql(
        name=TABLE_NAME,
        con=conn,
        if_exists="replace",
        index=False
    )
    
    console.print(
        f"\nRows : {df.shape[0]}"
    )
    
    console.print(
        f"Columns : {df.shape[1]}"
    )

    show_welcome_message()
    
    chat_history = []
    
    while True:
    
        user_question = input(
            "\nAsk your question : "
        ).strip()
    
        if user_question.lower() in [
    
            "exit",
            "quit",
            "bye"
    
        ]:
    
            console.print(
                f"\n[bold red]Exiting {TABLE_TITLE} Analyser Chatbot[/bold red]"
            )
    
            break
    
        try:
    
            result = run_talk2table_query(
                user_question,
                chat_history
            )            
            
            console.print(
    
                Panel(
    
                    str(
                        result["final_answer"]
                    ),
    
                    title="Final Answer",
    
                    border_style="green"
    
                )
    
            )
        
        except Exception as e:
    
            console.print(
    
                Panel(
    
                    str(e),
    
                    title="Workflow Error",
    
                    border_style="red"
    
                )
    
            )

    