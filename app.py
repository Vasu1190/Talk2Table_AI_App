from talk2table import (
    run_talk2table_query
)

import streamlit as st
import pandas as pd
import sqlite3
import os


st.set_page_config(
    page_title="Talk2Table AI",
    layout="wide"
)

if "messages" not in st.session_state:

    st.session_state.messages = []

if "chat_history" not in st.session_state:

    st.session_state.chat_history = []


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("Talk2Table AI")

    if st.button("🗑️ New Chat"):

        st.session_state.messages = []

        st.session_state.chat_history = []

        st.rerun()


# =========================================================
# MAIN PAGE
# =========================================================

st.title("Talk2Table AI")

st.subheader(
    "Analyze preprocessed CSV data using natural language instead of SQL"
)

uploaded_file = st.file_uploader(
    "Upload your CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.write("### Dataset Preview")

    st.dataframe(
        df.head(20),
        width="stretch"
    )

    st.write("### Dataset Information")

    st.write(f"Rows: {df.shape[0]}")

    st.write(f"Columns: {df.shape[1]}")

    st.write("### Column Names")

    st.table(
        pd.DataFrame(
            {
                "Column Names": df.columns
            }
        )
    )

    TABLE_NAME = uploaded_file.name.replace(
        ".csv",
        ""
    ).replace(
        " ",
        "_"
    )

    conn = sqlite3.connect(
        ":memory:"
    )

    df.to_sql(
        name=TABLE_NAME,
        con=conn,
        if_exists="replace",
        index=False
    )

    st.success(
        f"SQLite table created successfully: {TABLE_NAME}"
    )

    cursor = conn.cursor()

    cursor.execute(
        f"PRAGMA table_info({TABLE_NAME})"
    )

    schema_info = cursor.fetchall()

    schema = []

    for col in schema_info:

        schema.append(
            f"{col[1]} ({col[2]})"
        )

    schema_text = (
        "Table Name : "
        + TABLE_NAME
        + "\n\n"
        + "\n".join(schema)
    )

    st.write("### Table Schema")

    st.code(
        schema_text
    )

    st.write("### Ask Questions About Your Data")

    user_question = st.chat_input(
        "Ask your question here..."
    )

    for i, message in enumerate(st.session_state.messages):

        with st.chat_message(
            message["role"]
        ):

            st.write(
                message["content"]
            )

            if message.get(
                "chart_file"
            ):

                chart_file = message["chart_file"]

                if os.path.exists(
                    chart_file
                ):

                    st.image(
                        chart_file,
                        caption="Generated Visualization"
                    )

                    with open(
                        chart_file,
                        "rb"
                    ) as file:

                        st.download_button(
                            label="Download Chart PNG",
                            data=file,
                            file_name=os.path.basename(
                                chart_file
                            ),
                            mime="image/png",
                            key=f"download_{i}"
                        )
    if user_question:

        st.session_state.messages.append({

            "role": "user",

            "content": user_question

        })

        with st.chat_message("user"):

            st.write(
                user_question
            )

        with st.spinner("Talk2Table AI is analyzing your data..."):

            result = run_talk2table_query(
                user_question,
                st.session_state.chat_history,
                df,
                TABLE_NAME,
                TABLE_NAME,
                conn
            )

        answer = result["final_answer"]

        chart_file = result.get(
            "chart_file",
            ""
        )

        st.session_state.messages.append({

            "role": "assistant",

            "content": answer,

            "chart_file": chart_file

        })

        with st.chat_message("assistant"):

            st.write(
                answer
            )

            if chart_file:

                if os.path.exists(
                    chart_file
                ):

                    st.image(
                        chart_file,
                        caption="Generated Visualization"
                    )

                    with open(
                        chart_file,
                        "rb"
                    ) as file:

                        st.download_button(
                            label="Download Chart PNG",
                            data=file,
                            file_name=os.path.basename(
                                chart_file
                            ),
                            mime="image/png",
                            key="download_latest"
                        )
else:

    st.info(
        "Please upload a CSV file to begin."
    )

        #what is the table all about
            
        #which manager has the highest revenue

        #what did he sell

        #show revenue by manager as a bar chart

        # show it as a pie chart

    #show it as a line chart
    #show it as a scatter chart



        # Show the pictorial representation of managers & their ratings and the product they sold
        #who is mj

    # git add app.py talk2table.py
# git commit -m "Preserve chart messages and visualization context"
 # git push